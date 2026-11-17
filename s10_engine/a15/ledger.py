"""Evidence ledger (S7 layer L2; S8 equations 25-28).

h(0) = H(enc(study id)); h(i) = H(enc(entry_i) || h(i-1)); each batch of chain hashes gets an RFC 6962 Merkle root
R_b, and an RFC 3161 time-stamp token over R_b is written in the anchor entry that opens the next batch.
The encoding and Merkle functions are byte-for-byte those verified by check C23 at S8.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def H(b: bytes) -> bytes:                                   # SHA-256, the only hash used by the ledger
    return hashlib.sha256(b).digest()


def enc(entry: dict) -> bytes:                              # canonical, injective: sorted keys, length-prefixed
    out = bytearray()
    for key in sorted(entry):
        kb, vb = key.encode(), str(entry[key]).encode()
        out += len(kb).to_bytes(4, "big") + kb + len(vb).to_bytes(8, "big") + vb
    return bytes(out)


def leaf_h(d: bytes) -> bytes:                              # RFC 6962 leaf hash, prefix 0x00
    return H(b"\x00" + d)


def node_h(left: bytes, right: bytes) -> bytes:             # RFC 6962 interior hash, prefix 0x01
    return H(b"\x01" + left + right)


def mth(ds: list[bytes]) -> bytes:                          # Merkle tree hash, eq. (26)
    if len(ds) == 1:
        return leaf_h(ds[0])
    n_l = 1 << ((len(ds) - 1).bit_length() - 1)            # largest power of two below n
    return node_h(mth(ds[:n_l]), mth(ds[n_l:]))


def audit_path(m: int, ds: list[bytes]) -> list[bytes]:     # inclusion proof, at most ceil(log2 n) hashes
    if len(ds) == 1:
        return []
    n_l = 1 << ((len(ds) - 1).bit_length() - 1)
    if m < n_l:
        return audit_path(m, ds[:n_l]) + [mth(ds[n_l:])]
    return audit_path(m - n_l, ds[n_l:]) + [mth(ds[:n_l])]


def verify_inclusion(m: int, n: int, d: bytes, proof: list[bytes], root: bytes) -> bool:   # RFC 9162 2.1.3.2
    if m >= n:
        return False
    fn, sn, r = m, n - 1, leaf_h(d)
    for p in proof:
        if sn == 0:
            return False
        if (fn & 1) or fn == sn:
            r = node_h(p, r)
            while not (fn & 1) and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = node_h(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def utc_now() -> str:                                       # self-asserted clock; only the TSA token is external
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# RFC 3161 time-stamping through the OpenSSL command line (no Python TSA library needed)
# ------------------------------------------------------------------------------------------------
class TimeStamper:
    """External TSA when reachable; otherwise a local development TSA whose tokens are labelled as such.

    The message imprint is SHA-256(R_b), as in TST_b = Sign_TSA(H(R_b), T_b), eq. (26).
    """

    def __init__(self, workdir: Path, tsa_url: str | None = None, ca_file: str | None = None,
                 untrusted: str | None = None, timeout: int = 30):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tsa_url, self.ca_file, self.untrusted, self.timeout = tsa_url, ca_file, untrusted, timeout

    def _run(self, args: list[str], stdin: bytes | None = None) -> bytes:
        return subprocess.run(args, input=stdin, capture_output=True, check=True).stdout

    def _local_tsa(self) -> tuple[str, str, str, str]:      # self-signed CA + TSA certificate, development only
        ca_key, ca_crt = self.dir / "devca.key", self.dir / "devca.crt"
        tsa_key, tsa_crt, cnf = self.dir / "devtsa.key", self.dir / "devtsa.crt", self.dir / "devtsa.cnf"
        if not tsa_crt.exists():
            cnf.write_text(
                "[ tsa ]\ndefault_tsa = tsa_config1\n[ tsa_config1 ]\n"
                f"dir = {self.dir}\nserial = $dir/tsaserial\ncrypto_device = builtin\n"
                f"signer_cert = {tsa_crt}\nsigner_key = {tsa_key}\nsigner_digest = sha256\n"
                "default_policy = 1.3.6.1.4.1.99999.1.1\ndigests = sha256\naccuracy = secs:1\n"
                "ordering = yes\ntsa_name = no\ness_cert_id_chain = no\ness_cert_id_alg = sha256\n"
                "[ v3_tsa ]\nbasicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature\n"
                "extendedKeyUsage=critical,timeStamping\n")
            (self.dir / "tsaserial").write_text("01\n")
            self._run(["openssl", "req", "-x509", "-newkey", "rsa:3072", "-nodes", "-keyout", str(ca_key),
                       "-out", str(ca_crt), "-days", "3650", "-subj", "/CN=A15 LOCAL DEVELOPMENT CA (NOT TRUSTED)"])
            self._run(["openssl", "req", "-newkey", "rsa:3072", "-nodes", "-keyout", str(tsa_key),
                       "-out", str(self.dir / "devtsa.csr"), "-subj", "/CN=A15 LOCAL DEVELOPMENT TSA (NOT TRUSTED)"])
            self._run(["openssl", "x509", "-req", "-in", str(self.dir / "devtsa.csr"), "-CA", str(ca_crt),
                       "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(tsa_crt), "-days", "3650",
                       "-extfile", str(cnf), "-extensions", "v3_tsa"])
        return str(ca_crt), str(tsa_key), str(tsa_crt), str(cnf)

    def stamp(self, root: bytes) -> dict:
        data = self.dir / f"root_{root.hex()[:16]}.bin"
        data.write_bytes(root)
        tsq = data.with_suffix(".tsq")
        self._run(["openssl", "ts", "-query", "-data", str(data), "-sha256", "-cert", "-out", str(tsq)])
        if self.tsa_url:
            try:
                req = urllib.request.Request(self.tsa_url, data=tsq.read_bytes(),
                                             headers={"Content-Type": "application/timestamp-query"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    tsr = resp.read()
                return {"tsa": self.tsa_url, "trusted": True, "token_b64": base64.b64encode(tsr).decode()}
            except Exception as exc:                        # unreachable TSA: pending anchor (S7 fallback table)
                pending_reason = f"{type(exc).__name__}: {exc}"[:200]
        else:
            pending_reason = "no external TSA configured"
        ca, key, crt, cnf = self._local_tsa()
        tsr_path = data.with_suffix(".tsr")
        self._run(["openssl", "ts", "-reply", "-config", cnf, "-section", "tsa_config1", "-queryfile", str(tsq),
                   "-inkey", key, "-signer", crt, "-out", str(tsr_path)])
        return {"tsa": "LOCAL-DEV-TSA", "trusted": False, "pending_reason": pending_reason,
                "token_b64": base64.b64encode(tsr_path.read_bytes()).decode()}

    def verify(self, root: bytes, token: dict) -> bool:
        data, tsr = self.dir / "verify_root.bin", self.dir / "verify.tsr"
        data.write_bytes(root)
        tsr.write_bytes(base64.b64decode(token["token_b64"]))
        if token["tsa"] == "LOCAL-DEV-TSA":
            ca, _, crt, _ = self._local_tsa()
            untrusted = crt
        else:
            if not self.ca_file:
                return False
            ca, untrusted = self.ca_file, self.untrusted
        args = ["openssl", "ts", "-verify", "-data", str(data), "-in", str(tsr), "-CAfile", ca]
        if untrusted:
            args += ["-untrusted", untrusted]
        return subprocess.run(args, capture_output=True).returncode == 0

    @staticmethod
    def gen_time(token: dict) -> str:                      # TSA time T_b read from the token itself
        with tempfile.NamedTemporaryFile(suffix=".tsr", delete=False) as f:
            f.write(base64.b64decode(token["token_b64"]))
        out = subprocess.run(["openssl", "ts", "-reply", "-in", f.name, "-text"], capture_output=True, text=True)
        os.unlink(f.name)
        for line in out.stdout.splitlines():
            if line.strip().startswith("Time stamp:"):
                return line.split(":", 1)[1].strip()
        return ""


# ------------------------------------------------------------------------------------------------
# The ledger itself: append-only JSON lines, one object {"entry": {...}, "h": hex} per line
# ------------------------------------------------------------------------------------------------
class Ledger:
    def __init__(self, path: Path, study_id: str, stamper: TimeStamper, custodian_key: bytes | None = None):
        self.path, self.stamper = Path(path), stamper
        self.h0 = H(enc({"study_identifier": study_id}))
        key = custodian_key or os.environ.get("A15_CUSTODIAN_KEY", "").encode()
        if not key:                                        # development key; the custodian sets a real one
            key = H(b"A15-development-custodian-key")
        self._key = key
        self.records: list[dict] = []
        if self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    # -- writing ------------------------------------------------------------------------------
    def pseudonym(self, actor: str) -> str:                 # HMAC-SHA-256 pseudonym of the actor (GDPR Art. 4(5))
        return hmac.new(self._key, actor.encode(), hashlib.sha256).hexdigest()[:32]

    def last_hash(self) -> bytes:
        return bytes.fromhex(self.records[-1]["h"]) if self.records else self.h0

    def append(self, action: str, actor: str = "pipeline", **fields) -> dict:
        prev = self.last_hash()
        entry = {"i": len(self.records) + 1, "action": action, "actor": self.pseudonym(actor),
                 "tau": utc_now(), "prev": prev.hex()}
        for k, v in fields.items():                        # digests, evidence IDs, payload digests, notes
            entry[k] = v if isinstance(v, (str, int, float)) else json.dumps(v, sort_keys=True)
        h = H(enc(entry) + prev)                           # eq. (25)
        rec = {"entry": entry, "h": h.hex()}
        with open(self.path, "a", encoding="utf-8") as f:   # append-only; fsync before returning
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(rec)
        return rec

    def open_batch(self) -> tuple[int, int]:               # first and last index after the last anchor
        start = 1
        for rec in self.records:
            if rec["entry"]["action"] == "anchor":
                start = rec["entry"]["i"] + 1
        return start, len(self.records)

    def anchor(self) -> dict | None:
        start, end = self.open_batch()
        if end < start:
            return None
        hs = [bytes.fromhex(r["h"]) for r in self.records[start - 1:end]]
        root = mth(hs)
        token = self.stamper.stamp(root)
        rec = self.append("anchor", batch_start=start, batch_end=end, root=root.hex(), tsa=token["tsa"],
                          trusted=int(token["trusted"]), token_b64=token["token_b64"],
                          tsa_time=TimeStamper.gen_time(token), pending=token.get("pending_reason", ""))
        sealed = self.path.parent / f"batch_{start:08d}_{end:08d}.jsonl"   # read-only sealed copy (WORM emulation)
        sealed.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in self.records[start - 1:end]))
        os.chmod(sealed, 0o444)
        return rec

    # -- verification -------------------------------------------------------------------------
    def verify(self, check_tokens: bool = True) -> tuple[bool, str]:
        prev = self.h0
        for rec in self.records:                           # recompute h(1) ... h(n)
            e = rec["entry"]
            if e["prev"] != prev.hex() or H(enc(e) + prev).hex() != rec["h"]:
                return False, f"chain breaks at entry {e['i']}"
            prev = bytes.fromhex(rec["h"])
        for rec in self.records:                           # recompute every anchored root and its token
            e = rec["entry"]
            if e["action"] != "anchor":
                continue
            hs = [bytes.fromhex(r["h"]) for r in self.records[e["batch_start"] - 1:e["batch_end"]]]
            if mth(hs).hex() != e["root"]:
                return False, f"root mismatch for batch ending at {e['batch_end']}"
            if check_tokens and not self.stamper.verify(bytes.fromhex(e["root"]),
                                                        {"tsa": e["tsa"], "token_b64": e["token_b64"]}):
                return False, f"token fails for batch ending at {e['batch_end']}"
        return True, "ok"

    def v_frz(self, freeze_action: str = "freeze", scoring_action: str = "test_score") -> int:
        """Eq. (28): 1 iff the ledger verifies and the anchor of the freeze batch precedes the first test score."""
        ok, _ = self.verify()
        i_frz = next((r["entry"]["i"] for r in self.records if r["entry"]["action"] == freeze_action), None)
        i_1 = next((r["entry"]["i"] for r in self.records if r["entry"]["action"] == scoring_action), None)
        if not ok or i_frz is None:
            return 0
        i_anc = next((r["entry"]["i"] for r in self.records
                      if r["entry"]["action"] == "anchor" and r["entry"]["batch_start"] <= i_frz <= r["entry"]["batch_end"]),
                     None)
        if i_anc is None:
            return 0
        return int(i_1 is None or i_anc < i_1)

    def inclusion_proof(self, i: int) -> dict:
        anc = next(r["entry"] for r in self.records
                   if r["entry"]["action"] == "anchor" and r["entry"]["batch_start"] <= i <= r["entry"]["batch_end"])
        hs = [bytes.fromhex(r["h"]) for r in self.records[anc["batch_start"] - 1:anc["batch_end"]]]
        m = i - anc["batch_start"]
        return {"index": i, "leaf": hs[m].hex(), "proof": [p.hex() for p in audit_path(m, hs)],
                "root": anc["root"], "anchor_index": anc["i"], "batch_size": len(hs)}


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
