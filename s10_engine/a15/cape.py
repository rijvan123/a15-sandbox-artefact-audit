"""L1 report ingest (S7 Algorithm 1, steps 2-3; pipeline step 1).

Archives are streamed one member at a time, so neither the 479 GB of MELD-DS-448 nor the WinMET volumes are ever
unpacked to disk. Each report is split into a trace record (calls only), an environment record and a label record;
label-bearing and static fields never enter the trace record. Raw strings stay inside this module: the tokeniser
receives (name, value) pairs and returns classes only.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

EARLY_EXIT_CALLS = 100        # traces with fewer calls form the early-exit stratum (kept, flagged; S7 curation)


# ------------------------------------------------------------------------------------------------
# Archive streaming
# ------------------------------------------------------------------------------------------------
class ConcatReader(io.RawIOBase):
    """Seekable read-only view over split volumes (name.7z.001, .002, ...), without concatenating them on disk."""

    def __init__(self, parts: Iterable[Path]):
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total, self.pos, self._fh = sum(self.sizes), 0, {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.total + off}[whence]
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.total:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            fh = self._fh.setdefault(i, open(self.parts[i], "rb"))
            fh.seek(self.pos - self.starts[i])
            chunk = fh.read(min(n - done, self.sizes[i] - (self.pos - self.starts[i])))
            b[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done

    def close(self):
        for fh in self._fh.values():
            fh.close()
        super().close()


class _Sink:
    """py7zr writes each member into this object; the handler runs when the member is complete and its CRC passed."""

    def __init__(self, name: str, handler: Callable[[str, bytes], None]):
        self.name, self.handler, self.buf = name, handler, io.BytesIO()

    def write(self, d): return self.buf.write(d)
    def read(self, n=None): return self.buf.read() if n is None else self.buf.read(n)
    def seek(self, p): self.buf.seek(p)
    def flush(self): pass
    def close(self): pass
    def open(self, mode=None): return self
    def exists(self): return False
    def touch(self): pass
    def mkdir(self, parents=None, exist_ok=False): return None
    @property
    def parent(self): return self
    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:                               # only complete, CRC-checked members reach the handler
            data, self.buf = self.buf.getvalue(), io.BytesIO()
            self.handler(self.name, data)


def stream_7z(archive, handler: Callable[[str, bytes], None], password: str | None = None,
              keep: Callable[[str], bool] = lambda n: n.endswith(".json")) -> int:
    """Decompress sequentially and hand each kept member to `handler`; returns the number of members handled.

    Pinned to py7zr 0.22.0, whose worker accepts per-member file-like targets (register_filelike).
    py7zr keeps the decompressor of every finished folder (with its LZMA2 dictionary) on the Folder object until the
    archive is closed, so memory grows with every folder of a many-folder archive. Each finished folder's
    decompressor is released as soon as the next folder starts (amendment of 3 November 2026, after the Cowork ingest
    of I1 stopped twice in folder 54 of 61).
    """
    import gc

    import py7zr
    n = 0

    with py7zr.SevenZipFile(archive, "r", password=password) as z:
        mains = z.header.main_streams
        folders = list(mains.unpackinfo.folders) if mains is not None and mains.unpackinfo is not None else []
        position = {id(fo): k for k, fo in enumerate(folders)}
        released = [0]                                     # folders with an index below this value are released

        def counted(name, data, k):
            nonlocal n
            if k > released[0]:                            # folder k has started, so every earlier folder is done
                for fo in folders[released[0]:k]:
                    fo.decompressor = None
                released[0] = k
                gc.collect()
            n += 1
            handler(name, data)

        for f in z.files:
            if f.is_directory or not keep(f.filename):
                z.worker.register_filelike(f.id, None)     # decompressed but discarded
            else:
                k = position.get(id(f.folder), 0)
                z.worker.register_filelike(f.id, _Sink(f.filename, lambda name, data, k=k: counted(name, data, k)))
        z.worker.extract(z.fp, None, parallel=False)
    return n


def volumes(directory: Path, stem: str) -> list[Path]:
    return sorted(Path(directory).glob(f"{stem}.[0-9][0-9][0-9]"))


# ------------------------------------------------------------------------------------------------
# Report parsing
# ------------------------------------------------------------------------------------------------
def _ts(s: str) -> float:
    """CAPE call time 'YYYY-MM-DD HH:MM:SS,mmm' to seconds since the epoch (UTC; only differences are used)."""
    s = s.replace(",", ".")
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


@dataclass
class Report:
    installation: str
    sha256: str                        # sample SHA-256 (join key, eq. join rule)
    report_sha256: str                 # digest of the report bytes as received (d_x source)
    calls: list = field(default_factory=list)      # (t, proc_rank, call_id, api, args, ok, ret, n_rep)
    self_paths: tuple = ()             # the sample's own guest path(s), for the SELF path class
    env: dict = field(default_factory=dict)
    env_provenance: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    curation: dict = field(default_factory=dict)
    length: dict = field(default_factory=dict)
    time_keys: dict = field(default_factory=dict)


def pseudonym(value: str, key: bytes) -> str:
    return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:16]


def parse_report(raw: bytes, installation: str, key: bytes, name_hint: str = "") -> Report:
    r = json.loads(raw)
    target = r.get("target", {}) or {}
    tfile = target.get("file", {}) if isinstance(target, dict) else {}
    sha = (tfile.get("sha256") or os.path.splitext(os.path.basename(name_hint))[0]).lower()
    rep = Report(installation, sha, hashlib.sha256(raw).hexdigest())

    procs = (r.get("behavior") or {}).get("processes") or []
    # curation: MALVADA's documented criteria (get_incorrect_cape_reports.py, commit 2497402)
    rep.curation = {"fatal_no_target": int(not target), "no_processes": int(not procs),
                    "no_calls_main": int(bool(procs) and not procs[0].get("calls"))}

    guest = tfile.get("guest_paths") or []
    guest = [guest] if isinstance(guest, str) else list(guest)
    if procs and procs[0].get("module_path"):
        guest.append(procs[0]["module_path"])
    rep.self_paths = tuple(sorted({g for g in guest if isinstance(g, str)}))

    for rank, p in enumerate(procs):                       # every process; time order is restored below
        for c in p.get("calls", []):
            args = tuple((a.get("name", ""), a.get("value", "")) for a in c.get("arguments", []) or [])
            rep.calls.append((_ts(c.get("timestamp", "1970-01-01 00:00:00,000")), rank, c.get("id", 0),
                              c.get("api", ""), args, bool(c.get("status", False)), str(c.get("return", "")),
                              1 + int(c.get("repeated", 0) or 0)))
    rep.calls.sort(key=lambda x: (x[0], x[1], x[2]))       # call-time order across processes (eq. 2)

    # environment record (S7 evidence-item table); per-report fields where the report has them
    info = r.get("info") or {}
    machine = info.get("machine") or {}
    environ = procs[0].get("environ") if procs and isinstance(procs[0].get("environ"), dict) else {}
    rep.env = {
        "cape_version": info.get("version"),
        "guest_build": environ.get("ProductName") or environ.get("WindowsVersion"),
        "machine": pseudonym(str(machine.get("label") or machine.get("name") or environ.get("ComputerName")), key)
        if (machine or environ.get("ComputerName")) else None,
        "timeout": info.get("timeout"),
        "duration": info.get("duration"),
        "route": info.get("route"),
        "analysis_start": info.get("started") or (procs[0].get("first_seen") if procs else None),
    }
    rep.env_provenance = {k: ("report" if v is not None else "missing") for k, v in rep.env.items()}

    # label record, kept apart from the trace (never a model input)
    det = r.get("detections")
    rep.labels = {"cape": sorted({d.get("family", "") for d in det if isinstance(d, dict)}) if isinstance(det, list)
                  else ([det] if isinstance(det, str) else []),
                  "avclass": r.get("avclass_detection") or "", "malfamily": r.get("malfamily") or ""}

    # time keys for the WinMET field audit: VirusTotal v2 scan_id ends in the scan time (Unix seconds)
    vt = tfile.get("virustotal") if isinstance(tfile, dict) else None
    scan = None
    if isinstance(vt, dict) and isinstance(vt.get("scan_id"), str) and "-" in vt["scan_id"]:
        tail = vt["scan_id"].rsplit("-", 1)[-1]
        scan = datetime.fromtimestamp(int(tail), tz=timezone.utc).date().isoformat() if tail.isdigit() else None
    rep.time_keys = {"vt_scan_date": scan, "analysis_date": (rep.env["analysis_start"] or "")[:10] or None}

    n_calls = sum(c[7] for c in rep.calls)
    rep.length = {"n_calls": n_calls, "n_apis": len({c[3] for c in rep.calls}), "n_procs": len(procs),
                  "duration_s": (rep.calls[-1][0] - rep.calls[0][0]) if rep.calls else 0.0,
                  "early_exit": int(n_calls < EARLY_EXIT_CALLS)}
    return rep


def curated_out(rep: Report) -> bool:
    """True if MALVADA's documented filter removes the report; applied identically to both installations."""
    return bool(rep.curation["fatal_no_target"] or rep.curation["no_processes"] or rep.curation["no_calls_main"])
