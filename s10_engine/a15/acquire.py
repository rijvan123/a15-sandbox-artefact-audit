"""L1 acquisition with provenance (S7 Algorithm 1, step 1; S7 failure table).

Every file is downloaded from its official host at a pinned revision, hashed (SHA-256), compared with the digest the
host publishes (Hugging Face LFS SHA-256, Zenodo MD5), logged in the Data Provenance Log and sealed in the ledger.
A mismatch quarantines the file and retries up to three times; a blocked host is logged as a substitution.
No binaries are ever requested: the skip patterns exclude disassembly and every other non-report artefact.
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SOURCES, Source
from .ledger import Ledger, utc_now

PROV_FIELDS = ("source", "dataset", "file", "url_or_doi", "version", "licence", "size_bytes", "sha256",
               "published_digest", "published_algo", "digest_match", "access_utc", "status", "note")


@dataclass
class ProvRow:
    source: str
    dataset: str
    file: str
    url_or_doi: str
    version: str
    licence: str
    size_bytes: int = 0
    sha256: str = ""
    published_digest: str = ""
    published_algo: str = ""
    digest_match: str = ""
    access_utc: str = ""
    status: str = ""
    note: str = ""


class ProvenanceLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(PROV_FIELDS)

    def add(self, row: ProvRow) -> None:
        row.access_utc = row.access_utc or utc_now()
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([asdict(row)[k] for k in PROV_FIELDS])


def probe(url: str, timeout: int = 15) -> dict:
    """Reachability probe; records the egress proxy's deny reason when a sandbox blocks the host."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            return {"url": url, "reachable": True, "status": r.status, "deny_reason": ""}
    except urllib.error.HTTPError as e:
        return {"url": url, "reachable": e.code < 500 and e.code != 403, "status": e.code,
                "deny_reason": e.headers.get("x-deny-reason", "")}
    except Exception as e:
        return {"url": url, "reachable": False, "status": 0, "deny_reason": f"{type(e).__name__}: {e}"[:160]}


def disk_ok(path: Path, need_bytes: int, margin: float = 1.25) -> tuple[bool, int]:
    free = shutil.disk_usage(path).free
    return free >= need_bytes * margin, free


def _hashes(path: Path) -> tuple[str, str]:
    s, m = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            s.update(b)
            m.update(b)
    return s.hexdigest(), m.hexdigest()


def _keep(name: str, src: Source) -> bool:
    if any(fnmatch.fnmatch(name, p) for p in src.skip):
        return False
    return any(fnmatch.fnmatch(name, p) for p in src.files)


# ------------------------------------------------------------------------------------------------
# Listings: what the host publishes (path, size, digest), before anything is downloaded
# ------------------------------------------------------------------------------------------------
def list_huggingface(src: Source) -> list[dict]:
    from huggingface_hub import HfApi                      # imported lazily: not needed in the sandbox tests
    api = HfApi()
    out = []
    for item in api.list_repo_tree(src.locator, repo_type="dataset", revision=src.revision or None,
                                   recursive=True, expand=True):
        if not hasattr(item, "size"):
            continue                                       # folders
        lfs = getattr(item, "lfs", None)
        out.append({"path": item.path, "size": item.size, "digest": getattr(lfs, "sha256", "") if lfs else "",
                    "algo": "sha256" if lfs else "", "keep": _keep(item.path, src)})
    return out


def list_zenodo(src: Source) -> tuple[dict, list[dict]]:
    url = f"https://zenodo.org/api/records/{src.locator}/versions/latest"
    with urllib.request.urlopen(url, timeout=60) as r:
        rec = json.load(r)
    files = []
    for f in rec.get("files", []):
        algo, _, dig = f.get("checksum", ":").partition(":")
        files.append({"path": f["key"], "size": f["size"], "digest": dig, "algo": algo,
                      "url": f["links"]["self"], "keep": _keep(f["key"], src)})
    meta = {"record_id": rec.get("id"), "version": rec.get("metadata", {}).get("version", ""),
            "doi": rec.get("doi", ""), "licence": rec.get("metadata", {}).get("license", {}).get("id", "")}
    return meta, files


# ------------------------------------------------------------------------------------------------
# Download, verify, log and seal one file (Algorithm 1, step 1)
# ------------------------------------------------------------------------------------------------
def fetch_verified(src: Source, item: dict, dest_dir: Path, prov: ProvenanceLog, ledger: Ledger,
                   version: str, retries: int = 3) -> Path | None:
    dest = Path(dest_dir) / src.code / item["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        if src.host == "huggingface":
            from huggingface_hub import hf_hub_download
            p = Path(hf_hub_download(src.locator, item["path"], repo_type="dataset", revision=src.revision or None,
                                     local_dir=str(Path(dest_dir) / src.code)))
            dest = p
        else:                                              # Zenodo and other plain HTTPS hosts
            with urllib.request.urlopen(item["url"], timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
        sha, md5 = _hashes(dest)
        published = item.get("digest", "")
        got = {"sha256": sha, "md5": md5}.get(item.get("algo", ""), "")
        match = (got == published) if published else None
        row = ProvRow(src.code, src.name, item["path"], f"{src.host}:{src.locator}", version, src.licence,
                      dest.stat().st_size, sha, published, item.get("algo", ""),
                      "" if match is None else str(match), status="ok" if match in (True, None) else "quarantine",
                      note=f"attempt {attempt}")
        prov.add(row)
        if match in (True, None):
            ledger.append("acquire", evidence_id=f"{src.code}:{item['path']}", d_f=sha,
                          source=src.code, version=version, published_match="" if match is None else int(match))
            return dest
        ledger.append("quarantine", evidence_id=f"{src.code}:{item['path']}", d_f=sha, attempt=attempt)
        quarantine = dest.parent / "quarantine"
        quarantine.mkdir(exist_ok=True)
        shutil.move(str(dest), quarantine / f"{dest.name}.attempt{attempt}")
    ledger.append("exclude_volume", evidence_id=f"{src.code}:{item['path']}", reason="digest mismatch x3")
    return None


def git_pinned(src: Source, dest_dir: Path, prov: ProvenanceLog, ledger: Ledger) -> Path:
    dest = Path(dest_dir) / src.code
    if not dest.exists():
        subprocess.run(["git", "clone", "--quiet", src.locator, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", src.revision], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    prov.add(ProvRow(src.code, src.name, "(git tree)", src.locator, head, src.licence, status="ok",
                     digest_match=str(head == src.revision), note="git commit is the content digest"))
    ledger.append("acquire", evidence_id=f"{src.code}:git", d_f=head, source=src.code)
    return dest


def record_blocked(src: Source, prov: ProvenanceLog, ledger: Ledger, probe_result: dict, where: str) -> None:
    """Log a host that cannot be reached from this environment as a substitution, never as a licence failure."""
    prov.add(ProvRow(src.code, src.name, "(all)", f"{src.host}:{src.locator}", src.revision, src.licence,
                     status="blocked_here", note=f"{where}: {probe_result.get('deny_reason') or probe_result.get('status')}"))
    ledger.append("substitution", evidence_id=f"{src.code}:(all)", reason="host unreachable from " + where,
                  probe=probe_result, k2_triggered=0)


def acquire_all(root: Path, prov: ProvenanceLog, ledger: Ledger, where: str, include=("I1", "I2")) -> dict:
    """Cowork entry point: list, check disk, then download the kept files of each primary source."""
    status = {}
    for code in include:
        src = SOURCES[code]
        if src.host == "huggingface":
            listing, version = list_huggingface(src), src.revision
        elif src.host == "zenodo":
            meta, listing = list_zenodo(src)
            version = f"record {meta['record_id']} v{meta['version']}"
            ledger.append("resolve_version", evidence_id=f"{code}:record", payload=meta)
        else:
            continue
        kept = [x for x in listing if x["keep"]]
        need = sum(x["size"] for x in kept)
        ok, free = disk_ok(root, need)
        ledger.append("disk_check", evidence_id=f"{code}:(kept)", need_bytes=need, free_bytes=free, ok=int(ok))
        for x in listing:                                  # every skipped file is listed, never fetched
            if not x["keep"]:
                prov.add(ProvRow(code, src.name, x["path"], f"{src.host}:{src.locator}", version, src.licence,
                                 x["size"], "", x["digest"], x["algo"], status="skipped_by_design"))
        if not ok:
            status[code] = f"insufficient disk: need {need}, free {free}"
            continue
        got = [fetch_verified(src, x, root, prov, ledger, version) for x in kept]
        status[code] = f"{sum(g is not None for g in got)}/{len(kept)} files verified"
    return status
