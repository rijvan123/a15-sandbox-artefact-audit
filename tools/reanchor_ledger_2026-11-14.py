"""A15 ledger: trusted re-anchor after an anchor from the local development TSA (14 Nov 2026).
The untrusted anchor stays (append-only). One `reanchor` entry is added and anchored with the external TSA; every
entry's hash includes the previous one, so this trusted time-stamp also covers all earlier entries and S10 results."""
import json, socket, sys
from pathlib import Path
from urllib.parse import urlparse
root, tsa = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s10_engine"))
import run_s9 as R9
from a15 import config as C
socket.getaddrinfo(urlparse(tsa).hostname, 443)        # stops here, writing nothing, if the TSA name does not resolve
led = R9.open_ledger(C.Paths(Path(root)), tsa)
ok, msg = led.verify()
last = [r["entry"] for r in led.records if r["entry"]["action"] == "anchor"][-1]
if not ok or last["trusted"]:
    sys.exit(f"nothing written: verify {ok} {msg}; last anchor {last['i']} trusted {last['trusted']}")
res = [r["entry"] for r in led.records if r["entry"]["action"] == "results"][-1]
led.append("reanchor", evidence_id="S10", note=f"trusted re-anchor: anchor {last['i']} came from the local development TSA",
           payload={"untrusted_anchor": last["i"], "batch": [last["batch_start"], last["batch_end"]],
                    "reason": last.get("pending"), "results_entry": res["i"], "results_d_f": res["d_f"]})
e = led.anchor()["entry"]
print(json.dumps({"reanchor_entry": e["batch_start"], "anchor": e["i"], "trusted": e["trusted"], "tsa_time": e["tsa_time"],
                  "verify": led.verify()}))
if not e["trusted"]:
    print("NOT TRUSTED: the time-stamp server did not answer; run the same command again when the internet works")
