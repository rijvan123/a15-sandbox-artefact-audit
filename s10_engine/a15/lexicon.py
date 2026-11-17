"""Frozen lexicon c(t) in {B, F, P} (S7 Algorithm 2; S8 equation 23).

Rules are semantic and documented: P from MITRE ATT&CK T1497 (.001 system checks, .002 user-activity checks,
.003 time-based evasion); F from values the analysis environment sets (CAPE sleep skipping, the guest agent, CAPE
artefacts, the out-of-set token, route-dependent network outcomes). No rule is mined from differences between
installations: that would make prediction P3 (J_w(F) < J_w(B)) true by construction. The lexicon file is sealed in
the ledger before J_w is computed, so the P3 result cannot have shaped it.
Boundary rules carry a primary class and an alternative class for the sensitivity analysis.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .config import OOS

USER_ACTIVITY_APIS = {"GetCursorPos", "GetLastInputInfo", "GetForegroundWindow", "GetKeyboardLayout"}
TIMING_APIS = {"GetTickCount", "GetTickCount64", "QueryPerformanceCounter", "NtQueryPerformanceCounter",
               "NtQuerySystemTime", "GetSystemTimeAsFileTime", "timeGetTime", "GetLocalTime", "GetSystemTime"}
SYSTEM_CHECK_APIS = {"GetSystemFirmwareTable", "NtQuerySystemInformation", "GlobalMemoryStatusEx",
                     "GetDiskFreeSpaceExW", "GetDiskFreeSpaceExA", "GetDiskFreeSpaceW", "GetDiskFreeSpaceA",
                     "DeviceIoControl", "NtDeviceIoControlFile", "GetAdaptersInfo", "GetAdaptersAddresses",
                     "EnumDisplayDevicesW", "EnumDisplayDevicesA", "SetupDiGetDeviceRegistryPropertyW",
                     "SetupDiGetDeviceRegistryPropertyA", "GetSystemInfo", "GetNativeSystemInfo",
                     "NtQueryVolumeInformationFile", "GetVolumeInformationW", "GetVolumeInformationA"}
DEBUGGER_APIS = {"IsDebuggerPresent", "CheckRemoteDebuggerPresent", "OutputDebugStringA", "OutputDebugStringW",
                 "NtQueryInformationProcess", "NtSetInformationThread"}            # ATT&CK T1622, not T1497
RECON_APIS = {"GetComputerNameW", "GetComputerNameA", "GetComputerNameExW", "GetComputerNameExA", "GetUserNameW",
              "GetUserNameA", "GetUserNameExW", "GetUserNameExA", "NetGetJoinInformation", "LookupAccountSidW"}
NETWORK_APIS = {"connect", "WSAConnect", "send", "recv", "sendto", "recvfrom", "getaddrinfo", "GetAddrInfoW",
                "gethostbyname", "DnsQuery_A", "DnsQuery_W", "DnsQueryEx", "InternetOpenUrlA", "InternetOpenUrlW",
                "InternetConnectA", "InternetConnectW", "HttpSendRequestA", "HttpSendRequestW", "WinHttpConnect",
                "WinHttpSendRequest", "WinHttpReceiveResponse", "URLDownloadToFileW", "URLDownloadToFileA",
                "InternetReadFile", "WSASend", "WSARecv", "NtDeviceIoControlFile_AFD"}


@dataclass(frozen=True)
class Rule:
    rid: str
    cls: str            # primary class
    alt: str            # class in the boundary sensitivity analysis ('' if not a boundary rule)
    source: str         # documentation the rule derives from
    test: str           # human-readable condition (the code below is authoritative)


RULES = (
    Rule("F0", "F", "B", "S8 eq. 23", "token is <OOS>: instrumentation decides which calls fall outside H_c"),
    Rule("F1", "F", "", "CAPEv2 sleep skipping", "an argument class is SLEEP_SKIPPED"),
    Rule("F2", "F", "", "CAPEv2 guest agent", "a process-name class is AGENT"),
    Rule("F3", "F", "P", "CAPEv2 monitor and analyzer", "a path class is CAPE_ARTEFACT"),
    Rule("F4", "F", "", "S7 privacy and route", "a network API with an IP class PRIVATE or LINK_LOCAL"),
    Rule("F5", "F", "B", "S7 route harmonisation", "a network API whose return class is not OK"),
    Rule("P1", "P", "", "ATT&CK T1497.001", "registry class VM_ARTEFACT, HARDWARE_DESC or DISK_ENUM"),
    Rule("P2", "P", "", "ATT&CK T1497.001", "value-name class VM_ARTEFACT or BIOS_INFO"),
    Rule("P3", "P", "", "ATT&CK T1497.001", "path class VM_ARTEFACT"),
    Rule("P4", "P", "", "ATT&CK T1497.001", "process-name class ANALYSIS_TOOL"),
    Rule("P5", "P", "B", "ATT&CK T1497.002", "user-activity API (cursor, last input, foreground window)"),
    Rule("P6", "P", "", "ATT&CK T1497.003", "sleep class LT60S, LT10MIN, GE10MIN or INFINITE (not skipped)"),
    Rule("P7", "P", "B", "ATT&CK T1497.001", "system-information API used for hardware or firmware checks"),
    Rule("P8", "B", "P", "ATT&CK T1497.003", "timing API (boundary; primary B because it is ubiquitous)"),
    Rule("P9", "B", "P", "ATT&CK T1622", "debugger check (boundary; not a T1497 technique)"),
    Rule("P10", "B", "P", "ATT&CK T1082/T1033", "computer or user name query (boundary)"),
    Rule("B0", "B", "", "default", "any other token"),
)
RULE_BY_ID = {r.rid: r for r in RULES}
LEXICON_VERSION = "A15-lexicon-v1 (seed rules, S9)"


def _parts(tok: str) -> tuple[str, list[str], str]:
    if tok == OOS or "|" not in tok:
        return tok, [], ""
    api, kap, ret = tok.split("|")
    return api, [c.split(":", 1)[1] for c in kap.split(";") if c], ret


def classify(tok: str) -> tuple[str, str]:
    """c(t) and the id of the rule that fired, from the full token (before vocabulary lookup, S8 Section 1)."""
    if tok == OOS:
        return "F", "F0"
    api, classes, ret = _parts(tok)
    cs = set(classes)
    if "SLEEP_SKIPPED" in cs:
        return "F", "F1"
    if "proc=AGENT" in cs:
        return "F", "F2"
    if "path=CAPE_ARTEFACT" in cs:
        return "F", "F3"
    if api in NETWORK_APIS and ({"ip=PRIVATE", "ip=LINK_LOCAL"} & cs):
        return "F", "F4"
    if api in NETWORK_APIS and ret != "OK":
        return "F", "F5"
    if any(c.endswith((":VM_ARTEFACT", ":HARDWARE_DESC", ":DISK_ENUM")) and c.startswith("reg=") for c in cs):
        return "P", "P1"
    if {"val=VM_ARTEFACT", "val=BIOS_INFO"} & cs:
        return "P", "P2"
    if "path=VM_ARTEFACT" in cs:
        return "P", "P3"
    if "proc=ANALYSIS_TOOL" in cs:
        return "P", "P4"
    if api in USER_ACTIVITY_APIS:
        return "P", "P5"
    if {"sleep=LT60S", "sleep=LT10MIN", "sleep=GE10MIN", "sleep=INFINITE"} & cs and api not in NETWORK_APIS:
        return "P", "P6"
    if api in SYSTEM_CHECK_APIS:
        return "P", "P7"
    if api in TIMING_APIS:
        return "B", "P8"
    if api in DEBUGGER_APIS:
        return "B", "P9"
    if api in RECON_APIS:
        return "B", "P10"
    return "B", "B0"


def classify_alt(tok: str) -> str:
    """Class under the boundary sensitivity reading (alternative class where the rule has one)."""
    cls, rid = classify(tok)
    return RULE_BY_ID[rid].alt or cls


def lexicon_document(observed_tokens: list[str] | None = None) -> dict:
    """The file that is hashed, sealed and registered: rules, version and the token-level table (embargoed)."""
    doc = {"version": LEXICON_VERSION, "rules": [asdict(r) for r in RULES],
           "classes": {"B": "behaviour", "F": "footprint (values the environment sets)",
                       "P": "probing (ATT&CK T1497 checks by the malware)"},
           "boundary_rules": [r.rid for r in RULES if r.alt]}
    if observed_tokens is not None:
        doc["token_table"] = {t: classify(t) for t in sorted(set(observed_tokens))}
    return doc


def digest(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
