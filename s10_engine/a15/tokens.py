"""L3 class tokenisation (S8 equations 1-2; S7 pipeline steps 2-4; privacy plane).

A call becomes t = (API, argument classes, return class). Classes come from closed enumerations, so no raw string,
handle, address, process ID or time stamp can reach a token (unit test: tests/test_tokens.py). Anonymised values
(MALVADA writes [REDACTED]) and raw values map to the same class: user names never enter a path class.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter

from .config import BETA_MAX, L_CAP, OOS, UNK, VOCAB_MIN_TRACES

# --- argument names that are classified; every other argument (handles, buffers, addresses, sizes) is dropped
PATH_ARGS = {"filename", "filepath", "objectattributes", "objectattributesname", "pathname", "directoryname",
             "existingfilename", "newfilename", "modulename", "dllname", "lpfilename", "currentdirectory",
             "imagepathname", "fullname", "targetfilename", "dirname", "path", "handlename", "name"}
REG_ARGS = {"fullname", "registry", "subkey", "keyname", "registrykey", "objectattributesname", "handlename",
            "name"}
VALUE_ARGS = {"valuename"}
NET_ARGS = {"ip", "ipaddress", "address", "servername", "hostname", "url", "host", "nodename", "domain"}
SLEEP_ARGS = {"milliseconds", "interval", "dwmilliseconds", "timeout"}
PROC_ARGS = {"processname", "commandline", "imagename", "applicationname", "lpcommandline", "lpapplicationname"}

SYSTEM_PROCS = {"svchost.exe", "explorer.exe", "lsass.exe", "winlogon.exe", "services.exe", "csrss.exe", "smss.exe",
                "wininit.exe", "spoolsv.exe", "taskhostw.exe", "dllhost.exe", "rundll32.exe", "regsvr32.exe",
                "conhost.exe", "wmiprvse.exe", "searchindexer.exe", "sihost.exe", "ctfmon.exe", "msiexec.exe",
                "werfault.exe", "runtimebroker.exe", "audiodg.exe", "fontdrvhost.exe", "dwm.exe"}
SHELL_TOOLS = {"cmd.exe", "powershell.exe", "pwsh.exe", "wscript.exe", "cscript.exe", "mshta.exe", "schtasks.exe",
               "reg.exe", "sc.exe", "net.exe", "net1.exe", "netsh.exe", "wmic.exe", "bitsadmin.exe", "certutil.exe",
               "vssadmin.exe", "bcdedit.exe", "taskkill.exe", "timeout.exe", "ping.exe", "attrib.exe", "icacls.exe",
               "installutil.exe", "regasm.exe", "regsvcs.exe", "msbuild.exe", "aspnet_compiler.exe", "cmstp.exe"}
BROWSERS = {"chrome.exe", "firefox.exe", "msedge.exe", "iexplore.exe", "opera.exe", "brave.exe", "vivaldi.exe"}
OFFICE = {"winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe", "onenote.exe"}
ANALYSIS_TOOLS = {"vboxservice.exe", "vboxtray.exe", "vmtoolsd.exe", "vmwaretray.exe", "vmwareuser.exe",
                  "vmsrvc.exe", "vmusrvc.exe", "qemu-ga.exe", "xenservice.exe", "prl_tools.exe", "procmon.exe",
                  "procmon64.exe", "procexp.exe", "procexp64.exe", "wireshark.exe", "fiddler.exe", "x64dbg.exe",
                  "x32dbg.exe", "ollydbg.exe", "ida.exe", "ida64.exe", "idaq.exe", "processhacker.exe",
                  "tcpview.exe", "autoruns.exe", "sysmon.exe", "sysmon64.exe", "dumpcap.exe", "joeboxserver.exe",
                  "sandboxiedcomlaunch.exe", "windbg.exe", "immunitydebugger.exe", "regmon.exe", "filemon.exe"}
SECURITY = {"msmpeng.exe", "mpcmdrun.exe", "nissrv.exe", "securityhealthservice.exe", "avp.exe", "mbam.exe",
            "ekrn.exe", "avgnt.exe", "bdagent.exe"}
AGENT = {"python.exe", "pythonw.exe"}                     # CAPE's guest agent runs under Python
VM_MARKERS = ("vbox", "virtualbox", "vmware", "vmtools", "qemu", "xen", "parallels", "prl_", "sandboxie",
              "sbiedll", "cuckoo", "wine", "bochs", "hyper-v", "vmmouse", "vmhgfs", "vmci")
CAPE_MARKERS = ("capemon", "\\analyzer\\", "\\cape\\")

EXT_CLASS = {**dict.fromkeys(("exe", "scr", "com", "pif", "cpl"), "EXE"), **dict.fromkeys(("dll", "ocx"), "DLL"),
             "sys": "DRIVER",
             **dict.fromkeys(("ps1", "psm1", "vbs", "vbe", "js", "jse", "wsf", "hta", "bat", "cmd", "py"), "SCRIPT"),
             **dict.fromkeys(("doc", "docx", "docm", "xls", "xlsx", "xlsm", "xlsb", "ppt", "pptx", "pdf", "rtf",
                              "odt"), "DOC"),
             **dict.fromkeys(("zip", "rar", "7z", "cab", "gz", "tar", "iso", "img"), "ARCHIVE"),
             **dict.fromkeys(("sqlite", "db", "ldb", "sqlite3"), "DB"),
             **dict.fromkeys(("ini", "cfg", "conf", "xml", "json", "yaml", "yml", "inf", "config"), "CONFIG"),
             **dict.fromkeys(("txt", "csv", "log"), "TEXT"),
             **dict.fromkeys(("jpg", "jpeg", "png", "bmp", "gif", "ico"), "IMAGE"),
             **dict.fromkeys(("lnk", "url"), "LINK"), **dict.fromkeys(("tmp", "dat", "bin"), "TMP")}

RET_NOT_FOUND = {0xC0000034, 0xC000003A, 0xC0000135, 0xC0000225, 0x2, 0x3, 0x7E, 0x490}
RET_DENIED = {0xC0000022, 0x5, 0xC0000061}
RET_NO_MORE = {0x8000001A, 0x103, 0x12, 0x80000006}
RET_BUFFER = {0xC0000023, 0x80000005, 0x7A, 0xEA, 0xC0000004}

_drive = re.compile(r"^[a-z]:\\")
_ipv4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def norm_path(v: str) -> str:
    p = v.strip().strip('"').lower().replace("/", "\\")
    for pre in ("\\??\\", "\\\\?\\", "\\\\.\\globalroot\\"):
        if p.startswith(pre):
            p = p[len(pre):]
    p = re.sub(r"^\\device\\harddiskvolume\d+\\", "c:\\\\", p)
    p = re.sub(r"^\\systemroot\\", "c:\\\\windows\\\\", p)
    p = re.sub(r"%(temp|tmp)%", "c:\\\\users\\\\*\\\\appdata\\\\local\\\\temp", p)
    p = re.sub(r"%appdata%", "c:\\\\users\\\\*\\\\appdata\\\\roaming", p)
    p = re.sub(r"%localappdata%", "c:\\\\users\\\\*\\\\appdata\\\\local", p)
    p = re.sub(r"%(userprofile|homepath)%", "c:\\\\users\\\\*", p)
    p = re.sub(r"%(systemroot|windir)%", "c:\\\\windows", p)
    p = re.sub(r"^c:\\users\\[^\\]+", "c:\\\\users\\\\*", p)        # user names, raw or [REDACTED], dropped
    p = re.sub(r"^c:\\documents and settings\\[^\\]+", "c:\\\\users\\\\*", p)
    return p


def path_class(v: str, self_paths: tuple = ()) -> str:
    p = norm_path(v)
    if any(norm_path(s) == p for s in self_paths):
        return "SELF_IMAGE"
    if any(p.startswith(norm_path(s).rsplit("\\", 1)[0] + "\\") for s in self_paths if "\\" in s):
        return "SELF_DIR"
    if any(m in p for m in CAPE_MARKERS):
        return "CAPE_ARTEFACT"
    if any(m in p for m in VM_MARKERS):
        return "VM_ARTEFACT"
    if p.startswith("\\\\.\\pipe\\") or p.startswith("\\device\\namedpipe"):
        return "PIPE"
    if p.startswith("\\\\.\\") or p.startswith("\\device\\"):
        return "DEVICE"
    if p.startswith("\\\\"):
        return "UNC"
    if not _drive.match(p):
        return "RELATIVE"
    if not p.startswith("c:\\"):
        return "OTHER_DRIVE"
    rules = (("c:\\windows\\system32\\", "SYS32"), ("c:\\windows\\syswow64\\", "SYSWOW64"),
             ("c:\\windows\\temp\\", "SYSTEM_TEMP"), ("c:\\windows\\", "WINDOWS"),
             ("c:\\program files", "PROGRAM_FILES"), ("c:\\programdata\\", "PROGRAMDATA"),
             ("c:\\users\\public\\", "USERS_PUBLIC"))
    for pre, cls in rules:
        if p.startswith(pre):
            return cls
    if p.startswith("c:\\users\\*\\"):
        rest = p[len("c:\\users\\*\\"):]
        sub = (("appdata\\local\\temp", "USER_TEMP"), ("appdata\\local\\google\\chrome\\user data", "BROWSER_PROFILE"),
               ("appdata\\local\\microsoft\\edge\\user data", "BROWSER_PROFILE"),
               ("appdata\\roaming\\mozilla\\firefox", "BROWSER_PROFILE"),
               ("appdata\\roaming\\opera software", "BROWSER_PROFILE"),
               ("appdata\\roaming\\microsoft\\windows\\start menu\\programs\\startup", "USER_STARTUP"),
               ("appdata\\roaming", "USER_APPDATA_ROAMING"), ("appdata\\local", "USER_APPDATA_LOCAL"),
               ("appdata\\locallow", "USER_APPDATA_LOCAL"), ("desktop", "USER_DESKTOP"),
               ("documents", "USER_DOCUMENTS"), ("downloads", "USER_DOWNLOADS"))
        for pre, cls in sub:
            if rest.startswith(pre):
                return cls
        return "USER_OTHER"
    if p.count("\\") >= 1 and p not in ("c:\\",):
        return "ROOT_OTHER"
    return "ROOT"


def ext_class(v: str) -> str:
    base = norm_path(v).rsplit("\\", 1)[-1]
    if "." not in base:
        return "NONE"
    return EXT_CLASS.get(base.rsplit(".", 1)[-1], "OTHER")


def reg_class(v: str) -> str | None:
    k = v.strip().lower().replace("/", "\\")
    hive = None
    for pre, h in (("\\registry\\machine\\", "HKLM"), ("hkey_local_machine\\", "HKLM"), ("hklm\\", "HKLM"),
                   ("\\registry\\user\\", "HKU"), ("hkey_users\\", "HKU"), ("hkey_current_user\\", "HKCU"),
                   ("hkcu\\", "HKCU"), ("hkey_classes_root\\", "HKCR"), ("hkcr\\", "HKCR"),
                   ("hkey_current_config\\", "HKCC")):
        if k.startswith(pre):
            hive, k = h, k[len(pre):]
            break
    if hive is None:
        return None
    if hive == "HKU":
        k = re.sub(r"^s-1-5-[\d-]+(_classes)?\\", "", k)  # SIDs dropped
    if any(m in k for m in VM_MARKERS):
        cat = "VM_ARTEFACT"
    else:
        cats = (("\\currentversion\\run", "RUN"), ("\\winlogon", "WINLOGON"), ("image file execution options", "IFEO"),
                ("\\services\\", "SERVICES"), ("startupapproved", "RUN"), ("\\clsid", "COM"),
                ("\\shell\\open\\command", "FILE_ASSOC"), ("\\policies\\", "POLICIES"),
                ("windows defender", "DEFENDER"), ("firewallpolicy", "FIREWALL"), ("internet settings", "INTERNET"),
                ("\\cryptography", "CRYPTOGRAPHY"), ("\\control\\nls", "LOCALE"), ("\\international", "LOCALE"),
                ("timezoneinformation", "TIMEZONE"), ("hardware\\description\\system", "HARDWARE_DESC"),
                ("\\enum\\ide", "DISK_ENUM"), ("\\enum\\scsi", "DISK_ENUM"), ("session manager", "SESSION_MANAGER"),
                ("appcompat", "APP_COMPAT"), ("\\explorer", "EXPLORER"), ("terminal server", "TERMINAL_SERVER"),
                ("safeboot", "SAFEBOOT"), ("software\\microsoft", "MICROSOFT"), ("software\\classes", "CLASSES"),
                ("software\\", "SOFTWARE_OTHER"), ("system\\", "SYSTEM_OTHER"))
        cat = next((c for pre, c in cats if pre in k), "OTHER")
    return f"{hive}:{cat}"


def value_class(v: str) -> str:
    n = v.strip().lower()
    if any(m in n for m in VM_MARKERS):
        return "VM_ARTEFACT"
    if n in {"systembiosversion", "videobiosversion", "systembiosdate", "identifier", "systemmanufacturer",
             "systemproductname", "biosversion", "biosvendor", "systemfamily"}:
        return "BIOS_INFO"
    if n in {"machineguid", "productid", "installdate", "digitalproductid"}:
        return "SYSTEM_ID"
    return "OTHER"


def ip_class(v: str) -> str | None:
    m = _ipv4.search(v)
    cand = m.group(0) if m else v.strip().strip("[]")
    try:
        ip = ipaddress.ip_address(cand)
    except ValueError:
        return "DOMAIN" if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", v.strip().lower()) else None
    if ip.is_loopback:
        return "LOOPBACK"
    if ip.is_private:
        return "PRIVATE"
    if ip.is_link_local:
        return "LINK_LOCAL"
    if ip.is_multicast:
        return "MULTICAST"
    return "PUBLIC"


def sleep_class(v) -> str | None:
    try:
        ms = int(str(v), 0)
    except ValueError:
        return None
    if ms in (0xFFFFFFFF, -1):
        return "INFINITE"
    for lim, cls in ((0, "ZERO"), (99, "LT100MS"), (999, "LT1S"), (9_999, "LT10S"), (59_999, "LT60S"),
                     (599_999, "LT10MIN")):
        if ms <= lim:
            return cls
    return "GE10MIN"


def proc_class(v: str, self_paths: tuple = ()) -> str:
    s = v.strip().strip('"').lower()
    exe = re.split(r"\.exe\b", s)[0].rsplit("\\", 1)[-1] + ".exe" if ".exe" in s else s.split(" ")[0].rsplit("\\", 1)[-1]
    if any(exe == p.lower().replace("/", "\\").rsplit("\\", 1)[-1] for p in self_paths):
        return "SELF"
    for group, cls in ((AGENT, "AGENT"), (ANALYSIS_TOOLS, "ANALYSIS_TOOL"), (SECURITY, "SECURITY"),
                       (SHELL_TOOLS, "SHELL_TOOL"), (BROWSERS, "BROWSER"), (OFFICE, "OFFICE"),
                       (SYSTEM_PROCS, "SYSTEM_PROC")):
        if exe in group:
            return cls
    return "OTHER_EXE"


def ret_class(ok: bool, ret: str) -> str:
    if ok:
        return "OK"
    try:
        code = int(ret, 16) & 0xFFFFFFFF if ret.lower().startswith("0x") else int(ret) & 0xFFFFFFFF
    except ValueError:
        return "FAIL"
    for group, cls in ((RET_NOT_FOUND, "NOT_FOUND"), (RET_DENIED, "DENIED"), (RET_NO_MORE, "NO_MORE"),
                       (RET_BUFFER, "BUFFER")):
        if code in group:
            return cls
    return "FAIL"


def arg_classes(api: str, args: tuple, self_paths: tuple = ()) -> tuple:
    """kappa(args): sorted (slot, class) pairs from whitelisted argument names only."""
    out = []
    for name, value in args:
        n = name.lower()
        v = value if isinstance(value, str) else str(value)
        if n == "status" and v.lower().startswith("skip"):
            out.append(("Status", "SLEEP_SKIPPED"))           # CAPE skipped the sleep (environment-set)
            continue
        if n in SLEEP_ARGS:
            c = sleep_class(v)
            if c:
                out.append((name, "sleep=" + c))
            continue
        if n in VALUE_ARGS:
            out.append((name, "val=" + value_class(v)))
            continue
        if n in REG_ARGS or n in PATH_ARGS:
            rc = reg_class(v) if n in REG_ARGS else None
            if rc:
                out.append((name, "reg=" + rc))
            elif n in PATH_ARGS and ("\\" in v or "/" in v or v.lower().startswith("%")):
                out.append((name, "path=" + path_class(v, self_paths)))
                out.append((name + "#", "ext=" + ext_class(v)))
            continue
        if n in NET_ARGS:
            c = ip_class(v)
            if c:
                out.append((name, "ip=" + c))
            continue
        if n in PROC_ARGS:
            out.append((name, "proc=" + proc_class(v, self_paths)))
    return tuple(sorted(set(out)))


def token(api: str, args: tuple, ok: bool, ret: str, hook_set: set | None, self_paths: tuple = ()) -> str:
    """Eq. (1): (a, kappa(args), r(ret)) if a is in the common hook set H_c, else <OOS>."""
    if hook_set is not None and api not in hook_set:
        return OOS
    kap = ";".join(f"{s}:{c}" for s, c in arg_classes(api, args, self_paths))
    return f"{api}|{kap}|{ret_class(ok, ret)}"


def beta(n_run: int) -> int:                               # run-length bucket, eq. (2)
    return min(int(math.floor(math.log2(max(n_run, 1)))), BETA_MAX)


def collapse(tokens: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """(t, n) pairs, n = 1 + CAPE 'repeated', to (t, beta(n_run)) elements; consecutive repeats merge."""
    out, prev, run = [], None, 0
    for t, n in tokens:
        if t == prev:
            run += n
        else:
            if prev is not None:
                out.append((prev, beta(run)))
            prev, run = t, n
    if prev is not None:
        out.append((prev, beta(run)))
    return out


def trunc_head_tail(seq: list, L: int) -> list:
    return seq if len(seq) <= L else seq[: L // 2] + seq[-(L // 2):]


def head_tail_store(seq: list, cap: int = L_CAP) -> list:
    """The part of a sequence a stage must keep to apply trunc_head_tail with any L in [2, cap] later: the whole
    sequence up to cap elements, else its first and last cap // 2 elements (S9 dry run, amendment of 3 Nov 2026)."""
    return seq if len(seq) <= cap else seq[: cap // 2] + seq[-(cap // 2):]


def trunc_head(seq: list, L: int) -> list:
    return seq[:L]


def choose_L(train_lengths: list[int]) -> int:
    """Smallest power of two at or above the 95th percentile of collapsed training lengths, capped at L_CAP."""
    if not train_lengths:
        return L_CAP
    s = sorted(train_lengths)
    p95 = s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)]
    return min(L_CAP, 1 << max(0, math.ceil(math.log2(max(p95, 1)))))


def common_hook_set(train_api_sets: dict[str, set]) -> set:
    """H_c: APIs observed in the training traces of every installation (S7 harmonisation step 2)."""
    sets = list(train_api_sets.values())
    return set.intersection(*sets) if sets else set()


class Vocab:
    """Token types seen in at least VOCAB_MIN_TRACES training traces; others become <UNK> after class assignment."""

    def __init__(self, train_sequences: list[list[tuple[str, int]]], min_traces: int = VOCAB_MIN_TRACES):
        df = Counter()
        for seq in train_sequences:
            df.update({t for t, _ in seq})
        specials = ["<PAD>", "<M>", UNK, OOS]
        self.itos = specials + sorted(t for t, c in df.items() if c >= min_traces and t not in specials)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, t: str) -> int:
        return self.stoi.get(t, self.stoi[UNK])
