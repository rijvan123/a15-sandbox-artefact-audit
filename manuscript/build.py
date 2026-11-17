#!/usr/bin/env python3
"""Build the A15 Scientific Reports manuscript, Supplementary Information, cover letter and S12 logs.

Usage (here or on the authors' PC):
    python build.py --out out                                   # draft build, Figs 4-6 as labelled slots
    python build.py --out out --s10 /path/to/WORK/outputs/s10   # final build from the real evaluation outputs

Needs pandoc (>= 3.1), LibreOffice (soffice) and Python packages python-docx, pypdf, matplotlib.
Citations are written [@key; @key2] and numbered by first appearance (main text, then figure legends, then tables).
Equations are written {{EQ:n}} with n the number in the S8 model; they are renumbered by first appearance.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refs import REFS, R, formatted  # noqa: E402

CIT = re.compile(r"\[@([^\]]+)\]")
REAL_SHA_PREFIX = "b2e80138"           # results.json digest recorded in the ledger (entry 130,782)
FIG_FROM_S10 = {"probes": "Fig6_probes_and_attributions", "masking": "Fig4_masking_effects", "transfer": "Fig5_transfer_and_recovery"}


# ----------------------------------------------------------------------------------------------- text processing
def load_sources():
    parts = [(HERE / f).read_text(encoding="utf-8") for f in ("ms_main.md", "ms_methods.md", "ms_results.md", "ms_back.md")]
    return "\n\n".join(parts)


def fix_bars(tex: str) -> str:
    """Write paired \\big delimiters as \\left ... \\right so every renderer draws matched, importable delimiters."""
    n = [0]

    def bar(m):
        n[0] += 1
        return r"\left|" if n[0] % 2 else r"\right|"

    out = re.sub(r"\\(?:big|Big|bigg|Bigg)[lr]?\|", bar, tex)
    if n[0] % 2:
        raise ValueError("unpaired \\big| bar in an equation")
    out = re.sub(r"\\(?:big|Big|bigg|Bigg)l?(\(|\[|\\\{)", lambda m: r"\left" + m.group(1), out)
    out = re.sub(r"\\(?:big|Big|bigg|Bigg)r?(\)|\]|\\\})", lambda m: r"\right" + m.group(1), out)
    depth = 0                                     # \left/\right may not span an alignment cell or row
    for tok in re.findall(r"\\left|\\right|&|\\\\", out):
        if tok == r"\left":
            depth += 1
        elif tok == r"\right":
            depth -= 1
        elif depth:
            return tex                            # leave this equation as written
        if depth < 0:
            return tex
    return out if depth == 0 else tex


def place_equations(text: str, eqs: dict):
    order = []

    def put(m):
        n = m.group(1)
        if n in order:
            raise ValueError(f"equation {n} placed twice")
        order.append(n)
        return f"$$\n{eqs[n]}\\qquad({len(order)})\n$$"

    text = re.sub(r"\{\{EQ:(\d+)\}\}", put, text)
    mapping = {n: i + 1 for i, n in enumerate(order)}

    def ref(m):
        if m.group(1) not in mapping:
            raise KeyError(f"equation {m.group(1)} referenced but not placed")
        return str(mapping[m.group(1)])

    return re.sub(r"\{\{EQREF:(\d+)\}\}", ref, text), mapping


def keys_of(m):
    return [k.strip().lstrip("@") for k in m.group(1).split(";")]


def number_citations(text: str):
    """Order of first appearance in the assembled manuscript; legends and tables sit where they are first cited."""
    order, where, section = {}, {}, "Title page"
    for line in text.splitlines():
        h = re.match(r"^#\s+(.+)$", line)
        if h:
            section = h.group(1).strip()
        for m in CIT.finditer(line):
            for k in keys_of(m):
                if k not in REFS:
                    raise KeyError(f"unknown reference key {k}")
                order.setdefault(k, len(order) + 1)
                where.setdefault(k, [])
                if section not in where[k]:
                    where[k].append(section)
    return order, where


def fmt_nums(nums):
    nums, out, i = sorted(set(nums)), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out += [f"{nums[i]}–{nums[j]}"] if j - i >= 2 else [str(n) for n in nums[i:j + 1]]
        i = j + 1
    return "[" + ",".join(out) + "]"


def cite(text, order):
    return CIT.sub(lambda m: fmt_nums([order[k] for k in keys_of(m)]), text)


def reference_list(order):
    return "\n".join(f"{n}. {formatted(REFS[k])}" for k, n in sorted(order.items(), key=lambda kv: kv[1]))


def embed_copy(png: Path, max_px: int = 1900) -> Path:
    """Copy for embedding: about 300 dpi at the 16 cm print width keeps the single-file submission under 3 MB;
    full-resolution files are exported separately."""
    from PIL import Image
    dst = png.parent / "embed" / png.name
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < png.stat().st_mtime:
        im = Image.open(png)
        if im.width > max_px:
            im = im.resize((max_px, round(im.height * max_px / im.width)), Image.LANCZOS)
        im = im.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)   # palette PNG
        im.save(dst, optimize=True)
    return dst


def place_figures(text, figdir: Path):
    def img(m):
        key = m.group(1)
        stem = f"Fig{key}" if key.isdigit() else f"Fig_{key}"
        return f"![]({embed_copy(figdir / (stem + '.png')).as_posix()}){{width=16cm}}"
    return re.sub(r"\{\{FIG:(\w+)\}\}", img, text)


# ----------------------------------------------------------------------------------------------- Word and PDF
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-2000:]}")
    return r


def make_reference_docx(path: Path, landscape=False):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    with open(path, "wb") as fh:
        fh.write(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                                capture_output=True, check=True).stdout)
    d = Document(path)

    def style(name, size=11, bold=False, italic=False, spacing=1.5, before=0, after=6, keep=False):
        want = {name.lower(), name.replace(" ", "").lower()}
        s = next((x for x in d.styles if (x.name or "").lower() in want or (x.style_id or "").lower() in want), None)
        if s is None:
            return
        f = s.font
        f.name, f.size, f.bold, f.italic = "Times New Roman", Pt(size), bold, italic
        f.color.rgb = RGBColor(0, 0, 0)
        rpr = s.element.get_or_add_rPr()
        col = rpr.find(qn("w:color"))
        if col is not None:
            for a in ("w:themeColor", "w:themeShade", "w:themeTint"):
                col.attrib.pop(qn(a), None)
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = rpr.makeelement(qn("w:rFonts"), {})
            rpr.insert(0, rf)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            rf.attrib.pop(qn(a), None)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), "Times New Roman")
        pf = s.paragraph_format
        pf.line_spacing, pf.space_before, pf.space_after = spacing, Pt(before), Pt(after)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.keep_with_next = keep

    for n in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Bibliography", "Figure",
              "Captioned Figure"):
        style(n)
    style("Title", 14, bold=True, after=12)
    style("Heading 1", 12, bold=True, before=12, after=6, keep=True)
    style("Heading 2", 11, bold=True, before=10, after=4, keep=True)
    style("Heading 3", 11, bold=True, italic=True, before=8, after=4, keep=True)
    style("Image Caption", 10, italic=False, spacing=1.0, after=10)
    style("Table Caption", 11, bold=True, spacing=1.0, before=6, after=6, keep=True)
    style("Caption", 10, spacing=1.0)
    for sec in d.sections:
        if landscape:
            from docx.enum.section import WD_ORIENT
            from docx.shared import Cm
            sec.orientation, sec.page_width, sec.page_height = WD_ORIENT.LANDSCAPE, Cm(29.7), Cm(21.0)
    d.save(path)


def postprocess(path: Path, line_numbers=True, landscape=False, table_pt=9):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    d = Document(path)
    names = {st.name for st in d.styles}
    if "Figure Legend" in names:                       # legends sit under their figures, set compactly
        st = d.styles["Figure Legend"]
        st.font.size = Pt(10)
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.space_before = Pt(4)
        st.paragraph_format.space_after = Pt(12)
    for p in d.paragraphs:                              # keep figures and table titles with what follows
        if p._p.findall(".//" + qn("w:drawing")) or re.match(r"Table \d+\. ", p.text):
            p.paragraph_format.keep_with_next = True
    for sec in d.sections:
        if landscape:
            from docx.enum.section import WD_ORIENT
            sec.orientation, sec.page_width, sec.page_height = WD_ORIENT.LANDSCAPE, Cm(29.7), Cm(21.0)
        else:
            sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, m, Cm(2.5))
        sp = sec._sectPr
        if line_numbers and sp.find(qn("w:lnNumType")) is None:
            ln = OxmlElement("w:lnNumType")
            for k, v in (("w:countBy", "1"), ("w:restart", "continuous"), ("w:distance", "340")):
                ln.set(qn(k), v)
            later = [qn(t) for t in ("w:pgNumType", "w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg",
                                     "w:textDirection", "w:bidi", "w:rtlGutter", "w:docGrid", "w:printerSettings")]
            nxt = next((c for c in sp if c.tag in later), None)
            nxt.addprevious(ln) if nxt is not None else sp.append(ln)
        fp = sec.footer.paragraphs[0]
        fp.alignment = 1
        ppr = fp._p.get_or_add_pPr()
        if ppr.find(qn("w:suppressLineNumbers")) is None:
            sln = OxmlElement("w:suppressLineNumbers")       # schema order: before pBdr ... jc
            nxt = next((c for c in ppr if c.tag in [qn(t) for t in (
                "w:pBdr", "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
                "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd", "w:snapToGrid",
                "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
                "w:textDirection", "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
                "w:rPr", "w:sectPr", "w:pPrChange")]), None)
            nxt.addprevious(sln) if nxt is not None else ppr.append(sln)
        if not fp._p.findall(qn("w:fldSimple")):
            fld = OxmlElement("w:fldSimple")
            fld.set(qn("w:instr"), "PAGE")
            r = OxmlElement("w:r")
            t = OxmlElement("w:t")
            t.text = "1"
            r.append(t)
            fld.append(r)
            fp._p.append(fld)
    for t in d.tables:
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "bottom", "insideH"):
            b = OxmlElement(f"w:{side}")
            for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", "000000")):
                b.set(qn(k), v)
            borders.append(b)
        old = tblPr.find(qn("w:tblBorders"))
        if old is not None:
            tblPr.remove(old)
        tblPr.append(borders)
        pt = 8 if (len(t.rows) >= 15 and len(t.columns) >= 7) else table_pt     # long review tables
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    for r in p.runs:
                        r.font.size = Pt(pt)
                        r.font.name = "Times New Roman"
    landscape_blocks(d)
    d.save(path)


LANDSCAPE_TEXT_TWIPS = 16838 - 2 * 1418          # A4 landscape width less 2.5 cm margins


def landscape_blocks(d):
    """Turn {{LANDSCAPE_START}} ... {{LANDSCAPE_END}} into a landscape section and widen its tables to the page."""
    from docx.oxml.ns import qn
    body = d.element.body
    final = body.find(qn("w:sectPr"))
    inside, tables = False, []
    for el in body.iterchildren():
        if el.tag == qn("w:p"):
            txt = "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
            if txt == "{{LANDSCAPE_START}}":
                inside = True
            elif txt == "{{LANDSCAPE_END}}":
                inside = False
        elif el.tag == qn("w:tbl") and inside:
            tables.append(el)
    for tbl in tables:
        grid = tbl.find(qn("w:tblGrid"))
        cols = grid.findall(qn("w:gridCol")) if grid is not None else []
        total = sum(int(c.get(qn("w:w"), "0")) for c in cols) or 1
        for c in cols:
            c.set(qn("w:w"), str(int(int(c.get(qn("w:w"), "0")) * LANDSCAPE_TEXT_TWIPS / total)))
        tblW = tbl.find(qn("w:tblPr")).find(qn("w:tblW"))
        if tblW is not None:
            tblW.set(qn("w:type"), "dxa")
            tblW.set(qn("w:w"), str(LANDSCAPE_TEXT_TWIPS))
        for tcW in list(tbl.iter(qn("w:tcW"))):
            tcW.getparent().remove(tcW)
    for p in d.paragraphs:
        txt = p.text.strip()
        if txt not in ("{{LANDSCAPE_START}}", "{{LANDSCAPE_END}}"):
            continue
        for r in p.runs:
            r.text = ""
        sp = copy.deepcopy(final)
        if txt == "{{LANDSCAPE_END}}":                 # the section ending here is the landscape one
            pg = sp.find(qn("w:pgSz"))
            w_, h_ = pg.get(qn("w:w")), pg.get(qn("w:h"))
            pg.set(qn("w:w"), h_)
            pg.set(qn("w:h"), w_)
            pg.set(qn("w:orient"), "landscape")
        p._p.get_or_add_pPr().append(sp)


# ----------------------------------------------------------------------------------------------- OOXML schema order
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ORDER = {
    "pPr": "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs "
           "suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd "
           "snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment "
           "textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
    "rPr": "rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid "
           "vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs "
           "em lang eastAsianLayout specVanish oMath",
    "style": "name aliases basedOn next link autoRedefine uiPriority semiHidden unhideWhenUsed qFormat locked personal "
             "personalCompose personalReply rsid pPr rPr tblPr trPr tcPr tblStylePr",
}
ORDER["tcPr"] = "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark"
ORDER["tblPr"] = ("tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing "
                  "tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription")
ORDER["settings"] = (
    "writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries "
    "displayBackgroundShape printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts "
    "embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins alignBordersAndEdges bordersDoNotSurroundHeader "
    "bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle proofState "
    "formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge "
    "revisionView trackRevisions doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride "
    "styleLockTheme styleLockQFSet defaultTabStop autoHyphenation consecutiveHyphenLimit hyphenationZone "
    "doNotHyphenateCaps showEnvelope summaryLength clickAndTypeStyle defaultTableStyle evenAndOddHeaders "
    "bookFoldRevPrinting bookFoldPrinting bookFoldPrintingSheets drawingGridHorizontalSpacing "
    "drawingGridVerticalSpacing displayHorizontalDrawingGridEvery displayVerticalDrawingGridEvery "
    "doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin drawingGridVerticalOrigin doNotShadeFormData "
    "noPunctuationKerning characterSpacingControl printTwoOnOne strictFirstAndLastChars noLineBreaksAfter "
    "noLineBreaksBefore savePreviewPicture doNotValidateAgainstSchema saveInvalidXml ignoreMixedContent "
    "alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly useXSLTWhenSaving saveThroughXslt showXMLTags "
    "alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr endnotePr compat docVars rsids mathPr "
    "attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats doNotAutoCompressPictures forceUpgrade "
    "captions readModeInkLockDown smartTagType schemaLibrary shapeDefaults doNotEmbedSmartTags decimalSymbol "
    "listSeparator")
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
MATH_RPR = "lit nor scr sty brk aln"


def _reorder(el, order):
    rank = {t: i for i, t in enumerate(order.split())}
    kids = list(el)
    keyed = sorted(range(len(kids)), key=lambda i: (rank.get(str(kids[i].tag).split("}")[-1], 10 ** 6), i))
    for k in kids:
        el.remove(k)
    for i in keyed:
        el.append(kids[i])


def schema_fix(path: Path):
    import zipfile
    from lxml import etree
    tmp = path.with_suffix(".fixing")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                root = etree.fromstring(data)
                for tag in ("pPr", "rPr", "style", "tcPr", "tblPr"):
                    for el in root.iter(W + tag):
                        if tag == "rPr":
                            el.text = None
                            for c in el:
                                c.tail = None
                        _reorder(el, ORDER[tag])
                for pm in root.iter(W + "pgMar"):
                    for a, v in (("header", "708"), ("footer", "708"), ("gutter", "0")):
                        if pm.get(W + a) is None:
                            pm.set(W + a, v)
                for ns in root.iter(W + "nsid"):
                    v = ns.get(W + "val", "")
                    if len(v) < 8:
                        ns.set(W + "val", v.rjust(8, "0"))
                for tag, order in (("dPr", "begChr sepChr endChr grow shp ctrlPr"), ("mcPr", "count mcJc")):
                    for el in root.iter(M + tag):
                        _reorder(el, order)
                for el in root.iter(M + "rPr"):
                    if el.find(M + "nor") is not None:          # schema: nor excludes scr and sty
                        for c in list(el):
                            if c.tag in (M + "scr", M + "sty"):
                                el.remove(c)
                    rank = {t: i for i, t in enumerate(MATH_RPR.split())}
                    kids = sorted(list(el), key=lambda c: rank.get(c.tag.replace(M, ""), 99))
                    for c in list(el):
                        el.remove(c)
                    for c in kids:
                        el.append(c)
                if root.tag == W + "settings":
                    _reorder(root, ORDER["settings"])
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            zout.writestr(item, data)
    tmp.replace(path)


def pdf_variant(md: str) -> str:
    """PDF copy only. LibreOffice's equation import shows its error mark for alignment points and for characters it
    parses as operators (a bare * or < > in a superscript, #, set minus); write them as literal text instead."""
    def fix(m):
        t = m.group(0)
        t = re.sub(r"\\begin\{aligned\}(.*?)\\end\{aligned\}",
                   lambda a: "\\begin{aligned}" + a.group(1).replace("&", "") + "\\end{aligned}", t, flags=re.S)
        t = re.sub(r"\^\{\*", r"^{\\text{*}", t)
        t = re.sub(r"\^\*", r"^{\\text{*}}", t)
        t = re.sub(r"\^\{([<>])\}", r"^{\\text{\1}}", t)
        t = t.replace(r"\#", r"\text{\#}").replace(r"\setminus", r"\backslash ")
        t = t.replace(r"\underbrace{(t,\dots,t)}_{n_{\mathrm{run}}}", r"\left(t,\dots,t\right)_{n_{\mathrm{run}}}")
        return t
    return re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$", fix, md, flags=re.S)


def strip_zero_width(path: Path):
    """LibreOffice draws U+200B (placeholders the TeX converter puts in empty sum limits) as a stray glyph; PDF copy only."""
    import zipfile
    tmp = path.with_suffix(".zw")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = data.decode("utf-8").replace("\u200b", "").encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)


def to_pdf(docx: Path):
    helper = Path("/mnt/skills/public/docx/scripts/office/soffice.py")
    cmd = ([sys.executable, str(helper)] if helper.exists() else ["soffice"]) + \
          ["--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)]
    run(cmd)
    return docx.with_suffix(".pdf")


def pandoc_docx(md: str, out: Path, ref: Path, resource: Path):
    src = out.with_suffix(".md")
    src.write_text(md, encoding="utf-8")
    run(["pandoc", str(src), "-f", "markdown+tex_math_dollars+pipe_tables+superscript+subscript+fenced_divs",
         "-t", "docx", "--reference-doc", str(ref), "--resource-path", str(resource), "-o", str(out)])
    subprocess.run(["zip", "-d", str(out), "*/"], capture_output=True)   # drop folder entries (docx skill)


# ----------------------------------------------------------------------------------------------- counts
def plain_words(md: str) -> int:
    t = re.sub(r"\$\$.*?\$\$", " ", md, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", " ", t)
    t = re.sub(r"\[[0-9,–]+\]", " ", t)
    t = re.sub(r"[#*|:~^\\]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


def text_only(md: str) -> str:
    """Drop inline figure legends, table titles, table rows and the note under each table (not counted as text)."""
    md = re.sub(r'::: \{custom-style="Figure Legend"\}\n.*?\n:::', " ", md, flags=re.S)
    md = re.sub(r"(?m)^\*\*Table \d+\..*?\*\*\s*\n(\s*\n)?((\|.*\n)+)(\s*\n)?(?:(?!#)[^\n]+\n?)?", " ", md)
    return md


def sections(md: str):
    out, cur, buf = {}, "front", []
    for line in md.splitlines():
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            out[cur] = "\n".join(buf)
            cur, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    out[cur] = "\n".join(buf)
    return out


def legend_words(md: str):
    res = {}
    for m in re.finditer(r'custom-style="Figure Legend"\}\n\*\*Figure (\d+)\.(.*?)\n:::', md, re.S):
        res[f"Figure {m.group(1)}"] = plain_words(m.group(2))
    return res


# ----------------------------------------------------------------------------------------------- real S10 outputs
def check_real_results(s10: Path, allow_synthetic=False):
    rj = s10 / "results.json"
    data = json.loads(rj.read_text())
    sha = hashlib.sha256(rj.read_bytes()).hexdigest()
    label = str(data.get("label", "")) + str(data.get("plan", {}).get("label", ""))
    if "SYNTHETIC" in label.upper() and not allow_synthetic:
        raise SystemExit(f"{rj} is the synthetic rehearsal (label {label!r}); refusing to build from it")
    if not sha.startswith(REAL_SHA_PREFIX) and not allow_synthetic:
        raise SystemExit(f"{rj} has SHA-256 {sha[:16]}..., not the ledger-recorded {REAL_SHA_PREFIX}...")
    return data, sha


def primary(data):
    """The rehearsal nests the primary analysis; the measured file stores its threshold there and the results at top level."""
    pa = data.get("primary_analysis")
    return pa if isinstance(pa, dict) else data


def table2_from_results(data):
    base = primary(data)
    L = base.get("confirmatory", data.get("confirmatory", "transformer"))
    dec = base["decisions"][L]
    names = {"I1": "MELD-DS-448", "I2": "WinMET"}
    claim = {"H1": "H1: artefact tokens alone identify families",
             "H2": "H2: masking F and P costs more than matched random masks",
             "H3": "H3: cross-installation macro-F1 is lower than within",
             "H4": "H4: F-masked training recovers more than 10% of the gap"}

    def comp_name(c):
        if c == "rho":
            return "ρ, pooled over directions"
        sym, inst = c.split("|")
        sub = sym.replace("psi", "")
        if sym == "psi3":
            other = "WinMET" if inst == "I1" else "MELD-DS-448"
            return f"ψ~3~, {names[inst]} to {other}"
        return f"ψ~{sub}~, {names[inst]}"

    def f(x, d=4):
        return "—" if x is None else (f"{x:.{d}f}" if abs(x) < 10 else f"{x:.1f}").replace("-", "\u2212")

    def g(x, d):
        return f"{x:.{d}f}".replace("-", "\u2212")

    rows, seen = [], set()
    for r in dec["rows"]:
        h = r["hypothesis"]
        est = r["estimate"]
        d = 3 if r["component"] == "rho" else 2 if r["component"].startswith("psi2") else 1   # decimals as in the text
        e = g(est, d)
        lo, hi = r["ci_lo"], r["ci_hi"]
        ci = f"({g(lo, d)} to {g(hi, d)})"
        first = h not in seen
        seen.add(h)
        dp = f"{f(dec['p_support'].get(h))} / {f(dec['p_rejection'].get(h))}" if first else ""
        hp = (f"{f(dec['holm_support'].get(h))} / {f(dec.get('holm_rejection', {}).get(h))}"
              if first else "")
        rows.append(f"| {claim[h] if first else ''} | {comp_name(r['component'])} | {e} {ci} | {format(r['margin'], 'g').replace('-', chr(0x2212))} | "
                    f"{f(r.get('p_component'))} | {dp} | {hp} | {dec['verdict'][h].capitalize() if first else ''} |")
    head = ("| Hypothesis | Component | Estimate (95% CI) | Margin | P value | Decision P (support / rejection) | "
            "Holm-adjusted P (support / rejection) | Verdict |\n"
            "|:--------------|:-------------|:--------------|:---------|:---------|:------------|:------------|:---------------|")
    return head + "\n" + "\n".join(rows)


S6_COMPONENTS = {"ψ~1~, MELD-DS-448": "psi1|I1", "ψ~1~, WinMET": "psi1|I2", "ψ~2~, MELD-DS-448": "psi2|I1",
                 "ψ~2~, WinMET": "psi2|I2", "ψ~3~, MELD-DS-448 to WinMET": "psi3|I1",
                 "ψ~3~, WinMET to MELD-DS-448": "psi3|I2", "ρ, pooled": "rho",
                 "ψ~4b~, MELD-DS-448": "psi4b|I1", "ψ~4b~, WinMET": "psi4b|I2"}
S6_LEARNERS = ["transformer", "bilstm", "cnn", "lgbm"]          # column order of Supplementary Table S4


def fill_s6(si_mod, data):
    """Fill the cells of Supplementary Table S4 (exploratory learners) from results.json; keep a published interval if only a point is stored."""
    base = primary(data)

    def num(x, d):
        return f"{x:.{d}f}".replace("-", "\u2212")

    rows, filled = [], 0
    for row in si_mod.S6:
        cells, comp = list(row), S6_COMPONENTS.get(row[0])
        if comp:
            d = 3 if comp == "rho" else 2 if comp.startswith("psi2") else 1   # decimals as in Table 2
            for i, learner in enumerate(S6_LEARNERS, start=1):
                hit = next((r for r in base.get("decisions", {}).get(learner, {}).get("rows", [])
                            if r.get("component") == comp), None)
                point = base.get("psi", {}).get(learner, {}).get(comp)
                if hit is not None and hit.get("ci_lo") is not None:
                    cells[i] = f"{num(hit['estimate'], d)} ({num(hit['ci_lo'], d)} to {num(hit['ci_hi'], d)})"
                    filled += 1
                elif point is not None and cells[i] == "\u2020":
                    cells[i] = num(point, d)
                    filled += 1
        rows.append(tuple("\u2014" if c == "\u2020" else c for c in cells))
    si_mod.S6 = rows
    return filled


def install_real_figures(s10: Path, figdir: Path, exports: Path):
    for name, stem in FIG_FROM_S10.items():
        png, pdf = s10 / f"{stem}.png", s10 / f"{stem}.pdf"
        if not png.exists():
            raise SystemExit(f"missing {png}")
        shutil.copy(png, figdir / f"Fig_{name}.png")
        if pdf.exists():
            shutil.copy(pdf, figdir / f"Fig_{name}.pdf")


def resolve_symbols(md: str):
    """Number named figures (after Figs 1-3) and tables (after Table 1) in order of first appearance."""
    fnames = list(dict.fromkeys(m.group(1) for m in re.finditer(r"\{\{(?:F|FIG):([a-z]+)\}\}", md)))
    tnames = list(dict.fromkeys(m.group(1) for m in re.finditer(r"\{\{T:([a-z]+)\}\}", md)))
    fmap = {n: 4 + i for i, n in enumerate(fnames)}
    tmap = {n: 2 + i for i, n in enumerate(tnames)}
    return apply_maps(md, fmap, tmap), fmap, tmap


def apply_maps(text: str, fmap: dict, tmap: dict) -> str:
    text = re.sub(r"\{\{F:([a-z]+)\}\}", lambda m: str(fmap[m.group(1)]), text)
    return re.sub(r"\{\{T:([a-z]+)\}\}", lambda m: str(tmap[m.group(1)]), text)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--s10", help="folder with the real evaluation outputs (results.json, Fig4-6, Table2)")
    ap.add_argument("--allow-synthetic", action="store_true", help="testing only; never for the submission")
    ap.add_argument("--line-numbers", action="store_true", help="add continuous line numbers (off by default)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    figdir = HERE / "figs"
    exports = out / "figures"
    exports.mkdir(exist_ok=True)

    md = load_sources()
    real = None
    if a.s10:
        data, sha = check_real_results(Path(a.s10), a.allow_synthetic)
        install_real_figures(Path(a.s10), figdir, exports)
        md = re.sub(r"(\*\*Table \{\{T:hyp\}\}\..*?\*\*\n\n)\|.*?\n(?=\n)", lambda m: m.group(1) + table2_from_results(data) + "\n",
                    md, count=1, flags=re.S)
        real = {"results_sha256": sha, "label": data.get("label")}

    eqs = {k: fix_bars(v) for k, v in json.loads((HERE / "assets" / "equations.json").read_text(encoding="utf-8")).items()}
    md, fmap, tmap = resolve_symbols(md)
    md, eqmap = place_equations(md, eqs)
    order, where = number_citations(md)
    missing = [k for k in REFS if k not in order]
    if missing:
        raise SystemExit(f"references never cited: {missing}")
    md = cite(md, order).replace("{{REFERENCES}}", reference_list(order))
    md = place_figures(md, figdir)

    ref = out / "reference.docx"
    make_reference_docx(ref)
    docx = out / "A15_manuscript.docx"
    pandoc_docx(md, docx, ref, HERE)
    postprocess(docx, line_numbers=a.line_numbers)
    schema_fix(docx)
    pdf_src = out / "A15_manuscript_pdfsource.docx"
    pandoc_docx(pdf_variant(md), pdf_src, ref, HERE)
    postprocess(pdf_src, line_numbers=a.line_numbers)
    schema_fix(pdf_src)
    strip_zero_width(pdf_src)
    pdf = to_pdf(pdf_src).replace(out / "A15_manuscript.pdf")

    # Supplementary Information and cover letter
    import si
    if a.s10:
        print(f"Supplementary Table S4: {fill_s6(si, data)} cells filled from results.json")
    si_md = si.build(order, eqmap, eqs)
    si_md = apply_maps(cite(si_md, order), fmap, tmap)
    ref_l = out / "reference_landscape.docx"
    make_reference_docx(ref_l, landscape=True)
    si_docx = out / "A15_Supplementary_Information.docx"
    pandoc_docx(si_md, si_docx, ref_l, HERE)
    postprocess(si_docx, line_numbers=False, landscape=True, table_pt=8)
    schema_fix(si_docx)
    si_src = out / "A15_Supplementary_Information_pdfsource.docx"
    pandoc_docx(pdf_variant(si_md), si_src, ref_l, HERE)
    postprocess(si_src, line_numbers=False, landscape=True, table_pt=8)
    schema_fix(si_src)
    strip_zero_width(si_src)
    si_pdf = to_pdf(si_src).replace(out / "A15_Supplementary_Information.pdf")
    cover = (HERE / "cover_letter.md").read_text(encoding="utf-8")
    cv_docx = out / "A15_cover_letter.docx"
    pandoc_docx(cover, cv_docx, ref, HERE)
    postprocess(cv_docx, line_numbers=False)
    schema_fix(cv_docx)

    # figure exports (vector where available)
    for n in (1, 2, 3):
        src = HERE / "figs" / f"Fig{n}.pdf"
        if src.exists():
            shutil.copy(src, exports / f"Figure{n}.pdf")
    for name, n in fmap.items():
        for ext in (".pdf", ".png"):
            src = HERE / "figs" / f"Fig_{name}{ext}"
            if src.exists():
                shutil.copy(src, exports / (f"Figure{n}{ext}" if ext == ".pdf" else f"Figure{n}_600dpi{ext}"))

    # counts
    from pypdf import PdfReader
    secs = sections(md)
    title = re.search(r'custom-style="Title"\}\n(.*?)\n:::', md, re.S).group(1).strip()
    counts = {
        "title_words": len(title.split()),
        "abstract_words": plain_words(secs["Abstract"].split("**Keywords:**")[0]),
        "keywords": len(secs["Abstract"].split("**Keywords:**")[1].split(";")),
        "main_text_words": {k: plain_words(text_only(secs[k])) for k in ("Introduction", "Review of existing methods", "Results",
                                                              "Discussion", "Conclusions, limitations and future work")},
        "methods_words": plain_words(text_only(secs["Methodology"])) + plain_words(text_only(secs["Experimental setup"])),
        "legend_words": legend_words(md),
        "references": len(order),
        "reference_mix": {k: sum(REFS[x]["kind"] == k for x in order) for k in "JCE"},
        "share_2021_2026": round(sum(REFS[x]["year"] >= 2021 for x in order) / len(order), 3),
        "equations_methods": len(eqmap),
        "display_items": len(re.findall(r'custom-style="Figure Legend"', md)) + len(re.findall(r"\*\*Table \d\.", md)),
        "line_numbers": bool(a.line_numbers),
        "manuscript_pages": len(PdfReader(str(pdf)).pages),
        "si_pages": len(PdfReader(str(si_pdf)).pages),
        "docx_megabytes": round(docx.stat().st_size / 1e6, 2),
        "built_from_real_s10": real,
    }
    counts["main_text_total"] = sum(counts["main_text_words"].values())
    (out / "S12_counts.json").write_text(json.dumps(counts, indent=1, ensure_ascii=False))

    with open(out / "Reference_Verification_Log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Ref #", "Full reference", "DOI/URL", "Type", "Index & quartile", "Date verified / source checked",
                    "Sections citing it", "Why cited", "Retraction check"])
        for k, n in sorted(order.items(), key=lambda kv: kv[1]):
            r = REFS[k]
            typ = {"J": "Journal article", "C": "IEEE Xplore conference paper", "E": "Bounded exception"}[r["kind"]]
            retr = ("No retraction or correction notice seen at the S12 check; run verify_refs.py (Crossref updates "
                    "filter) before submission") if "S12" in r["src"] else "Not re-checked at S12; run verify_refs.py"
            if r["kind"] == "E" and not r.get("doi"):
                retr = "Not applicable (standard, dataset or proceedings without DOI)"
            w.writerow([n, re.sub(r"[*]", "", formatted(r)), r.get("doi") or "", typ, r["idx"], r["src"],
                        "; ".join(where[k]), r["why"], retr])
    print(json.dumps(counts, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
