"""Gera exercicios/lista.docx a partir de exercicios/lista.md, com o cabeçalho da UFAL/PPGEC.

    python tools/build_lista_docx.py                       (usa NUMERO e ENTREGA abaixo)
    python tools/build_lista_docx.py --numero 5 --entrega 24/11/2026

Converte o subconjunto de Markdown usado na lista: títulos, parágrafos, listas, citações,
tabelas, blocos de código, **negrito**, *itálico*, `código`, links e fórmulas $...$ simples."""

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = Path(__file__).resolve().parents[1]
MD = RAIZ / "exercicios" / "lista.md"
SAIDA = RAIZ / "exercicios" / "lista.docx"
LOGO = RAIZ / "exercicios" / "assets" / "logo_ufal.png"
SITE_REPO = "https://github.com/LucaswasTaken/Openhardware-hpc/blob/master/exercicios/"

NUMERO, ENTREGA, ANO = 5, "24/11/2026", 2026
INSTITUICAO = ["Universidade Federal de Alagoas – UFAL", "Centro de Tecnologia – CTEC",
               "Programa de Pós-Graduação em Engenharia Civil – PPGEC"]
DISCIPLINA = "Técnicas Computacionais Avançadas"
PROFESSORES = "Profs. Lucas Gouveia Omena Lopes e William Wagner Matos Lira"

FONTE, FONTE_COD = "Calibri", "Consolas"
AZUL = RGBColor(0x1F, 0x4E, 0x79)

# ---------------------------------------------------------------- fórmulas $...$ → texto
SUP = str.maketrans("0123456789n+-", "⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ⁺⁻")
SUB = str.maketrans("0123456789ijkn", "₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₙ")


def latex(s: str) -> str:
    s = s.replace(r"\tfrac12", "½").replace(r"\frac12", "½").replace(r"\,", " ")
    for a, b in ((r"\sum", "Σ"), (r"\times", "×"), (r"\delta", "δ"), (r"\cdot", "·"), (r"\pi", "π")):
        s = s.replace(a, b)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"√\1", s)
    s = re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(SUP), s)
    s = re.sub(r"\^(\w)", lambda m: m.group(1).translate(SUP), s)
    s = re.sub(r"_\{([^}]*)\}", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"_(\w)", lambda m: m.group(1).translate(SUB), s)
    return s.replace(" ", " ").replace("  ", " ")


# ---------------------------------------------------------------- texto com formatação
TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\$[^$]+\$)")


def link(par, texto, url, negrito=False):
    if not url.startswith("http"):
        url = SITE_REPO + url
    r_id = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                              is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), r_id)
    r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr")
    cor = OxmlElement("w:color"); cor.set(qn("w:val"), "1F4E79"); rpr.append(cor)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    if negrito:
        rpr.append(OxmlElement("w:b"))
    r.append(rpr); t = OxmlElement("w:t"); t.text = texto; t.set(qn("xml:space"), "preserve"); r.append(t)
    h.append(r); par._p.append(h)


def inline(par, texto, negrito=False, italico=False, tamanho=None):
    for parte in TOKEN.split(texto):
        if not parte:
            continue
        b, i, cod = negrito, italico, False
        if parte.startswith("**"):
            inline(par, parte[2:-2], True, italico, tamanho); continue
        if parte.startswith("*") and len(parte) > 2:
            inline(par, parte[1:-1], negrito, True, tamanho); continue
        if parte.startswith("`"):
            parte, cod = parte[1:-1], True
        elif parte.startswith("["):
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", parte)
            link(par, re.sub(r"[*`]", "", m.group(1)), m.group(2), negrito); continue
        elif parte.startswith("$"):
            parte, i = latex(parte[1:-1]), True
        run = par.add_run(parte)
        run.bold, run.italic = b or None, i or None
        if cod:
            run.font.name = FONTE_COD; run.font.size = Pt(9.5)
        elif tamanho:
            run.font.size = tamanho


def sombreia(celula, cor):
    tc = celula._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), cor); tc.append(sh)


def borda_inferior(par, cor="1F4E79"):
    ppr = par._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); b = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", cor)):
        b.set(qn(f"w:{k}"), v)
    bd.append(b); ppr.append(bd)


# ---------------------------------------------------------------- blocos
def cabecalho(doc, numero, entrega):
    t = doc.add_table(rows=1, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.columns[0].width = Cm(2.2); t.columns[1].width = Cm(14.8)
    c0, c1 = t.rows[0].cells
    c0.width, c1.width = Cm(2.2), Cm(14.8)
    c0.paragraphs[0].add_run().add_picture(str(LOGO), width=Cm(1.8))
    for j, linha in enumerate(INSTITUICAO):
        p = c1.paragraphs[0] if j == 0 else c1.add_paragraph()
        r = p.add_run(linha); r.bold = True; r.font.size = Pt(11)
        p.paragraph_format.space_after = Pt(0)
    p = doc.add_paragraph(); borda_inferior(p)
    for texto, tam, neg in ((f"{DISCIPLINA} – {ANO}", 12, True), (PROFESSORES, 10.5, False),
                            (f"{numero}ª Lista de Exercícios (entrega: {entrega})", 12, True)):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(texto); r.bold = neg; r.font.size = Pt(tam)


def tabela(doc, linhas):
    cel = [[c.strip() for c in l.strip().strip("|").split("|")] for l in linhas if not re.match(r"^\|[\s:|-]+\|$", l.strip())]
    t = doc.add_table(rows=len(cel), cols=len(cel[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, linha in enumerate(cel):
        for j, txt in enumerate(linha):
            c = t.cell(i, j); p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            inline(p, txt, negrito=(i == 0), tamanho=Pt(9.5))
            if i == 0:
                sombreia(c, "DCE6F0")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def codigo(doc, linhas):
    t = doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.cell(0, 0); sombreia(c, "F3F3F3"); p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
    for k, l in enumerate(linhas):
        r = p.add_run(l); r.font.name = FONTE_COD; r.font.size = Pt(8.5)
        if k < len(linhas) - 1:
            r.add_break()
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def converte(doc, md: str):
    linhas = md.splitlines()
    i = 0
    if linhas and linhas[0].startswith("# "):           # o título vira parte do cabeçalho
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(linhas[0][2:].replace("—", "–")); r.bold = True; r.font.size = Pt(14); r.font.color.rgb = AZUL
        i = 1
    while i < len(linhas):
        l = linhas[i]
        if l.startswith("```"):
            j = i + 1
            while not linhas[j].startswith("```"):
                j += 1
            codigo(doc, linhas[i + 1:j]); i = j + 1; continue
        if l.startswith("|"):
            j = i
            while j < len(linhas) and linhas[j].startswith("|"):
                j += 1
            tabela(doc, linhas[i:j]); i = j; continue
        if l.strip() == "---":
            borda_inferior(doc.add_paragraph(), "BFBFBF"); i += 1; continue
        m = re.match(r"^(#{2,3}) (.*)", l)
        if m:
            h = doc.add_heading(level=len(m.group(1)) - 1); inline(h, m.group(2)); i += 1; continue
        m = re.match(r"^(\s*)([-*]|\d+\.) (.*)", l)
        if m:
            estilo = "List Number" if m.group(2)[0].isdigit() else ("List Bullet 2" if m.group(1) else "List Bullet")
            p = doc.add_paragraph(style=estilo); inline(p, m.group(3)); p.paragraph_format.space_after = Pt(2)
            i += 1; continue
        if l.startswith(">"):
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.8)
            inline(p, l.lstrip("> "), italico=True); i += 1; continue
        if l.strip():
            p = doc.add_paragraph(); inline(p, l)
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numero", type=int, default=NUMERO)
    ap.add_argument("--entrega", default=ENTREGA)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    a = ap.parse_args()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2); sec.top_margin = sec.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]; st.font.name = FONTE; st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(4)
    for nome, tam in (("Heading 1", 13), ("Heading 2", 11.5)):
        h = doc.styles[nome]; h.font.name = FONTE; h.font.size = Pt(tam); h.font.color.rgb = AZUL; h.font.bold = True
        h.paragraph_format.space_before = Pt(10); h.paragraph_format.space_after = Pt(4); h.paragraph_format.keep_with_next = True

    cabecalho(doc, a.numero, a.entrega)
    converte(doc, MD.read_text(encoding="utf-8"))
    doc.save(a.saida)
    print(f"{a.saida} gerado")


if __name__ == "__main__":
    main()
