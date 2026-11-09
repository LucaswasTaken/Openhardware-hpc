"""Gera exercicios/lista.docx: a lista do professor (exercicios/assets/lista_base.docx), no formato original,
com correções pontuais, e um Anexo (exercicios/lista_anexo.md) com o passo a passo e o exemplo resolvido.

    python tools/build_lista_docx.py

Para mudar data, número da lista ou perguntas, edite o lista_base.docx no Word (ou as CORRECOES abaixo)."""

import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "exercicios" / "assets" / "lista_base.docx"
ANEXO = RAIZ / "exercicios" / "lista_anexo.md"
SAIDA = RAIZ / "exercicios" / "lista.docx"
SITE_REPO = "https://github.com/LucaswasTaken/Openhardware-hpc/blob/master/exercicios/"
FONTE_COD = "Consolas"

# ---------------------------------------------------------------- correções na lista do professor
# (início do parágrafo, novos textos dos runs) — os runs mantêm a formatação original (rótulo em negrito etc.)
CORRECOES = [
    ("Joblib", ["Numba", ": paralelização na CPU com @njit(parallel=True) e prange (recomendado)"]),
    ("Numba", ["CuPy", ": computação em GPU (troque np por cp)"]),
    ("CuPy", ["Joblib / Multiprocessing", ": paralelização na CPU com processos"]),
    ("Escalabilidade = ", ["Escalabilidade forte", ": mesmo tamanho de problema, variando o número de threads/processos "
                           "(na GPU: comparar com as versões de CPU)"]),
    ("Weak scaling", ["Strong scaling: número de pontos fixo, variar processos"]),
]
APAGAR = ["Multiprocessing: Paralelização tradicional", "Dask"]
TROCAS = [("dos 5 Exercícios", "dos 8 Exercícios"), ("Produto Vetorial", "Produto Escalar"), ("momente", "momento"),
          ("ponto dentor de", "ponto dentro de"), ("� Exercício 3", "🎲 Exercício 3")]
# exercícios 6–8: o texto de especificação/requisitos/análise estava copiado do 5
SECOES = {
    "Exercício 6": ["Tamanhos: 1M, 10M e 20M elementos; comparar com np.sort",
                    "Versão serial: np.sort ou um algoritmo próprio (ex.: merge sort)",
                    "Paralelizar ordenando blocos (chunks) do array ao mesmo tempo",
                    "Combinar os blocos ordenados (intercalação final)",
                    "Medir throughput (elementos ordenados por segundo)",
                    "Strong scaling: tamanho fixo, variar processos",
                    "Custo da intercalação final (parte que não paraleliza)",
                    "Comparar com np.sort"],
    "Exercício 7": ["Pontos: 1M; polígono com 20, 200 e 2000 vértices",
                    "Versão serial: algoritmo do raio (tiro), ponto a ponto",
                    "Paralelizar dividindo os pontos em chunks",
                    "Combinar os resultados (dentro/fora de cada ponto)",
                    "Medir throughput (pontos por segundo)",
                    "Strong scaling: número de pontos fixo, variar processos",
                    "Efeito do número de vértices no speedup",
                    "Validar com matplotlib.path.Path.contains_points"],
    "Exercício 8": ["Pontos: 1M e 10M, aleatórios no plano",
                    "Versão serial: Monotone Chain ou Quickhull",
                    "Paralelizar calculando o fecho de cada chunk",
                    "Combinar: calcular o fecho dos fechos",
                    "Validar com scipy.spatial.ConvexHull",
                    "Strong scaling: número de pontos fixo, variar processos",
                    "Quantos pontos sobram após a 1ª etapa",
                    "Custo da etapa final (serial)"],
}
AVISO_ANEXO = " Veja o Anexo, ao final: passo a passo, escalabilidade forte e um exemplo resolvido."


def textos(p, novos):
    """Põe `novos` nos primeiros runs do parágrafo e esvazia os demais (mantém a formatação de cada run)."""
    for r, t in zip(p.runs, novos + [""] * len(p.runs)):
        r.text = t


def tem_lista(p):
    pr = p._p.pPr
    return pr is not None and pr.numPr is not None


def corrige(doc):
    ps = doc.paragraphs
    feitos = set()
    for inicio, novos in CORRECOES:
        p = next(p for i, p in enumerate(ps) if i not in feitos and p.text.startswith(inicio))
        feitos.add(ps.index(p))
        textos(p, novos)
    for inicio in APAGAR:
        p = next(p for p in doc.paragraphs if p.text.startswith(inicio))
        p._p.getparent().remove(p._p)
    for p in doc.paragraphs:
        for a, b in TROCAS:
            if a in p.text:
                for r in p.runs:
                    r.text = r.text.replace(a, b).replace("�", "🎲") if ("�" in a) else r.text.replace(a, b)
        if p.text.startswith("Você deve escolher apenas UM"):
            p.runs[-1].text += AVISO_ANEXO
    ps = doc.paragraphs
    for chave, novos in SECOES.items():
        i = next(i for i, p in enumerate(ps) if chave in p.text and p.style.name.startswith("Heading 1"))
        itens = []
        for p in ps[i + 1:]:
            if p.style.name.startswith("Heading 1"):
                break
            if tem_lista(p):
                itens.append(p)
        assert len(itens) == len(novos), (chave, len(itens))
        for p, t in zip(itens, novos):
            textos(p, [t])


# ---------------------------------------------------------------- anexo (markdown simples → estilos do documento)
SUP = str.maketrans("0123456789n", "⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ")
SUB = str.maketrans("0123456789ijkn", "₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₙ")
TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\$[^$]+\$)")


def latex(s):
    s = s.replace(r"\tfrac12", "½").replace(r"\,", " ").replace(r"\sum", "Σ").replace(r"\delta", "δ")
    s = re.sub(r"\^(\w)", lambda m: m.group(1).translate(SUP), s)
    return re.sub(r"_(\w)", lambda m: m.group(1).translate(SUB), s)


def link(par, texto, url, tamanho=None):
    url = url if url.startswith("http") else SITE_REPO + url
    r_id = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                              is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), r_id)
    r = OxmlElement("w:r")
    rpr = copy.deepcopy(MODELO["corpo"]) if MODELO.get("corpo") is not None else OxmlElement("w:rPr")
    for velho in rpr.findall(qn("w:color")) + rpr.findall(qn("w:u")):
        rpr.remove(velho)
    cor = OxmlElement("w:color"); cor.set(qn("w:val"), "0563C1"); rpr.append(cor)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    if tamanho:
        for velho in rpr.findall(qn("w:sz")) + rpr.findall(qn("w:szCs")):
            rpr.remove(velho)
        for tag in ("w:sz", "w:szCs"):
            e = OxmlElement(tag); e.set(qn("w:val"), str(int(tamanho.pt * 2))); rpr.append(e)
    r.append(rpr); t = OxmlElement("w:t"); t.text = texto; t.set(qn("xml:space"), "preserve"); r.append(t)
    h.append(r); par._p.append(h)


MODELO = {}   # rPr de runs de referência da lista do professor (fonte, tamanho, cor) — preenchido em main()


def run_com(par, texto, papel):
    run = par.add_run(texto)
    base = MODELO.get(papel)
    if base is not None:
        if run._r.rPr is not None:
            run._r.remove(run._r.rPr)
        run._r.insert(0, copy.deepcopy(base))
    return run


def inline(par, texto, negrito=False, italico=False, tamanho=None, papel="corpo"):
    for parte in TOKEN.split(texto):
        if not parte:
            continue
        if parte.startswith("**"):
            inline(par, parte[2:-2], True, italico, tamanho, papel); continue
        if parte.startswith("*") and len(parte) > 2:
            inline(par, parte[1:-1], negrito, True, tamanho, papel); continue
        if parte.startswith("["):
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", parte)
            link(par, m.group(1), m.group(2), tamanho); continue
        cod = parte.startswith("`")
        it = italico or parte.startswith("$")
        parte = parte[1:-1] if cod else (latex(parte[1:-1]) if parte.startswith("$") else parte)
        run = run_com(par, parte, papel)
        if negrito:
            run.bold = True
        if it:
            run.italic = True
        if cod:
            run.font.name = FONTE_COD; run.font.size = Pt(9)
        elif tamanho:
            run.font.size = tamanho


def sombreia(celula, cor):
    tc = celula._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), cor); tc.append(sh)


def bordas(tabela):
    tblPr = tabela._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}")
        for k, v in (("val", "single"), ("sz", "4"), ("space", "0"), ("color", "A6A6A6")):
            e.set(qn(f"w:{k}"), v)
        b.append(e)
    tblPr.append(b)


def tabela(doc, linhas):
    cel = [[c.strip() for c in l.strip().strip("|").split("|")] for l in linhas if not re.match(r"^\|[\s:|-]+\|$", l.strip())]
    t = doc.add_table(rows=len(cel), cols=len(cel[0])); bordas(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, linha in enumerate(cel):
        for j, txt in enumerate(linha):
            c = t.cell(i, j); p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            inline(p, txt, negrito=(i == 0), tamanho=Pt(9.5))
            if i == 0:
                sombreia(c, "D9E2F3")
    doc.add_paragraph()


def codigo(doc, linhas):
    t = doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.cell(0, 0); sombreia(c, "F2F2F2"); p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
    for k, l in enumerate(linhas):
        r = p.add_run(l); r.font.name = FONTE_COD; r.font.size = Pt(8.5)
        if k < len(linhas) - 1:
            r.add_break()
    doc.add_paragraph()


def anexo(doc, md, numpr_marcador):
    quebra = doc.add_paragraph(); quebra.add_run().add_break(WD_BREAK.PAGE)
    linhas = md.splitlines()
    i = 0
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
        m = re.match(r"^(#{1,3}) (.*)", l)
        if m:
            nivel = 1 if len(m.group(1)) == 1 else 3          # como na lista: Heading 1 (exercício) e 3 (subtítulo)
            h = doc.add_paragraph(style=f"Heading {nivel}")
            inline(h, m.group(2), papel=f"h{nivel}"); i += 1; continue
        if l.startswith("- "):
            p = doc.add_paragraph(style="Normal"); inline(p, l[2:], papel="item")
            p._p.get_or_add_pPr().append(copy.deepcopy(numpr_marcador))
            i += 1; continue
        if l.startswith(">"):
            p = doc.add_paragraph(style="Normal (Web)"); inline(p, l.lstrip("> "), italico=True); i += 1; continue
        if l.strip():
            p = doc.add_paragraph(style="Normal (Web)"); inline(p, l)
        i += 1


def main():
    doc = Document(BASE)
    ps = doc.paragraphs
    acha = lambda inicio: next(p for p in ps if p.text.startswith(inicio))  # noqa: E731
    numpr = copy.deepcopy(acha("Joblib")._p.pPr.numPr)
    for papel, run in (("corpo", acha("Você deve escolher").runs[0]), ("h1", acha("🔢").runs[1]),
                       ("h3", acha("Especificações").runs[0]), ("item", acha("Implementar produto escalar").runs[0])):
        MODELO[papel] = copy.deepcopy(run._r.rPr)
    corrige(doc)
    anexo(doc, ANEXO.read_text(encoding="utf-8"), numpr)
    doc.save(SAIDA)
    print(f"{SAIDA} gerado")


if __name__ == "__main__":
    main()
