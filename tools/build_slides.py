"""Exporta os slides Marp (presentations/*.md) para apresentar:

    python tools/build_slides.py

- docs/slides/<aula>.html  → abre no navegador (GitHub Pages): F para tela cheia, setas para avançar,
                             P para o modo apresentador (com notas e próximo slide)
- presentations/<aula>.pdf → reserva para projetar sem internet

Requer Node.js (usa `npx @marp-team/marp-cli`) e Chrome ou Edge instalado (para o PDF)."""

import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PRES = RAIZ / "presentations"
SLIDES = RAIZ / "docs" / "slides"
MARP = ["npx", "--yes", "@marp-team/marp-cli@4.1.2"]


def marp(*args) -> None:
    subprocess.run(MARP + list(args), check=True, shell=sys.platform == "win32")


def main() -> None:
    SLIDES.mkdir(parents=True, exist_ok=True)
    destino_fig = SLIDES / "figures"
    if destino_fig.exists():
        shutil.rmtree(destino_fig)
    shutil.copytree(PRES / "figures", destino_fig, ignore=shutil.ignore_patterns("*.py"))
    for md in sorted(PRES.glob("Aula*.md")):
        marp(str(md), "--html", "-o", str(SLIDES / f"{md.stem}.html"))
        marp(str(md), "--pdf", "--allow-local-files", "-o", str(PRES / f"{md.stem}.pdf"))
        print(f"{md.stem}: docs/slides/{md.stem}.html · presentations/{md.stem}.pdf")


if __name__ == "__main__":
    main()
