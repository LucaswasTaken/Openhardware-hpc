"""Small helper to write the course notebooks from Python (the builders in this folder are the source of truth).

    python tools/build_aula1.py   →  notebooks/Aula1_CPU.ipynb
    python tools/build_aula2.py   →  notebooks/Aula2_GPU.ipynb

Then execute them (locally or on Colab) so the published notebooks carry real outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat

REPO = "LucaswasTaken/Openhardware-hpc"
BRANCH = "master"
SITE = "https://lucaswastaken.github.io/Openhardware-hpc"


def colab_badge(path: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{path}"
    return f"[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def video(num: str, title: str, minutes: str) -> str:
    return f"> 🎬 **Vídeo ({minutes}):** [{title}]({SITE}/#v{num})"


class Notebook:
    def __init__(self, gpu: bool = False):
        self.nb = nbformat.v4.new_notebook()
        self.nb.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        }
        if gpu:
            self.nb.metadata["accelerator"] = "GPU"
            self.nb.metadata["colab"]["gpuType"] = "T4"

    def md(self, text: str) -> None:
        self.nb.cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text: str, tags: list[str] | None = None) -> None:
        cell = nbformat.v4.new_code_cell(text.strip("\n"))
        if tags:
            cell.metadata["tags"] = tags
        self.nb.cells.append(cell)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(self.nb, path)
        print(f"{path} ({len(self.nb.cells)} células)")
