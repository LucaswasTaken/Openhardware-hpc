# Slides

Os slides são escritos em [Marp](https://marp.app/): Markdown com `---` separando os slides. No GitHub, o `.md` aparece como texto comum (o cabeçalho `marp: true …` vira uma tabela). Para **apresentar**, use uma das opções abaixo.

## Como apresentar

**1. No navegador (mais simples)** — versão já exportada, publicada na página do curso:
- Aula 1: <https://lucaswastaken.github.io/Openhardware-hpc/slides/Aula1_CPU.html>
- Aula 2: <https://lucaswastaken.github.io/Openhardware-hpc/slides/Aula2_GPU.html>

Atalhos: **F** tela cheia · **setas** ou **espaço** avançam · **P** abre o modo apresentador (slide atual, próximo e cronômetro numa janela separada).

**2. PDF (sem internet)** — [`Aula1_CPU.pdf`](Aula1_CPU.pdf) e [`Aula2_GPU.pdf`](Aula2_GPU.pdf): abra em tela cheia (no Edge/Acrobat, `Ctrl+L` ou F11).

**3. No VS Code, editando** — instale a extensão *Marp for VS Code*, abra o `.md` e clique no ícone de pré-visualização (canto superior direito). Pelo ícone do Marp dá para exportar para PDF, PPTX ou HTML.

## Depois de editar um `.md`

```bash
python presentations/figures/make_figures.py   # figuras a partir dos números medidos (notebooks/results/*.json)
python tools/build_slides.py                   # exporta docs/slides/*.html e presentations/*.pdf (requer Node.js)
```

Rode os notebooks no Colab, salve os `results/aula*_colab.json` e regere as figuras e os slides para mostrar os números da máquina da turma.
