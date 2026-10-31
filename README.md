# Computação de Alto Desempenho em Python
## Curso para Engenharia Civil — 2 aulas de 3h30 (CPU e GPU)

[![Aula 1 no Colab](https://img.shields.io/badge/Aula_1_CPU-Abrir_no_Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/LucaswasTaken/Openhardware-hpc/blob/master/notebooks/Aula1_CPU.ipynb)
[![Aula 2 no Colab](https://img.shields.io/badge/Aula_2_GPU-Abrir_no_Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/LucaswasTaken/Openhardware-hpc/blob/master/notebooks/Aula2_GPU.ipynb)
[![Vídeos](https://img.shields.io/badge/Vídeos-curtos-red?logo=youtube)](https://lucaswastaken.github.io/Openhardware-hpc/)
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)

Introdução prática à **computação de alto desempenho (HPC)** em Python para estudantes de Engenharia Civil: medir, entender os limites (Amdahl, memória, transferências) e acelerar problemas reais — da CPU com vários núcleos à GPU.

**Tudo roda no Google Colab, sem instalar nada.** Cada bloco das aulas tem um **vídeo curto** (1–3 min) com a ideia central animada.

---

## 📚 Estrutura

### 🧩 Aula 1 — Paralelismo e desempenho na CPU (3h30) · [notebook](notebooks/Aula1_CPU.ipynb)
| Tempo | Bloco | Exemplo de engenharia |
|---|---|---|
| 0:00–0:20 | Medir antes de otimizar (`perf_counter`, `%timeit`, `cProfile`, `line_profiler`) | envoltória de momentos numa viga |
| 0:20–0:50 | Speedup, eficiência, Amdahl, Gustafson, escalabilidade forte/fraca | — |
| 0:50–1:30 | Threads × processos, GIL e **Python 3.14 sem GIL** | probabilidade de falha por Monte Carlo |
| 1:45–2:30 | Vetorização, BLAS e `joblib` | estudo paramétrico de vigas (MEF) |
| 2:30–3:10 | Numba (`@njit`, `prange`) | matriz de rigidez de uma treliça com 2 milhões de barras |
| 3:10–3:30 | Exercício guiado e lista de exercícios | verificação de tensões em 1 milhão de seções |

### ⚡ Aula 2 — GPU: paralelismo massivo (3h30) · [notebook](notebooks/Aula2_GPU.ipynb)
| Tempo | Bloco | Exemplo |
|---|---|---|
| 0:00–0:25 | CPU × GPU: núcleos, warps (SIMT), memória | propriedades da sua GPU |
| 0:25–0:50 | **CPU e GPU ainda são mundos separados?** Memória unificada (Apple M, Grace Hopper, MI300A) × GPU discreta | cópia explícita × memória gerenciada × prefetch |
| 0:50–1:05 | CuPy e como medir direito (aquecer, sincronizar) | soma de vetores; matmul `float32` × `float64` |
| 1:05–1:45 | Kernels: `ElementwiseKernel` e Numba-CUDA (grid, block, thread) | tensão de von Mises; Monte Carlo |
| 2:00–3:00 | Projeto: equação do calor 2D, do Python puro à GPU, com **um código para CPU e GPU** (Array API) | laje em incêndio |
| 3:00–3:30 | Quando usar GPU? Checklist e exercícios | — |

---

## 🎬 Vídeos

Vídeos curtos, com animações e narração, para rever os conceitos — **[assista aqui](https://lucaswastaken.github.io/Openhardware-hpc/)**.

| # | Vídeo | Aula |
|---|---|---|
| 01 | Por que paralelizar e como medir | 1 |
| 02 | Speedup, eficiência e a lei de Amdahl | 1 |
| 03 | Threads × processos, GIL e Python sem GIL | 1 |
| 04 | Monte Carlo em paralelo com processos | 1 |
| 05 | joblib: estudo paramétrico em 3 linhas | 1 |
| 06 | Numba: do laço lento ao código compilado | 1 |
| 07 | CPU × GPU: milhares de núcleos simples | 2 |
| 08 | O gargalo das transferências | 2 |
| 08b | CPU e GPU: dois mundos ou um só? | 2 |
| 09 | Grid, block e thread: escrevendo um kernel | 2 |
| 10 | Equação do calor na GPU, passo a passo | 2 |
| 11 | Como fazer o relatório | Exercícios |
| 12–14 | Exemplos resolvidos (energia de deformação com Numba, probabilidade de falha com CuPy, busca de nós com joblib) | Exercícios |

---

## 📝 Exercícios

[`exercicios/lista.md`](exercicios/lista.md) ([.docx](exercicios/lista.docx)): escolha **um** de 8 exercícios (ou proponha o seu) e **uma** tecnologia, e entregue um relatório com speedup, eficiência e escalabilidade. O [modelo de relatório](exercicios/modelo_relatorio.ipynb) já mede e desenha os gráficos.

---

## 🚀 Como usar

### No Colab (recomendado)
Clique nos botões no topo. Na Aula 2, escolha *Ambiente de execução → Alterar o tipo → **T4 GPU***.

> O Colab gratuito tem **2 núcleos de CPU**: o speedup máximo na CPU fica perto de 2×. Isso faz parte da lição — rode também na sua máquina e compare.

### No seu computador
```bash
git clone https://github.com/LucaswasTaken/Openhardware-hpc && cd Openhardware-hpc
uv venv --python 3.12 && uv pip install -r requirements.txt
# GPU NVIDIA (opcional):
uv pip install "cupy-cuda12x[ctk]" "numba-cuda[cu12]"
uv run jupyter lab
```
Ou com conda: `conda env create -f environment.yml`.

---

## 📁 Organização

```
notebooks/            Aula1_CPU.ipynb, Aula2_GPU.ipynb (publicados já executados)
  results/            números medidos ao rodar as aulas (usados nos slides e vídeos)
exercicios/           lista.md, lista.docx, modelo_relatorio.ipynb
presentations/        slides das 2 aulas (Marp) e o script que gera as figuras
videos/               roteiros, cenas (Manim) e o script que gera os vídeos
docs/                 página dos vídeos (GitHub Pages)
tools/                scripts que geram e executam os notebooks
deprecated/           versão anterior (3 aulas de 2h, 2025) e o material MPI/OpenCL de 2020
```

Os notebooks são gerados por `tools/build_aula1.py` e `tools/build_aula2.py` — edite os scripts e rode `python tools/build_aula1.py && python tools/run_nb.py notebooks/Aula1_CPU.ipynb`.

---

## 📞 Contato

**Instrutor:** Lucas Gouveia Omena Lopes · dúvidas e sugestões: [issues](https://github.com/LucaswasTaken/Openhardware-hpc/issues).

**Última atualização:** outubro de 2026 — versão 2 (2 aulas de 3h30).
