"""Gera as figuras dos slides a partir dos números medidos pelos notebooks (notebooks/results/*.json).

    python presentations/figures/make_figures.py

Usa o resultado do Colab quando existir (é o que a turma vê); senão, o da execução local."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AQUI = Path(__file__).resolve().parent
RES = AQUI.parents[1] / "notebooks" / "results"
plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
AZUL, VERDE, AMARELO, VERMELHO, CINZA = "#4aa3c7", "#5aa469", "#d8b33a", "#d1544b", "#8a929c"


def carrega(aula):
    for quem in ("colab", "local"):
        f = RES / f"aula{aula}_{quem}.json"
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8")), quem
    return {}, ""


def onde(quem, d):
    return "Colab" if quem == "colab" else f"máquina local ({d.get('nucleos', '?')} núcleos lógicos)"


def salva(fig, nome):
    fig.tight_layout()
    fig.savefig(AQUI / nome)
    plt.close(fig)
    print("  ", nome)


def barras(ax, rotulos, valores, cores, fmt="{:.2g}", log=False):
    y = np.arange(len(rotulos))[::-1]
    ax.barh(y, valores, color=cores)
    ax.set_yticks(y, rotulos)
    if log:
        ax.set_xscale("log")
    for yi, v in zip(y, valores):
        ax.text(v * (1.05 if log else 1) + (0 if log else max(valores) * 0.01), yi, fmt.format(v), va="center")
    ax.grid(axis="y", visible=False)


def aula1():
    d, quem = carrega(1)
    if not d:
        return
    fonte = onde(quem, d)
    # Amdahl
    fig, ax = plt.subplots(figsize=(7, 4))
    p = np.arange(1, 129)
    for f, c in ((0.5, VERMELHO), (0.9, AMARELO), (0.95, AZUL), (0.99, VERDE)):
        ax.plot(p, 1 / ((1 - f) + f / p), color=c, lw=2.2, label=f"{f:.0%} paralelo (máx. {1/(1-f):.0f}×)")
    ax.plot(p, p, "k:", label="ideal")
    ax.set(xscale="log", yscale="log", xlabel="núcleos", ylabel="speedup", title="Lei de Amdahl")
    ax.legend(fontsize=9)
    salva(fig, "amdahl.png")

    # GIL e threads
    sg = {}
    for k, v in (d.get("python_sem_gil") or {}).items():
        m = re.search(r"speedup\s+([\d.]+)", v)
        if m:
            sg[k] = float(m.group(1))
    tg = d.get("threads_gil", {})
    if tg:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        rot = ["laço Python (GIL)", "espera / I/O (GIL)"] + (["laço Python 3.14t (sem GIL)"] if "3.14t" in sg else [])
        val = [tg["cpu_serial_s"] / tg["cpu_threads_s"], tg["io_serial_s"] / tg["io_threads_s"]] + \
              ([sg["3.14t"]] if "3.14t" in sg else [])
        barras(ax, rot, val, [VERMELHO, AMARELO, VERDE][:len(val)], "{:.1f}×")
        ax.set(xlabel="speedup com 4 threads", title=f"Threads e o GIL — {fonte}")
        salva(fig, "threads_gil.png")

    # escalabilidade: Monte Carlo (processos) e joblib
    mc = d.get("monte_carlo_processos", {}).get("tempos", {})
    jl = d.get("joblib_vigas", {})
    if mc and jl:
        fig, ax = plt.subplots(figsize=(7, 4))
        ps = sorted(int(k) for k in mc)
        ax.plot(ps, [mc["1"] / mc[str(k)] for k in ps], "o-", color=AZUL, lw=2, label="Monte Carlo (processos)")
        pj = sorted(int(k) for k in jl["tempos"])
        ax.plot([1] + pj, [1] + [jl["serial_s"] / jl["tempos"][str(k)] for k in pj], "s-", color=VERDE, lw=2,
                label="estudo de vigas (joblib)")
        top = max(ps + pj)
        ax.plot([1, top], [1, top], "k:", label="ideal")
        ax.set(xlabel="processos", ylabel="speedup", title=f"Escalabilidade forte — {fonte}")
        ax.legend()
        salva(fig, "escalabilidade_cpu.png")

    nb = d.get("numba_trelica", {})
    if nb:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        py = nb["python_s"]
        barras(ax, ["Python puro", "NumPy", "Numba (1 núcleo)", f"Numba paralelo ({nb['threads']} threads)"],
               [1, py / nb["numpy_s"], py / nb["numba_s"], py / nb["numba_par_s"]], [CINZA, AZUL, AMARELO, VERDE],
               "{:.0f}×", log=True)
        ax.set(xlabel="vezes mais rápido que Python puro (log)", title=f"Treliça com 2 M barras — {fonte}")
        salva(fig, "numba_trelica.png")


def dois_mundos():
    """Diagrama (não depende de medidas): GPU separada × memória unificada."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    def caixa(ax, x, y, w, h, txt, cor):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=cor, ec="k", lw=1.2, alpha=.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)
    for ax in axs:
        ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis("off")
    a = axs[0]; a.set_title("GPU separada (Colab, placas GeForce/RTX)", fontsize=12)
    caixa(a, .3, 3.3, 3.6, 1.8, "CPU", AZUL); caixa(a, .3, .5, 3.6, 1.8, "RAM", "#cfe6f2")
    caixa(a, 6.1, 3.3, 3.6, 1.8, "GPU", VERDE); caixa(a, 6.1, .5, 3.6, 1.8, "VRAM", "#d7ecd9")
    a.add_patch(FancyArrowPatch((4.0, 1.4), (6.0, 1.4), arrowstyle="<|-|>", mutation_scale=18, lw=2.5, color=VERMELHO))
    a.text(5.0, 2.0, "PCIe 🚚\nviagem cara", ha="center", fontsize=10, color=VERMELHO)
    b = axs[1]; b.set_title("Memória unificada (Apple M, Grace Hopper, MI300A)", fontsize=12)
    caixa(b, .6, 3.3, 3.8, 1.8, "CPU", AZUL); caixa(b, 5.6, 3.3, 3.8, 1.8, "GPU", VERDE)
    caixa(b, .6, .5, 8.8, 1.8, "uma memória só — sem viagem", "#efe3b8")
    salva(fig, "dois_mundos.png")


def aula2():
    dois_mundos()
    d, quem = carrega(2)
    if not d:
        return
    gpu = d.get("gpu", {}).get("nome", "GPU")

    cc = d.get("copia_calculo", {})
    if cc:
        fig, ax = plt.subplots(figsize=(7, 2.9))
        barras(ax, ["🚚 RAM → GPU", "conta leve", "conta pesada", "🚚 GPU → RAM"],
               [cc["ida_ms"], cc["leve_ms"], cc["pesada_ms"], cc["volta_ms"]], [VERMELHO, VERDE, AZUL, VERMELHO], "{:.1f} ms")
        ax.set(xlabel="milissegundos", title=f"{cc['mb']:.0f} MB: viagem × conta — {gpu}")
        salva(fig, "copia_calculo.png")

    mm = d.get("matmul", {}).get("tempos_s", {})
    if mm:
        n = d["matmul"]["n"]
        fig, ax = plt.subplots(figsize=(7, 2.6))
        barras(ax, list(mm), [2 * n**3 / t / 1e9 for t in mm.values()], [AZUL, VERMELHO, VERDE][:len(mm)], "{:.0f} GFLOP/s")
        ax.set(xlabel="GFLOP/s (bilhões de contas por segundo)", title=f"Matrizes {n}×{n} — {gpu}")
        salva(fig, "matmul.png")

    cal = d.get("calor", {}).get("tempos", {})
    if cal:
        fig, ax = plt.subplots(figsize=(7, 4))
        cores = [AZUL, AMARELO, VERDE]
        for (k, v), cor in zip(cal.items(), cores):
            if v:
                n = [int(x) for x in v]
                ax.plot([x * x for x in n], list(v.values()), "o-", color=cor, lw=2, label=k)
        ax.set(xscale="log", yscale="log", xlabel="pontos da malha", ylabel=f"tempo de {d['calor']['passos']} passos [s]",
               title=f"Laje em incêndio — {gpu}")
        ax.legend(fontsize=9)
        salva(fig, "calor.png")

    es = d.get("escada", {}).get("tempos_s", {})
    if es:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        base = es["Python puro"]
        cores = [CINZA, CINZA, AZUL, AZUL, VERDE, VERMELHO][:len(es)]
        barras(ax, list(es), [base / t for t in es.values()], cores, "{:,.0f}×", log=True)
        ax.set(xlabel="vezes mais rápido que Python puro (log)", title=f"Escada de speedup — energia de 20 M molas")
        salva(fig, "escada.png")


if __name__ == "__main__":
    aula1()
    aula2()
