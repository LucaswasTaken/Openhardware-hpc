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


def aula2():
    d, quem = carrega(2)
    if not d:
        return
    gpu = d.get("gpu", {}).get("nome", "GPU")
    mu = d.get("memoria_unificada", {})
    if mu:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        c, g = mu["copia_ms"], mu["gerenciada_ms"]
        rot = list(c) + list(g)
        barras(ax, rot, list(c.values()) + list(g.values()),
               [VERMELHO, VERDE, VERMELHO] + [VERMELHO, VERDE, AMARELO, VERMELHO], "{:.0f} ms", log=True)
        ax.set(xlabel=f"ms (log) · {gpu} · {mu['mib']:.0f} MiB", title="Cópia explícita × memória gerenciada")
        salva(fig, "memoria_unificada.png")

    sv = d.get("soma_vetores", {})
    if sv:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        barras(ax, ["CPU (NumPy)", "GPU, só a conta", "GPU, com as cópias"],
               [sv["cpu_ms"], sv["gpu_ms"], sv["gpu_com_copias_ms"]], [AZUL, VERDE, VERMELHO], "{:.1f} ms")
        ax.set(xlabel="ms", title=f"Soma de 20 M floats — {gpu}")
        salva(fig, "transferencias.png")

    mm = d.get("matmul", {}).get("linhas", [])
    if mm:
        fig, ax = plt.subplots(figsize=(7, 4))
        ns = sorted({l["n"] for l in mm})
        cpu = [2 * n**3 / next(l["cpu_s"] for l in mm if l["n"] == n) / 1e9 for n in ns]
        for tipo, cor in (("float32", VERDE), ("float64", VERMELHO)):
            ax.plot(ns, [2 * n**3 / next(l["gpu_s"] for l in mm if l["n"] == n and l["tipo"] == tipo) / 1e9 for n in ns],
                    "o-", color=cor, lw=2, label=f"GPU {tipo}")
        ax.plot(ns, cpu, "s--", color=AZUL, lw=2, label="CPU float32")
        ax.set(xscale="log", yscale="log", xlabel="n (matriz n×n)", ylabel="GFLOP/s", title=f"Multiplicação de matrizes — {gpu}")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.legend()
        salva(fig, "matmul.png")

    cal = d.get("calor", {}).get("tempos", {})
    if cal:
        fig, ax = plt.subplots(figsize=(7, 4))
        cores = {"Python puro": CINZA, "NumPy": AZUL, "Numba paralelo": AMARELO, "CuPy (GPU)": VERMELHO,
                 "Numba-CUDA kernel (GPU)": VERDE}
        passos = d["calor"]["passos"]
        for k, v in cal.items():
            if v:
                n = [int(x) for x in v]
                ax.plot([x * x for x in n], [passos * x * x / t / 1e6 for x, t in zip(n, v.values())], "o-",
                        color=cores.get(k, CINZA), lw=2, label=k)
        ax.set(xscale="log", yscale="log", xlabel="pontos da malha", ylabel="M pontos·passo/s",
               title=f"Equação do calor 2D — {gpu}")
        ax.legend(fontsize=9)
        salva(fig, "calor.png")


if __name__ == "__main__":
    aula1()
    aula2()
