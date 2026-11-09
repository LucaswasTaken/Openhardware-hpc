"""Mede o exemplo resolvido da lista (energia de deformação) e grava videos/data/exemplos.json.

    .venv\\Scripts\\python videos/data/medir_exemplos.py     (ambiente dos notebooks: numba e, se houver GPU, cupy)

O mesmo exemplo aparece em exercicios/lista.md, no vídeo 12 (Numba variando as threads)
e no vídeo 13 (escada de speedup até a GPU com CuPy)."""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

SAIDA = Path(__file__).with_name("exemplos.json")
N = 50_000_000            # 50 milhões de molas (≈ 800 MB em dois vetores float64)


def melhor(f, *a, rep=5, sync=lambda: None):
    f(*a); sync()                                   # aquecimento (compila / aloca)
    ts = []
    for _ in range(rep):
        sync(); t = time.perf_counter(); f(*a); sync()
        ts.append(time.perf_counter() - t)
    return min(ts)


# ---- energia de deformação U = ½ Σ k_i δ_i²
def energia_python(k, d):
    u = 0.0
    for i in range(len(k)):
        u += 0.5 * k[i] * d[i] * d[i]
    return u


def energia_numpy(k, d):
    return 0.5 * np.dot(k, d * d)


@njit(parallel=True)
def energia_numba(k, d):
    u = 0.0
    for i in prange(k.size):
        u += 0.5 * k[i] * d[i] * d[i]
    return u


def nucleos():
    return [p for p in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if p <= get_num_threads()]


def medir():
    rng = np.random.default_rng(0)
    k, d = rng.uniform(1e6, 1e8, N), rng.normal(0, 1e-3, N)
    ref = energia_numpy(k, d)

    # Python puro: 1 milhão de molas e multiplica por 50 (rodar os 50 milhões levaria minutos)
    t_py = melhor(energia_python, k[:1_000_000], d[:1_000_000], rep=1) * (N / 1_000_000)
    t_np = melhor(energia_numpy, k, d)

    # escalabilidade forte: o MESMO problema, 1, 2, 4… threads
    pmax = get_num_threads()
    threads = {}
    for p in nucleos():
        set_num_threads(p)
        assert np.isclose(energia_numba(k, d), ref, rtol=1e-9)
        threads[str(p)] = melhor(energia_numba, k, d)
    set_num_threads(pmax)

    out = {"n": N, "ref": ref, "python_s": t_py, "numpy_s": t_np, "threads_s": threads,
           "nucleos": pmax, "cpu": platform.processor() or platform.machine(),
           "banda_gbs": 2 * N * 8 / threads[str(pmax)] / 1e9}

    # degrau GPU: CuPy com o dado já na GPU e incluindo a cópia RAM → VRAM
    try:
        import cupy as cp
        sync = cp.cuda.Device().synchronize

        def energia_cupy(kg, dg):
            return 0.5 * cp.dot(kg, dg * dg)

        def energia_cupy_com_copia(k, d):
            return float(energia_cupy(cp.asarray(k), cp.asarray(d)))

        kg, dg = cp.asarray(k), cp.asarray(d)
        assert np.isclose(float(energia_cupy(kg, dg)), ref, rtol=1e-9)
        out["cupy_s"] = melhor(energia_cupy, kg, dg, sync=sync)
        out["cupy_copia_s"] = melhor(energia_cupy_com_copia, k, d, rep=3, sync=sync)
        nome = cp.cuda.runtime.getDeviceProperties(0)["name"]
        out["gpu"] = nome.decode() if isinstance(nome, bytes) else nome
    except Exception as e:  # noqa: BLE001 - sem GPU: só a parte de CPU
        out["gpu_erro"] = str(e)
    return out


if __name__ == "__main__":
    dados = {"energia": medir()}
    SAIDA.write_text(json.dumps(dados, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dados, indent=1, ensure_ascii=False))
