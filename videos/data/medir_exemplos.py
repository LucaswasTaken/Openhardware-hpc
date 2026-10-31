"""Mede os exemplos resolvidos dos vídeos 12–14 e grava videos/data/exemplos.json.

    python videos/data/medir_exemplos.py      (no ambiente dos notebooks: .venv com numba, joblib e, se houver, cupy)

São problemas PARECIDOS com os da lista, não iguais: energia de deformação (≈ produto escalar), probabilidade de
falha com CuPy (≈ Monte Carlo para π) e busca de nós numa região com joblib (≈ busca em array)."""

import json
import os
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange

SAIDA = Path(__file__).with_name("exemplos.json")


def melhor(f, *a, rep=5):
    f(*a)
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); f(*a); ts.append(time.perf_counter() - t)
    return min(ts)


# ---- 12: energia de deformação U = ½ Σ k_i δ_i²  (Numba)
def energia_python(k, d):
    u = 0.0
    for i in range(len(k)):
        u += 0.5 * k[i] * d[i] * d[i]
    return u


@njit
def energia_numba(k, d):
    u = 0.0
    for i in range(k.size):
        u += 0.5 * k[i] * d[i] * d[i]
    return u


@njit(parallel=True)
def energia_numba_par(k, d):
    u = 0.0
    for i in prange(k.size):
        u += 0.5 * k[i] * d[i] * d[i]
    return u


def ex12():
    rng = np.random.default_rng(0)
    n = 50_000_000
    k, d = rng.uniform(1e6, 1e8, n), rng.normal(0, 1e-3, n)
    t_py = melhor(energia_python, k[:1_000_000], d[:1_000_000], rep=1) * 50
    t_np = melhor(lambda: 0.5 * np.dot(k, d * d))
    t_nb = melhor(energia_numba, k, d)
    t_nbp = melhor(energia_numba_par, k, d)
    gbs = 2 * n * 8 / t_nbp / 1e9
    return {"n": n, "python_s": t_py, "numpy_s": t_np, "numba_s": t_nb, "numba_par_s": t_nbp,
            "banda_gbs": gbs, "nucleos": os.cpu_count()}


# ---- 13: probabilidade de falha com CuPy
def pf_numpy(n, seed=1):
    rng = np.random.default_rng(seed)
    R = rng.lognormal(np.log(350.0), 0.10, n); S = rng.gumbel(200.0, 25.0, n)
    return np.count_nonzero(R < S) / n


def ex13():
    out = {"cpu": {}, "gpu": {}, "convergencia": []}
    for n in (10**6, 10**7, 5 * 10**7):
        out["cpu"][str(n)] = melhor(pf_numpy, n, rep=2)
    try:
        import cupy as cp

        def pf_cupy(n, seed=1):
            rng = cp.random.default_rng(seed)
            # lognormal = exp(normal); Gumbel por inversão: loc − scale·ln(−ln U)
            R = cp.exp(np.log(350.0) + 0.10 * rng.standard_normal(n))
            S = 200.0 - 25.0 * cp.log(-cp.log(rng.random(n)))
            r = int(cp.count_nonzero(R < S)) / n
            return r

        for n in (10**6, 10**7, 5 * 10**7):
            pf_cupy(n); cp.cuda.Device().synchronize()
            out["gpu"][str(n)] = melhor(pf_cupy, n, rep=3)
        ref = pf_cupy(2 * 10**8, seed=99)
        for n in (10**3, 10**4, 10**5, 10**6, 10**7):
            erros = [abs(pf_cupy(n, seed=s) - ref) / ref for s in range(20)]
            out["convergencia"].append({"n": n, "erro_medio": float(np.mean(erros))})
        out["pf"] = ref
        out["gpu_nome"] = cp.cuda.runtime.getDeviceProperties(0)["name"]
        out["gpu_nome"] = out["gpu_nome"].decode() if isinstance(out["gpu_nome"], bytes) else out["gpu_nome"]
    except Exception as e:  # noqa: BLE001 - sem GPU: só a parte de CPU
        out["gpu_erro"] = str(e)
    return out


# ---- 14: nós de uma malha dentro de uma região (joblib)
def dentro_regiao(xy, cx=500.0, cy=500.0, r=120.0):
    d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
    return np.flatnonzero(d2 <= r * r)


def nos_serial_loop(xy, cx=500.0, cy=500.0, r=120.0):
    out = []
    for i in range(len(xy)):
        dx, dy = xy[i, 0] - cx, xy[i, 1] - cy
        if dx * dx + dy * dy <= r * r:
            out.append(i)
    return out


def bloco(xy, inicio):
    """Cada processo devolve os índices locais + deslocamento do bloco (para juntar no fim)."""
    return dentro_regiao(xy) + inicio


def ex14():
    rng = np.random.default_rng(3)
    n = 40_000_000
    xy = rng.random((n, 2)) * 1000
    t_loop = melhor(nos_serial_loop, xy[:500_000], rep=1) * (n / 500_000)
    t_np = melhor(dentro_regiao, xy, rep=3)
    tempos = {"processos": {}, "threads": {}}
    for prefer, chave in ((None, "processos"), ("threads", "threads")):
        for jobs in (1, 2, 4, os.cpu_count()):
            partes = np.array_split(np.arange(n), jobs * 4)

            def roda():  # processos: cada bloco é COPIADO para o processo; threads: compartilham a memória
                res = Parallel(n_jobs=jobs, prefer=prefer)(delayed(bloco)(xy[p[0]:p[-1] + 1], p[0]) for p in partes)
                return np.concatenate(res)
            assert np.array_equal(roda(), dentro_regiao(xy))
            tempos[chave][str(jobs)] = melhor(roda, rep=2)
    return {"n": n, "loop_s": t_loop, "numpy_s": t_np, "joblib_s": tempos, "achados": int(dentro_regiao(xy).size)}


if __name__ == "__main__":
    dados = {"ex12": ex12(), "ex13": ex13(), "ex14": ex14()}
    SAIDA.write_text(json.dumps(dados, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dados, indent=1, ensure_ascii=False))
