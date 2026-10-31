"""Renderiza os vídeos do curso e copia MP4 + legenda (.srt) para videos/mp4/.

    python videos/render.py              # todos, 720p
    python videos/render.py 01 08b       # só alguns
    python videos/render.py --rascunho   # 480p, rápido, para conferir

Requer o ambiente dos vídeos (pyproject: extra "videos") e internet (voz neural pt-BR via edge-tts; o áudio fica em
cache em videos/media/voiceovers, então só textos novos são sintetizados de novo)."""

import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent

VIDEOS = [  # (número, arquivo da cena, classe, nome do arquivo final, título)
    ("01", "aula1.py", "V01Medir", "v01_medir", "Por que paralelizar e como medir"),
    ("02", "aula1.py", "V02Amdahl", "v02_amdahl", "Speedup, eficiência e a lei de Amdahl"),
    ("03", "aula1.py", "V03GIL", "v03_gil", "Threads × processos, GIL e Python sem GIL"),
    ("04", "aula1.py", "V04MonteCarlo", "v04_monte_carlo", "Monte Carlo em paralelo com processos"),
    ("05", "aula1.py", "V05Joblib", "v05_joblib", "joblib: estudo paramétrico em 3 linhas"),
    ("06", "aula1.py", "V06Numba", "v06_numba", "Numba: do laço lento ao código compilado"),
    ("07", "aula2.py", "V07CpuGpu", "v07_cpu_gpu", "CPU × GPU: milhares de núcleos simples"),
    ("08", "aula2.py", "V08Transferencias", "v08_transferencias", "O gargalo das transferências"),
    ("08b", "v08b_dois_mundos.py", "DoisMundos", "v08b_dois_mundos", "CPU e GPU: dois mundos ou um só?"),
    ("09", "aula2.py", "V09Kernel", "v09_kernel", "Grid, block e thread: escrevendo um kernel"),
    ("10", "aula2.py", "V10Calor", "v10_calor", "Equação do calor na GPU, passo a passo"),
    ("11", "exercicios.py", "V11Relatorio", "v11_relatorio", "Como fazer o relatório"),
    ("12", "exercicios.py", "V12Energia", "v12_energia_numba", "Exemplo: energia de deformação com Numba"),
    ("13", "exercicios.py", "V13Falha", "v13_falha_cupy", "Exemplo: probabilidade de falha com CuPy"),
    ("14", "exercicios.py", "V14Busca", "v14_busca_joblib", "Exemplo: quando paralelizar não ajuda (e como resolver)"),
]


def main(args: list[str]) -> int:
    rascunho = "--rascunho" in args
    quais = [a for a in args if not a.startswith("--")]
    qualidade, pasta = ("-ql", "480p15") if rascunho else ("-qm", "720p30")
    destino = AQUI / "mp4"
    destino.mkdir(exist_ok=True)
    falhas = []
    for num, arq, classe, nome, _titulo in VIDEOS:
        if quais and num not in quais:
            continue
        print(f"\n=== {num} {classe} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", "manim", "render", qualidade, "--disable_caching", "--media_dir",
                            "media", f"scenes/{arq}", classe], cwd=AQUI, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        for linha in (r.stdout + r.stderr).splitlines():
            if "FORA DA TELA" in linha or "Error" in linha or "Traceback" in linha:
                print("  ", linha.strip())
        saida = AQUI / "media" / "videos" / Path(arq).stem / pasta
        mp4, srt = saida / f"{classe}.mp4", saida / f"{classe}.srt"
        if r.returncode or not mp4.is_file():
            falhas.append(num)
            print(r.stderr[-2000:])
            continue
        if not rascunho:
            shutil.copy2(mp4, destino / f"{nome}.mp4")
            if srt.is_file():
                shutil.copy2(srt, destino / f"{nome}.srt")
        print(f"   ok → {mp4.stat().st_size / 2**20:.1f} MiB")
    print("\nfalharam:", falhas or "nenhum")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
