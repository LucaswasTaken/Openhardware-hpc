"""Renderiza o vídeo-roteiro da Aula 2 (só para o professor) e junta os capítulos num MP4 só, com legenda.

    python videos/professor.py              # 720p → videos/professor/aula2_professor.mp4 (+ .srt)
    python videos/professor.py --rascunho   # 480p, para conferir

Não vai para o site (fica fora de render.VIDEOS) e a pasta videos/professor/ está no .gitignore."""

import re
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARQ = "scenes/professor_aula2.py"
CAPITULOS = ["P00Abertura", "P01Preparacao", "P02Bloco1", "P03Bloco2", "P04Bloco3", "P05Bloco4", "P06Bloco5",
             "P07Bloco6", "P08PlanoB"]


def tempo_srt(s: float) -> str:
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def segundos(t: str) -> float:
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def duracao(mp4: Path) -> float:
    import av
    with av.open(str(mp4)) as c:
        return c.duration / av.time_base


def main(args: list[str]) -> int:
    import imageio_ffmpeg

    rascunho = "--rascunho" in args
    qualidade, pasta = ("-ql", "480p15") if rascunho else ("-qm", "720p30")
    saida = AQUI / "media" / "videos" / Path(ARQ).stem / pasta
    destino = AQUI / "professor"
    destino.mkdir(exist_ok=True)

    for cap in CAPITULOS:
        print(f"=== {cap}", flush=True)
        r = subprocess.run([sys.executable, "-m", "manim", "render", qualidade, "--disable_caching", "--media_dir",
                            "media", ARQ, cap], cwd=AQUI, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for linha in (r.stdout + r.stderr).splitlines():
            if "FORA DA TELA" in linha or "Error" in linha or "Traceback" in linha:
                print("  ", linha.strip())
        if r.returncode or not (saida / f"{cap}.mp4").is_file():
            print(r.stderr[-2000:])
            return 1

    # junta os vídeos (concat demuxer, sem recodificar) e as legendas (com deslocamento de tempo)
    lista = destino / "capitulos.txt"
    lista.write_text("".join(f"file '{(saida / f'{c}.mp4').as_posix()}'\n" for c in CAPITULOS), encoding="utf-8")
    final = destino / "aula2_professor.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lista), "-c", "copy", str(final)], check=True)
    lista.unlink()

    blocos, inicio, k = [], 0.0, 1
    for c in CAPITULOS:
        srt = saida / f"{c}.srt"
        if srt.is_file():
            for b in re.split(r"\n\s*\n", srt.read_text(encoding="utf-8").strip()):
                linhas = b.splitlines()
                if len(linhas) < 3 or "-->" not in linhas[1]:
                    continue
                a, z = (segundos(x.strip()) + inicio for x in linhas[1].split("-->"))
                blocos.append(f"{k}\n{tempo_srt(a)} --> {tempo_srt(z)}\n" + "\n".join(linhas[2:]))
                k += 1
        inicio += duracao(saida / f"{c}.mp4")
    final.with_suffix(".srt").write_text("\n\n".join(blocos) + "\n", encoding="utf-8")
    print(f"\n{final}  ·  {inicio / 60:.1f} min  ·  legenda: {final.with_suffix('.srt').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
