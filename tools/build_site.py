"""Monta a página dos vídeos (GitHub Pages, pasta docs/):

    python tools/build_site.py

Move os MP4 renderizados (videos/mp4/) para docs/videos/, converte as legendas .srt em .vtt (formato do player do
navegador) e escreve docs/index.html com um player por vídeo. Cada vídeo tem âncora #v01, #v02… — os notebooks
apontam para esses endereços."""

import html
import re
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "videos"))
from render import VIDEOS  # noqa: E402

ORIGEM = RAIZ / "videos" / "mp4"
DOCS = RAIZ / "docs"
DESTINO = DOCS / "videos"
REPO = "https://github.com/LucaswasTaken/Openhardware-hpc"
COLAB = "https://colab.research.google.com/github/LucaswasTaken/Openhardware-hpc/blob/master"

SECOES = [("Aula 1 — CPU", ("01", "02", "03", "04", "05", "06"), f"{COLAB}/notebooks/Aula1_CPU.ipynb"),
          ("Aula 2 — GPU", ("07", "08", "08b", "09", "10"), f"{COLAB}/notebooks/Aula2_GPU.ipynb"),
          ("Exercícios", ("11", "12", "13", "14"), f"{REPO}/blob/master/exercicios/lista.md")]


def srt_para_vtt(srt: str) -> str:
    corpo = re.sub(r"(\d\d:\d\d:\d\d),(\d\d\d)", r"\1.\2", srt.replace("\r\n", "\n"))
    return "WEBVTT\n\n" + corpo


def duracao(mp4: Path) -> str:
    try:
        import av

        with av.open(str(mp4)) as c:
            s = int(c.duration / av.time_base)
        return f"{s // 60}:{s % 60:02d}"
    except Exception:  # noqa: BLE001 - sem PyAV, a página sai sem a duração
        return ""


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for f in ORIGEM.glob("*.mp4") if ORIGEM.is_dir() else []:
        shutil.move(str(f), DESTINO / f.name)
    for f in ORIGEM.glob("*.srt") if ORIGEM.is_dir() else []:
        (DESTINO / f.with_suffix(".vtt").name).write_text(srt_para_vtt(f.read_text(encoding="utf-8")), encoding="utf-8")
        f.unlink()
    if ORIGEM.is_dir() and not any(ORIGEM.iterdir()):
        ORIGEM.rmdir()

    por_num = {v[0]: v for v in VIDEOS}
    blocos = []
    for titulo, nums, link in SECOES:
        cards = []
        for n in nums:
            _, _, _, nome, t = por_num[n]
            mp4 = DESTINO / f"{nome}.mp4"
            if not mp4.is_file():
                continue
            vtt = DESTINO / f"{nome}.vtt"
            trilha = (f'<track kind="subtitles" srclang="pt-BR" label="Português" src="videos/{vtt.name}" default>'
                      if vtt.is_file() else "")
            cards.append(f"""
      <article class="card" id="v{n}">
        <video controls preload="metadata" src="videos/{mp4.name}">{trilha}</video>
        <h3><span class="num">{n}</span> {html.escape(t)}</h3>
        <p class="meta">{duracao(mp4)} · <a href="#v{n}">link</a> · <a href="videos/{mp4.name}" download>baixar</a></p>
      </article>""")
        blocos.append(f"""
    <section>
      <h2>{html.escape(titulo)} <a class="abrir" href="{link}">abrir {'no Colab' if 'colab' in link else 'a lista'} ↗</a></h2>
      <div class="grade">{''.join(cards)}
      </div>
    </section>""")

    (DOCS / "index.html").write_text(f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPC em Python — vídeos</title>
<style>
  :root {{ --bg: #ffffff; --fg: #1d2127; --muted: #5b6470; --card: #f4f6f8; --accent: #2f7d4f; --line: #dfe3e8; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #14161a; --fg: #e8eaed; --muted: #a0a7b1; --card: #1d2026; --accent: #7cc48f; --line: #2c3038; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, "Segoe UI", sans-serif; }}
  header, main {{ max-width: 1180px; margin: 0 auto; padding: 0 16px; }}
  header {{ padding-top: 32px; }}
  h1 {{ margin: 0 0 4px; font-size: 1.8rem; }}
  header p {{ color: var(--muted); margin: 0 0 8px; }}
  h2 {{ margin: 36px 0 12px; font-size: 1.3rem; display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
  .abrir {{ font-size: .9rem; font-weight: normal; }}
  a {{ color: var(--accent); }}
  .grade {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; scroll-margin-top: 16px; }}
  .card:target {{ outline: 3px solid var(--accent); }}
  video {{ width: 100%; display: block; background: #000; aspect-ratio: 16 / 9; }}
  h3 {{ font-size: 1rem; margin: 10px 12px 2px; }}
  .num {{ color: var(--muted); font-variant-numeric: tabular-nums; margin-right: 4px; }}
  .meta {{ color: var(--muted); font-size: .85rem; margin: 0 12px 12px; }}
  footer {{ max-width: 1180px; margin: 40px auto; padding: 0 16px; color: var(--muted); font-size: .85rem; }}
</style>
</head>
<body>
<header>
  <h1>Computação de Alto Desempenho em Python</h1>
  <p>Vídeos curtos do curso para Engenharia Civil — 2 aulas de 3h30 (CPU e GPU). Os números nos gráficos foram medidos nos próprios notebooks.</p>
  <p><a href="{REPO}">repositório</a> · <a href="{COLAB}/notebooks/Aula1_CPU.ipynb">Aula 1 no Colab</a> · <a href="{COLAB}/notebooks/Aula2_GPU.ipynb">Aula 2 no Colab</a></p>
</header>
<main>{''.join(blocos)}
</main>
<footer>Vídeos gerados com Manim e voz sintética (pt-BR); legendas disponíveis no player. Código-fonte das cenas em <a href="{REPO}/tree/master/videos">videos/</a>.</footer>
</body>
</html>
""", encoding="utf-8")
    print(f"{DOCS / 'index.html'}  ·  {len(list(DESTINO.glob('*.mp4')))} vídeos")


if __name__ == "__main__":
    main()
