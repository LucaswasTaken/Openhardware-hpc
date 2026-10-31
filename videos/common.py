"""Shared pieces for the course videos: Portuguese neural voice (edge-tts) for manim-voiceover, colors, and the
numbers measured by the notebooks (notebooks/results/*.json), so every chart in a video shows real data."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from manim import (BLUE_C, DOWN, GREEN_C, GREY_B, LEFT, ORANGE, RED_C, RIGHT, UP, WHITE, YELLOW_C, FadeIn, FadeOut,
                   Rectangle, RoundedRectangle, Text, VGroup, Write, config)
from manim_voiceover import VoiceoverScene
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import SpeechService

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "results"
VOICE = os.environ.get("HPC_VOZ", "pt-BR-FranciscaNeural")
RATE = os.environ.get("HPC_RITMO", "+0%")   # um pouco mais lento que o piloto (+5%)
LARGURA_UTIL = 12.8                         # o quadro tem 14,2 unidades: deixa margem dos dois lados

# palette (dark background, high contrast for a projector)
CPU_C, GPU_C, MEM_C, DATA_C, OK_C, BAD_C, MUTED = BLUE_C, GREEN_C, YELLOW_C, ORANGE, GREEN_C, RED_C, GREY_B
FONT = "Segoe UI" if os.name == "nt" else "DejaVu Sans"

config.background_color = "#14161a"


def results(aula: int) -> dict:
    """Measured numbers: Colab run if available, else the local one."""
    for who in ("colab", "local"):
        f = RESULTS / f"aula{aula}_{who}.json"
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    return {}


class EdgeTTSService(SpeechService):
    """Microsoft Edge neural voices (free, needs internet). Word timings come with the audio, so bookmarks and
    subtitles line up without transcribing."""

    def __init__(self, voice: str = VOICE, rate: str = RATE, **kwargs):
        super().__init__(**kwargs)
        self.voice, self.rate = voice, rate

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        import edge_tts

        cache_dir = cache_dir or self.cache_dir
        spoken = remove_bookmarks(text)
        input_data = {"input_text": spoken, "service": "edge-tts", "voice": self.voice, "rate": self.rate}
        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached
        audio_path = path or (self.get_audio_basename(input_data) + ".mp3")

        async def synth():
            audio, words = bytearray(), []
            comm = edge_tts.Communicate(spoken, self.voice, rate=self.rate, boundary="WordBoundary")
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    words.append(chunk)
            return bytes(audio), words

        audio, words = asyncio.run(synth())
        (Path(cache_dir) / audio_path).write_bytes(audio)
        boundaries, pos = [], 0
        for w in words:  # text_offset: where the word starts in the spoken text (needed by bookmarks)
            at = spoken.find(w["text"], pos)
            at = pos if at < 0 else at
            boundaries.append({"audio_offset": int(w["offset"]), "duration_milliseconds": int(w["duration"]) // 10_000,
                               "text_offset": at, "word_length": len(w["text"]), "text": w["text"],
                               "boundary_type": "Word"})
            pos = at + len(w["text"])
        return {"input_text": text, "input_data": input_data, "original_audio": audio_path,
                "word_boundaries": boundaries}


def cabe(m, largura: float = LARGURA_UTIL):
    """Reduz o objeto se ele passar da largura (set_max_width não encolhe no Manim 0.22)."""
    if m.width > largura:
        m.scale_to_fit_width(largura)
    return m


class CursoScene(VoiceoverScene):
    """Base scene: Portuguese voice + helpers for titles and boxes."""

    def setup(self):
        super().setup()
        self.set_speech_service(EdgeTTSService())
        self.fora_da_tela: list[str] = []

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        meia_l, meia_a = config.frame_width / 2, config.frame_height / 2
        for m in self.mobjects:
            if m.width == 0 or getattr(m, "fill_opacity", 1) == 0 and getattr(m, "stroke_opacity", 1) == 0:
                continue
            if m.get_left()[0] < -meia_l - 0.01 or m.get_right()[0] > meia_l + 0.01 \
                    or m.get_bottom()[1] < -meia_a - 0.01 or m.get_top()[1] > meia_a + 0.01:
                nome = getattr(m, "text", None) or type(m).__name__
                if nome not in self.fora_da_tela:
                    self.fora_da_tela.append(nome)
                    print(f"⚠️ FORA DA TELA: {nome!r} (x {m.get_left()[0]:.2f}..{m.get_right()[0]:.2f})")

    def titulo(self, texto: str, sub: str = "") -> VGroup:
        t = cabe(Text(texto, font=FONT, weight="BOLD", font_size=46))
        g = VGroup(t)
        if sub:
            g.add(cabe(Text(sub, font=FONT, font_size=26, color=MUTED)).next_to(t, DOWN, buff=0.3))
        return g

    def topo(self, texto: str) -> Text:
        return cabe(Text(texto, font=FONT, weight="BOLD", font_size=34)).to_edge(UP, buff=0.4)

    def caixa(self, rotulo: str, cor, w=2.6, h=1.3, fs=28) -> VGroup:
        r = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=cor, fill_opacity=0.18, stroke_width=3)
        return VGroup(r, cabe(Text(rotulo, font=FONT, font_size=fs, color=WHITE), w - 0.3).move_to(r))

    def texto(self, s: str, fs=26, cor=WHITE, **kw) -> Text:
        return cabe(Text(s, font=FONT, font_size=fs, color=cor, **kw))

    def limpa(self, *keep):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.6)

    def codigo(self, fonte: str, largura: float = 9.5) -> Mobject:
        """Bloco de código Python com destaque de sintaxe, numa 'janela'."""
        c = Code(code_string=fonte.strip("\n"), language="python", formatter_style="monokai", background="window",
                 add_line_numbers=False, paragraph_config={"font": "Consolas" if os.name == "nt" else "DejaVu Sans Mono"})
        if c.width < largura:  # amplia blocos curtos (até 1,6×) para ficarem legíveis no projetor
            c.scale(min(largura / c.width, 1.6))
        return cabe(c, largura)

    def barras(self, itens: list[tuple[str, float, object]], unidade: str = "s", largura: float = 6.5,
               fmt=None, log: bool = False) -> VGroup:
        """Barras horizontais com rótulo e valor; itens = [(rótulo, valor, cor)]. Escala linear pelo maior, ou
        logarítmica (log=True) quando os valores vão de 1× a centenas de vezes."""
        import math

        fmt = fmt or (lambda v: f"{v:.3g} {unidade}")
        maior = max(v for _, v, _ in itens) or 1
        if log:
            lo, hi = math.log10(min(v for _, v, _ in itens)), math.log10(maior)
            comp = lambda v: largura * (math.log10(v) - lo + 0.25) / (hi - lo + 0.25)  # noqa: E731
        else:
            comp = lambda v: largura * v / maior  # noqa: E731
        linhas = VGroup()
        for i, (rot, v, cor) in enumerate(itens):
            y = -0.62 * i
            nome = self.texto(rot, 22).move_to([0, y, 0], aligned_edge=RIGHT)          # rótulos alinhados à direita
            barra = Rectangle(width=max(comp(v), 0.04), height=0.36, color=cor, fill_opacity=0.9,
                              stroke_width=0).move_to([0.25, y, 0], aligned_edge=LEFT)
            valor = self.texto(fmt(v), 22).next_to(barra, RIGHT, buff=0.2)
            linhas.add(VGroup(nome, barra, valor))
        return cabe(linhas.center())

    def mostra_barras(self, grupo: VGroup, run_time: float = 1.8):
        anim = [AnimationGroup(FadeIn(l[0]), GrowFromEdge(l[1], LEFT), FadeIn(l[2])) for l in grupo]
        self.play(LaggedStart(*anim, lag_ratio=0.35), run_time=run_time)

    def fonte(self, s: str) -> Text:
        return self.texto(s, 18, MUTED).to_edge(DOWN, buff=0.25)


from manim import ORIGIN, AnimationGroup, Code, GrowFromEdge, LaggedStart, Mobject  # noqa: E402

__all__ = ["CursoScene", "cabe", "results", "CPU_C", "GPU_C", "MEM_C", "DATA_C", "OK_C", "BAD_C", "MUTED", "FONT",
           "Rectangle", "LEFT", "RIGHT", "UP", "DOWN", "FadeIn", "Write"]
