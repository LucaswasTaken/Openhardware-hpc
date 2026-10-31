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

    def __init__(self, voice: str = VOICE, rate: str = "+5%", **kwargs):
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


class CursoScene(VoiceoverScene):
    """Base scene: Portuguese voice + helpers for titles and boxes."""

    def setup(self):
        super().setup()
        self.set_speech_service(EdgeTTSService())

    def titulo(self, texto: str, sub: str = "") -> VGroup:
        t = Text(texto, font=FONT, weight="BOLD", font_size=46)
        g = VGroup(t)
        if sub:
            g.add(Text(sub, font=FONT, font_size=26, color=MUTED).next_to(t, DOWN, buff=0.3))
        return g

    def topo(self, texto: str) -> Text:
        return Text(texto, font=FONT, weight="BOLD", font_size=34).set_max_width(12.6).to_edge(UP, buff=0.4)

    def caixa(self, rotulo: str, cor, w=2.6, h=1.3, fs=28) -> VGroup:
        r = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=cor, fill_opacity=0.18, stroke_width=3)
        return VGroup(r, Text(rotulo, font=FONT, font_size=fs, color=WHITE).move_to(r))

    def texto(self, s: str, fs=26, cor=WHITE, **kw) -> Text:
        return Text(s, font=FONT, font_size=fs, color=cor, **kw)

    def limpa(self, *keep):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.6)


__all__ = ["CursoScene", "results", "CPU_C", "GPU_C", "MEM_C", "DATA_C", "OK_C", "BAD_C", "MUTED", "FONT",
           "Rectangle", "LEFT", "RIGHT", "UP", "DOWN", "FadeIn", "Write"]
