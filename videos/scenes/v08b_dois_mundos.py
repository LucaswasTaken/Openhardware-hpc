"""Vídeo 08b — CPU e GPU: dois mundos ou um só? (memória unificada, versão curta e aplicada)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import DOWN, LEFT, RIGHT, UP, Create, FadeIn, FadeOut, LaggedStart, Line, Square, VGroup, Write

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe, results

R = results(2)
CC = R.get("copia_calculo", {"ida_ms": 34.0, "leve_ms": 3.3})


class DoisMundos(CursoScene):
    def construct(self):
        t = self.titulo("CPU e GPU: dois mundos ou um só?", "memória unificada, sem mistério")
        with self.voiceover("A viagem dos dados até a GPU custa caro. E se não houvesse viagem?"):
            self.play(Write(t[0]), FadeIn(t[1], shift=UP * 0.2))
        self.play(FadeOut(t))

        # ---- mundo 1: placa separada
        topo = self.topo("Hoje, no Colab: dois mundos e uma estrada")
        cpu = self.caixa("CPU", CPU_C).shift(LEFT * 4.2 + UP * 0.9)
        ram = self.caixa("RAM", MEM_C, h=1.0).next_to(cpu, DOWN, buff=0.35)
        gpu = self.caixa("GPU", GPU_C).shift(RIGHT * 4.2 + UP * 0.9)
        vram = self.caixa("VRAM", MEM_C, h=1.0).next_to(gpu, DOWN, buff=0.35)
        cano = Line(ram.get_right(), vram.get_left(), stroke_width=6, color=MUTED)
        rot = self.texto("cabo PCIe: a viagem", 24, MUTED).next_to(cano, UP, buff=0.15)
        carga = VGroup(*[Square(0.28, color=DATA_C, fill_opacity=0.9) for _ in range(6)]).arrange(RIGHT, buff=0.08)
        carga.move_to(ram).shift(DOWN * 0.95)
        nota = self.texto(f"na aula: a viagem custou {CC['ida_ms'] / CC['leve_ms']:.0f}× a conta", 26, BAD_C).to_edge(DOWN, buff=0.5)
        with self.voiceover(
            "No Colab, e na maioria dos computadores com placa de vídeo, CPU e GPU vivem em dois mundos: cada uma com a "
            "sua memória, ligadas por um cabo. <bookmark mark='v'/>Os dados viajam de um lado para o outro, "
            f"<bookmark mark='n'/>e na aula essa viagem custou cerca de {CC['ida_ms'] / CC['leve_ms']:.0f} vezes a própria conta."):
            self.play(FadeIn(topo), FadeIn(cpu), FadeIn(ram), FadeIn(gpu), FadeIn(vram), Create(cano), FadeIn(rot))
            self.wait_until_bookmark("v")
            self.play(FadeIn(carga)); self.play(carga.animate.move_to(vram).shift(DOWN * 0.95), run_time=2.0)
            self.wait_until_bookmark("n")
            self.play(FadeIn(nota))
        self.limpa()

        # ---- mundo 2: memória unificada
        topo = self.topo("Memória unificada: o depósito dentro da obra")
        cpu = self.caixa("CPU", CPU_C).shift(LEFT * 2.0 + UP * 0.9)
        gpu = self.caixa("GPU", GPU_C).shift(RIGHT * 2.0 + UP * 0.9)
        mem = self.caixa("uma memória só", MEM_C, w=6.6, h=1.0).shift(DOWN * 0.8)
        ex = VGroup(self.texto("Apple M1–M5 (MacBooks)", 24),
                    self.texto("NVIDIA Grace Hopper · DGX Spark", 24),
                    self.texto("AMD MI300A (supercomputador El Capitan)", 24)).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.4)
        with self.voiceover(
            "Em algumas máquinas novas, CPU e GPU usam a mesma memória. É como ter o depósito dentro da obra: não há "
            "viagem. <bookmark mark='e'/>Isso já existe nos MacBooks com chip M, nos sistemas Grace Hopper da NVIDIA e no "
            "processador AMD MI300A, do supercomputador El Capitan."):
            self.play(FadeIn(topo), FadeIn(cpu), FadeIn(gpu), FadeIn(mem))
            self.wait_until_bookmark("e")
            self.play(LaggedStart(*[FadeIn(e) for e in ex], lag_ratio=0.5), run_time=2.5)
        self.limpa()

        # ---- o que muda e o que não muda
        topo = self.topo("O que muda, e o que NÃO muda")
        muda = VGroup(self.texto("MUDA", 30, OK_C),
                      self.texto("acaba a viagem", 26), self.texto("código mais simples", 26)).arrange(DOWN, buff=0.3)
        nao = VGroup(self.texto("NÃO MUDA", 30, BAD_C),
                     self.texto("precisa de MUITOS dados", 26), self.texto("com a MESMA conta", 26),
                     self.texto("memória rápida continua mandando", 26)).arrange(DOWN, buff=0.3)
        VGroup(muda, nao).arrange(RIGHT, buff=2.0, aligned_edge=UP).next_to(topo, DOWN, buff=0.8)
        with self.voiceover(
            "O que muda: acaba a viagem e o código fica mais simples. <bookmark mark='n'/>O que não muda: a GPU continua "
            "sendo milhares de operários fazendo a mesma conta. Ela continua precisando de muitos dados, e memória rápida "
            "continua mandando."):
            self.play(FadeIn(topo), FadeIn(muda))
            self.wait_until_bookmark("n")
            self.play(FadeIn(nao))
        self.limpa()

        p1 = self.texto("No Colab (placa separada):", 30, MUTED)
        p2 = self.texto("leve os dados uma vez.", 40, MEM_C)
        p3 = self.texto("Em qualquer máquina: pense em ONDE o dado mora.", 32, OK_C)
        fecho = VGroup(p1, p2, p3).arrange(DOWN, buff=0.45)
        with self.voiceover("Na aula, com placa separada, vale a regra de ouro: leve os dados uma vez. E em qualquer "
                            "máquina, pense em onde o dado mora."):
            self.play(FadeIn(p1), FadeIn(p2)); self.play(FadeIn(p3, shift=UP * 0.2))
        self.wait(1.5)
