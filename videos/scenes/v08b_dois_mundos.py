"""Vídeo 08b — CPU e GPU: dois mundos ou um só? (memória unificada)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Create, DoubleArrow, FadeIn, FadeOut, GrowFromEdge,
                   Indicate, LaggedStart, Line, ReplacementTransform, Square, Transform, VGroup, Write)

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, results

R = results(2)
GPU_NOME = R.get("gpu", {}).get("nome", "GPU")
COPIA = R.get("memoria_unificada", {}).get("copia_ms", {})
GER = R.get("memoria_unificada", {}).get("gerenciada_ms", {})
H2D = COPIA.get("CPU→GPU (cópia)", 27.6)
KER = COPIA.get("kernel", 3.7)
G1 = GER.get("1º kernel (páginas migram)", 163.1)
G2 = GER.get("2º kernel (já na GPU)", 4.7)


def ms(x):
    return f"{x:.0f} ms" if x >= 10 else f"{x:.1f} ms"


class DoisMundos(CursoScene):
    def construct(self):
        # ---- abertura
        t = self.titulo("CPU e GPU: dois mundos ou um só?", "memória unificada e o que ela muda")
        with self.voiceover("Por muito tempo, CPU e GPU foram dois mundos separados. Será que isso ainda vale?"):
            self.play(Write(t[0]), FadeIn(t[1], shift=UP * 0.2))
        self.play(FadeOut(t))

        # ---- modelo 1: GPU discreta + PCIe
        topo = self.topo("1. GPU discreta: duas memórias e um cano estreito")
        cpu = self.caixa("CPU", CPU_C).shift(LEFT * 4.2 + UP * 0.9)
        ram = self.caixa("RAM", MEM_C, h=1.0).next_to(cpu, DOWN, buff=0.35)
        gpu = self.caixa("GPU", GPU_C).shift(RIGHT * 4.2 + UP * 0.9)
        vram = self.caixa("VRAM", MEM_C, h=1.0).next_to(gpu, DOWN, buff=0.35)
        cano = Line(ram.get_right(), vram.get_left(), stroke_width=6, color=MUTED)
        rot_cano = self.texto("PCIe  ~16–32 GB/s", 24, MUTED).next_to(cano, UP, buff=0.15)
        with self.voiceover(
            "No modelo clássico, o da T4 do Colab e da maioria das placas de vídeo, a CPU tem a sua memória, "
            "a RAM, e a GPU tem a dela, a VRAM. <bookmark mark='cano'/>Entre as duas há um barramento, o PCIe, "
            "bem mais estreito do que as próprias memórias."):
            self.play(FadeIn(topo), FadeIn(cpu), FadeIn(ram), FadeIn(gpu), FadeIn(vram))
            self.wait_until_bookmark("cano")
            self.play(Create(cano), FadeIn(rot_cano))

        blocos = VGroup(*[Square(0.28, color=DATA_C, fill_opacity=0.9) for _ in range(6)]).arrange(RIGHT, buff=0.08)
        blocos.move_to(ram).shift(DOWN * 0.95)
        with self.voiceover(
            "Para calcular na GPU, os dados precisam viajar: <bookmark mark='ida'/>copiamos da RAM para a VRAM, "
            "calculamos, <bookmark mark='volta'/>e copiamos o resultado de volta."):
            self.play(FadeIn(blocos))
            self.wait_until_bookmark("ida")
            self.play(blocos.animate.move_to(vram).shift(DOWN * 0.95), run_time=2.2)
            self.play(Indicate(gpu, color=GPU_C), run_time=0.6)
            self.wait_until_bookmark("volta")
            self.play(blocos.animate.move_to(ram).shift(DOWN * 0.95), run_time=2.2)

        # números medidos
        barra_max = 7.5
        escala = barra_max / max(H2D, KER)
        b1 = VGroup(self.texto("cópia CPU→GPU", 24).set_width(2.6),
                    VGroup(Line(ORIGIN, RIGHT * H2D * escala, stroke_width=26, color=BAD_C)),
                    self.texto(ms(H2D), 24)).arrange(RIGHT, buff=0.3)
        b2 = VGroup(self.texto("cálculo na GPU", 24).set_width(2.6),
                    VGroup(Line(ORIGIN, RIGHT * max(KER * escala, 0.08), stroke_width=26, color=OK_C)),
                    self.texto(ms(KER), 24)).arrange(RIGHT, buff=0.3)
        barras = VGroup(b1, b2).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(DOWN, buff=0.5)
        fonte = self.texto(f"medido na aula · {GPU_NOME} · 128 MiB", 18, MUTED).next_to(barras, UP, buff=0.2)
        with self.voiceover(
            f"Medimos isso na aula. Copiar cento e vinte e oito megabytes levou {H2D:.0f} milissegundos; "
            f"a conta em si, só {KER:.0f}. A viagem custa muito mais do que o cálculo."):
            self.play(blocos.animate.set_opacity(0.0), FadeIn(fonte))
            self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in (b1, b2)], lag_ratio=0.4), run_time=1.6)
        self.wait(0.5)
        self.limpa()

        # ---- memória gerenciada: a ilusão
        topo = self.topo("Memória gerenciada (CUDA): um ponteiro só… em cima do mesmo cano")
        cpu = self.caixa("CPU", CPU_C).shift(LEFT * 4.2 + UP * 0.9)
        gpu = self.caixa("GPU", GPU_C).shift(RIGHT * 4.2 + UP * 0.9)
        unica = self.caixa("um array: x", MEM_C, w=6.0, h=0.9).shift(DOWN * 0.6)
        seta1 = Arrow(cpu.get_bottom(), unica.get_left() + RIGHT * 0.8, buff=0.1, color=CPU_C)
        seta2 = Arrow(gpu.get_bottom(), unica.get_right() + LEFT * 0.8, buff=0.1, color=GPU_C)
        with self.voiceover(
            "O CUDA oferece a memória gerenciada: um único array, que a CPU e a GPU enxergam com o mesmo endereço. "
            "O código fica mais simples. <bookmark mark='mas'/>Mas, numa GPU discreta, isso é uma ilusão de software."):
            self.play(FadeIn(topo), FadeIn(cpu), FadeIn(gpu), FadeIn(unica), Create(seta1), Create(seta2))
            self.wait_until_bookmark("mas")
            self.play(Indicate(unica, color=BAD_C))
        cano = Line(LEFT * 3, RIGHT * 3, stroke_width=6, color=MUTED).shift(DOWN * 2.0)
        pag = VGroup(*[Square(0.25, color=DATA_C, fill_opacity=0.9) for _ in range(8)]).arrange(RIGHT, buff=0.06)
        pag.next_to(cano, LEFT, buff=0.1)
        rot = self.texto("páginas migram pelo PCIe", 22, MUTED).next_to(cano, DOWN, buff=0.15)
        with self.voiceover(
            f"Por baixo, as páginas de memória migram pelo mesmo PCIe quando a GPU as toca. "
            f"<bookmark mark='n1'/>Na nossa medida, o primeiro kernel levou {G1:.0f} milissegundos, porque os dados "
            f"tiveram que viajar. <bookmark mark='n2'/>O segundo, com tudo já na GPU, {G2:.0f}. "
            f"O custo continua lá, só que escondido."):
            self.play(Create(cano), FadeIn(rot), FadeIn(pag))
            self.play(pag.animate.next_to(cano, RIGHT, buff=0.1), run_time=2.0)
            self.wait_until_bookmark("n1")
            n1 = self.texto(f"1º kernel: {ms(G1)}", 30, BAD_C).to_edge(DOWN, buff=0.15).shift(LEFT * 3)
            self.play(FadeIn(n1))
            self.wait_until_bookmark("n2")
            n2 = self.texto(f"2º kernel: {ms(G2)}", 30, OK_C).to_edge(DOWN, buff=0.15).shift(RIGHT * 3)
            self.play(FadeIn(n2))
        self.wait(0.5)
        self.limpa()

        # ---- os três modelos lado a lado
        topo = self.topo("O hardware está mudando")
        def coluna(nome, exemplos, banda, desenho):
            cab = self.texto(nome, 24).set_max_width(4.0)
            ex = self.texto(exemplos, 18, MUTED).set_max_width(4.0)
            bd = self.texto(banda, 20, MEM_C)
            return VGroup(cab, desenho, bd, ex).arrange(DOWN, buff=0.3)

        def par(link_w, link_c, sep=True):
            c = self.caixa("CPU", CPU_C, w=1.2, h=0.7, fs=22)
            g = self.caixa("GPU", GPU_C, w=1.2, h=0.7, fs=22)
            if sep:
                m1 = self.caixa("RAM", MEM_C, w=1.2, h=0.55, fs=18)
                m2 = self.caixa("VRAM", MEM_C, w=1.2, h=0.55, fs=18)
                left, right = VGroup(c, m1).arrange(DOWN, buff=0.15), VGroup(g, m2).arrange(DOWN, buff=0.15)
                d = VGroup(left, right).arrange(RIGHT, buff=0.9)
                d.add(Line(m1.get_right(), m2.get_left(), stroke_width=link_w, color=link_c))
                return d
            cg = VGroup(c, g).arrange(RIGHT, buff=0.25)
            m = self.caixa("memória única", MEM_C, w=2.65, h=0.55, fs=18).next_to(cg, DOWN, buff=0.15)
            return VGroup(cg, m)

        col1 = coluna("GPU discreta + PCIe", "T4 (Colab), GeForce/RTX,\nmaioria dos clusters", "~32 GB/s entre elas",
                      par(4, MUTED))
        col2 = coluna("CPU+GPU coerentes", "NVIDIA Grace Hopper,\nGrace Blackwell, DGX Spark", "NVLink-C2C ~900 GB/s",
                      par(16, MEM_C))
        col3 = coluna("Memória física única", "Apple M1–M5, AMD MI300A,\nRyzen AI Max, Lunar Lake", "sem cópia",
                      par(0, MEM_C, sep=False))
        cols = VGroup(col1, col2, col3).arrange(RIGHT, buff=0.7).set_width(12.4).next_to(topo, DOWN, buff=0.6)
        with self.voiceover(
            "E o hardware está mudando. <bookmark mark='c1'/>Ao lado da GPU discreta, "
            "<bookmark mark='c2'/>surgiram sistemas em que CPU e GPU são ligadas por um canal quase tão largo quanto a "
            "memória, com endereços coerentes, como o Grace Hopper da NVIDIA. "
            "<bookmark mark='c3'/>E chips com uma única memória física para os dois, como os processadores Apple M "
            "e o AMD MI300A, do supercomputador El Capitan. Aqui, simplesmente não existe cópia."):
            self.play(FadeIn(topo))
            self.wait_until_bookmark("c1"); self.play(FadeIn(col1, shift=UP * 0.2))
            self.wait_until_bookmark("c2"); self.play(FadeIn(col2, shift=UP * 0.2))
            self.wait_until_bookmark("c3"); self.play(FadeIn(col3, shift=UP * 0.2))
        self.wait(0.5)
        self.limpa()

        # ---- o que não muda
        topo = self.topo("O que a memória unificada NÃO muda")
        cpu_n = VGroup(*[Square(0.75, color=CPU_C, fill_opacity=0.35) for _ in range(8)]).arrange_in_grid(2, 4, buff=0.12)
        gpu_n = VGroup(*[Square(0.12, color=GPU_C, fill_opacity=0.7) for _ in range(32 * 12)]).arrange_in_grid(12, 32, buff=0.04)
        lab1 = self.texto("CPU: poucos núcleos espertos", 22, CPU_C).next_to(cpu_n, UP, buff=0.25)
        lab2 = self.texto("GPU: milhares de núcleos simples, em grupos de 32", 22, GPU_C).next_to(gpu_n, UP, buff=0.25)
        bloco = VGroup(VGroup(lab1, cpu_n), VGroup(lab2, gpu_n)).arrange(RIGHT, buff=0.9).next_to(topo, DOWN, buff=0.5)
        itens = VGroup(self.texto("• ainda é preciso ter MUITO paralelismo de dados", 24),
                       self.texto("• largura de banda e localidade continuam mandando", 24),
                       self.texto("• um laço sequencial continua sequencial", 24)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        itens.to_edge(DOWN, buff=0.5)
        with self.voiceover(
            "Mas atenção ao que não muda. A GPU continua sendo milhares de núcleos simples, que executam a mesma "
            "instrução em grupos de trinta e dois. <bookmark mark='i'/>Ela só vale a pena com muito paralelismo de "
            "dados. A largura de banda e a localidade continuam mandando. E um laço sequencial continua sequencial, "
            "com ou sem memória unificada."):
            self.play(FadeIn(topo), FadeIn(bloco[0]), LaggedStart(FadeIn(lab2), FadeIn(gpu_n, lag_ratio=0.002)))
            self.wait_until_bookmark("i")
            self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in itens], lag_ratio=0.6), run_time=3)
        self.wait(0.5)
        self.limpa()

        # ---- fecho
        p1 = self.texto("A pergunta mudou:", 32, MUTED)
        p2 = self.texto("de  “quanto custa copiar?”", 38, BAD_C)
        p3 = self.texto("para  “onde o dado mora, e quem o processa melhor?”", 38, OK_C)
        fecho = VGroup(p1, p2, p3).arrange(DOWN, buff=0.45)
        with self.voiceover(
            "Então a pergunta mudou. Em vez de quanto custa copiar, <bookmark mark='q'/>pergunte: onde o dado mora, "
            "e quem processa melhor esse dado? Na aula, vamos medir isso na prática."):
            self.play(FadeIn(p1), FadeIn(p2))
            self.wait_until_bookmark("q")
            self.play(FadeIn(p3, shift=UP * 0.2))
        self.wait(1.5)
