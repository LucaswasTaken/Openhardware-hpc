"""Vídeos da Aula 2 (GPU na prática, CuPy): 07 operários · 08 leve os dados uma vez · 09 CuPy em 3 passos ·
10 laje em incêndio.  (08b, memória unificada: v08b_dois_mundos.py)

Aplicados e com o mínimo de fórmulas; os números vêm de notebooks/results/aula2_*.json."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, FadeIn, FadeOut, Indicate, LaggedStart, Line, ManimColor,
                   Square, SurroundingRectangle, VGroup, color_gradient)

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe, results

R = results(2)
GPU = R.get("gpu", {})
GPU_NOME = GPU.get("nome", "GPU")


def vezes(x):
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×".replace(".", ",")


def falado(x):
    """Número para a narração: inteiro se ≥ 10, uma casa decimal com vírgula se menor."""
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}".replace(".", ",")


# ------------------------------------------------------------------ 07
class V07Operarios(CursoScene):
    def construct(self):
        t = self.titulo("GPU: milhares de operários", "Aula 2 · vídeo 7")
        with self.voiceover("Por que uma placa de vídeo consegue acelerar um cálculo de engenharia?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("8 engenheiros × 2 500 operários")
        cpu = VGroup(*[Square(0.9, color=CPU_C, fill_opacity=0.35) for _ in range(8)]).arrange_in_grid(2, 4, buff=0.15)
        gpu = VGroup(*[Square(0.11, color=GPU_C, fill_opacity=0.75) for _ in range(40 * 20)]).arrange_in_grid(20, 40, buff=0.035)
        lc = self.texto("CPU: poucos, experientes", 26, CPU_C).next_to(cpu, UP, buff=0.25)
        lg = self.texto("GPU: milhares, simples", 26, GPU_C).next_to(gpu, UP, buff=0.25)
        VGroup(VGroup(lc, cpu), VGroup(lg, gpu)).arrange(RIGHT, buff=1.0).next_to(topo, DOWN, buff=0.6)
        with self.voiceover(
            "Pense numa obra. A CPU é uma equipe de oito engenheiros experientes: cada um resolve qualquer problema, "
            "rápido, tomando decisões diferentes. <bookmark mark='g'/>A GPU é uma equipe de milhares de operários. Cada "
            "um é mais lento e só sabe uma tarefa simples, mas todos trabalham ao mesmo tempo, cada um no seu pedaço."):
            self.play(FadeIn(topo), FadeIn(lc), FadeIn(cpu))
            self.wait_until_bookmark("g")
            self.play(FadeIn(lg), FadeIn(gpu, lag_ratio=0.001), run_time=2.5)
        self.limpa()

        topo = self.topo("Quando a GPU ajuda")
        pontos = VGroup(*[Square(0.3, color=DATA_C, fill_opacity=0.7) for _ in range(60)]).arrange_in_grid(5, 12, buff=0.06)
        pontos.next_to(topo, DOWN, buff=0.6)
        conta = self.texto("a MESMA conta em cada ponto:  tensão de von Mises", 26, MEM_C).next_to(pontos, DOWN, buff=0.4)
        ok = self.texto("✓ milhões de pontos   ✓ mesma conta   ✓ dados ficam na GPU", 26, OK_C).next_to(conta, DOWN, buff=0.5)
        with self.voiceover(
            "A GPU ajuda quando o problema tem muitos dados recebendo a mesma conta: a tensão em um milhão de pontos de "
            "integração, a temperatura em cada nó de uma malha, as amostras de um Monte Carlo. "
            "<bookmark mark='c'/>Cada operário pega um ponto, e todos calculam juntos."):
            self.play(FadeIn(topo), FadeIn(pontos, lag_ratio=0.02))
            self.wait_until_bookmark("c")
            self.play(FadeIn(conta), pontos.animate.set_color(GPU_C), run_time=1.2)
            self.play(FadeIn(ok))
        self.limpa()

        topo = self.topo("Quando a GPU atrapalha")
        itens = VGroup(self.texto("✗ poucos dados: não há trabalho para tantos operários", 26, BAD_C),
                       self.texto("✗ cada item pede uma decisão diferente (muitos if)", 26, BAD_C),
                       self.texto("✗ os dados precisam viajar até a GPU para uma conta só", 26, BAD_C)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(topo, DOWN, buff=0.8)
        with self.voiceover(
            "E atrapalha quando há poucos dados, quando cada item pede uma decisão diferente, ou quando os dados precisam "
            "viajar até a placa para uma única conta rápida. Esse último ponto é o assunto do próximo vídeo."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in itens], lag_ratio=0.7), run_time=4)
        self.limpa()

        nuc = GPU.get("nucleos", 2560)
        topo = self.topo(f"A GPU da aula: {GPU_NOME}")
        info = VGroup(self.texto(f"{nuc:,} núcleos".replace(",", "."), 40, GPU_C),
                      self.texto(f"contra {GPU.get('cpu_nucleos', 2)} núcleos de CPU", 30, CPU_C),
                      self.texto(f"{GPU.get('memoria_gib', 15):.0f} GB de memória própria", 30, MEM_C)
                      ).arrange(DOWN, buff=0.4).next_to(topo, DOWN, buff=1.0)
        with self.voiceover(
            f"A GPU usada na aula tem {nuc} núcleos e memória própria. "
            "A pergunta para levar: o meu problema tem muitos dados recebendo a mesma conta?"):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(i) for i in info], lag_ratio=0.5), run_time=3)
        self.wait(1)


# ------------------------------------------------------------------ 08
class V08Viagem(CursoScene):
    def construct(self):
        t = self.titulo("Leve os dados uma vez", "Aula 2 · vídeo 8")
        with self.voiceover("Os operários são rápidos. Mas e o material?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Duas memórias e uma estrada")
        ram = self.caixa("RAM\n(computador)", CPU_C, w=3.0, h=1.6).shift(LEFT * 4.3)
        vram = self.caixa("VRAM\n(GPU)", GPU_C, w=3.0, h=1.6).shift(RIGHT * 4.3)
        estrada = Line(ram.get_right(), vram.get_left(), stroke_width=8, color=MUTED)
        rot = self.texto("cabo PCIe", 24, MUTED).next_to(estrada, UP, buff=0.2)
        carga = VGroup(*[Square(0.28, color=DATA_C, fill_opacity=0.9) for _ in range(6)]).arrange(RIGHT, buff=0.06)
        carga.move_to(ram).shift(DOWN * 1.3)
        with self.voiceover(
            "No Colab e na maioria dos computadores, a GPU é uma placa separada, com memória própria. Os dados nascem na "
            "memória do computador, a RAM. <bookmark mark='v'/>Para a GPU calcular, eles precisam viajar pelo cabo PCIe, "
            "como um caminhão levando material do depósito até a obra."):
            self.play(FadeIn(topo), FadeIn(ram), FadeIn(vram))
            self.wait_until_bookmark("v")
            self.play(Create(estrada), FadeIn(rot), FadeIn(carga))
            self.play(carga.animate.move_to(vram).shift(DOWN * 1.3), run_time=2.5)
        self.limpa()

        cc = R.get("copia_calculo", {"mb": 400, "ida_ms": 34.0, "leve_ms": 3.3, "pesada_ms": 14.0, "volta_ms": 36.0})
        topo = self.topo(f"Medido: {cc['mb']:.0f} MB, viagem × conta")
        bar = self.barras([("viagem RAM → GPU", cc["ida_ms"], BAD_C), ("conta leve", cc["leve_ms"], OK_C),
                           ("conta pesada", cc["pesada_ms"], CPU_C), ("viagem GPU → RAM", cc["volta_ms"], BAD_C)],
                          fmt=lambda v: f"{v:.1f} ms".replace(".", ",")).next_to(topo, DOWN, buff=0.8)
        fonte = self.fonte(f"medido na aula · {GPU_NOME}")
        with self.voiceover(
            f"Medimos na aula. Levar {cc['mb']:.0f} megabytes até a GPU levou {falado(cc['ida_ms'])} milissegundos. "
            f"<bookmark mark='l'/>Uma conta leve com esses dados, multiplicar e somar, só {falado(cc['leve_ms'])}. "
            f"<bookmark mark='p'/>Mesmo uma conta pesada, com seno, cosseno e exponencial, levou {falado(cc['pesada_ms'])}. "
            f"<bookmark mark='r'/>E ainda tem a volta."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar[:1], run_time=1.0)
            self.wait_until_bookmark("l"); self.mostra_barras(bar[1:2], run_time=1.0)
            self.wait_until_bookmark("p"); self.mostra_barras(bar[2:3], run_time=1.0)
            self.wait_until_bookmark("r"); self.mostra_barras(bar[3:], run_time=1.0)
        self.limpa()

        topo = self.topo("A regra de ouro")
        regra = VGroup(self.texto("1. leve os dados para a GPU UMA vez", 32, MEM_C),
                       self.texto("2. faça TODAS as contas lá", 32, GPU_C),
                       self.texto("3. traga de volta só o resultado", 32, OK_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        regra.next_to(topo, DOWN, buff=0.8)
        ex = self.texto("Simulação com 10 000 passos: paga a viagem uma vez → a GPU compensa.", 24, MUTED).to_edge(DOWN, buff=0.6)
        with self.voiceover(
            "Daí a regra de ouro: leve os dados uma vez, faça todas as contas lá, e traga de volta só o resultado. "
            "<bookmark mark='e'/>Uma simulação com milhares de passos paga a viagem uma vez só, e aí a GPU compensa. "
            "Uma conta única, com os dados na RAM, quase nunca."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in regra], lag_ratio=0.6), run_time=3)
            self.wait_until_bookmark("e")
            self.play(FadeIn(ex))
        self.wait(1)


# ------------------------------------------------------------------ 09
class V09CuPy(CursoScene):
    def construct(self):
        t = self.titulo("CuPy em 3 passos", "Aula 2 · vídeo 9")
        with self.voiceover("Se você sabe usar o NumPy, já sabe usar a GPU. Quase."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Troque np por cp")
        antes = self.codigo('''
import numpy as np
vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
''', largura=10.5)
        depois = self.codigo('''
import cupy as cp
vm = cp.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
''', largura=10.5)
        VGroup(antes, depois).arrange(DOWN, buff=0.5).next_to(topo, DOWN, buff=0.5)
        rot_a = self.texto("CPU", 24, CPU_C).next_to(antes, LEFT, buff=0.2)
        rot_d = self.texto("GPU", 24, GPU_C).next_to(depois, LEFT, buff=0.2)
        with self.voiceover(
            "O CuPy copia o jeito do NumPy. A tensão de von Mises em milhões de pontos, com NumPy, roda na CPU. "
            "<bookmark mark='d'/>Troque n p por c p, e a mesma conta roda na GPU."):
            self.play(FadeIn(topo), FadeIn(antes), FadeIn(rot_a))
            self.wait_until_bookmark("d")
            self.play(FadeIn(depois), FadeIn(rot_d))
        self.limpa()

        topo = self.topo("Os 3 passos")
        passos = VGroup(*[self.caixa(s, c, w=11, h=0.95, fs=26) for s, c in [
            ("1. LEVE:  x = cp.asarray(x_ram)", MEM_C),
            ("2. CALCULE:  troque np por cp  →  cp.sqrt, cp.sum, A @ B", GPU_C),
            ("3. TRAGA só o resultado:  cp.asnumpy(y)  ou  float(total)", OK_C)]]).arrange(DOWN, buff=0.35)
        passos.next_to(topo, DOWN, buff=0.6)
        with self.voiceover(
            "São três passos. <bookmark mark='a'/>Leve os dados para a GPU, com c p ponto asarray. "
            "<bookmark mark='b'/>Calcule, trocando n p por c p. <bookmark mark='c'/>E traga de volta só o resultado, "
            "com c p ponto asnumpy, ou convertendo um número com float."):
            self.play(FadeIn(topo))
            for k, p in zip("abc", passos):
                self.wait_until_bookmark(k)
                self.play(FadeIn(p, shift=RIGHT * 0.2))
        self.limpa()

        vm = R.get("von_mises", {"n": 20_000_000, "cpu_ms": 380.0, "gpu_ms": 9.0})
        topo = self.topo(f"Von Mises em {vm['n'] / 1e6:.0f} milhões de pontos")
        bar = self.barras([("NumPy (CPU)", vm["cpu_ms"], CPU_C), ("CuPy (GPU)", vm["gpu_ms"], GPU_C)],
                          fmt=lambda v: f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms".replace(".", ",")).next_to(topo, DOWN, buff=1.0)
        fonte = self.fonte(f"dados já na GPU · medido na aula · {GPU_NOME}")
        with self.voiceover(
            f"Na aula, com os dados já na GPU, a GPU fez a conta {falado(vm['cpu_ms'] / vm['gpu_ms'])} vezes mais rápido "
            "que o NumPy."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar)
        self.limpa()

        topo = self.topo("O erro nº 1: esquecer de sincronizar")
        a = self.texto("a CPU PEDE a conta e segue em frente…", 28, CPU_C)
        b = self.texto("…a GPU ainda está calculando", 28, GPU_C)
        c = self.texto("cp.cuda.Device().synchronize()  →  espere antes de parar o relógio", 26, MEM_C)
        d = self.texto("Resultado de GPU bom demais? Pergunte se sincronizou.", 30, BAD_C)
        VGroup(a, b, c, d).arrange(DOWN, buff=0.5).next_to(topo, DOWN, buff=0.7)
        with self.voiceover(
            "Um cuidado ao medir. A CPU pede a conta à GPU e segue em frente, sem esperar. <bookmark mark='b'/>Se você "
            "parar o relógio nessa hora, mediu só o pedido. <bookmark mark='c'/>Por isso, sincronize antes de parar o "
            "relógio. <bookmark mark='d'/>Se um tempo de GPU parece bom demais, a primeira pergunta é: sincronizou? "
            "A inteligência artificial erra muito isso."):
            self.play(FadeIn(topo), FadeIn(a))
            self.wait_until_bookmark("b"); self.play(FadeIn(b))
            self.wait_until_bookmark("c"); self.play(FadeIn(c))
            self.wait_until_bookmark("d"); self.play(FadeIn(d))
        self.wait(1)


# ------------------------------------------------------------------ 10
def _calor(n=40, passos=(0, 60, 250, 900), r=0.24):
    T = np.full((n, n), 20.0)
    T[-1, :] = 600.0
    quadros, feito = [], 0
    for alvo in passos:
        for _ in range(alvo - feito):
            T[1:-1, 1:-1] = T[1:-1, 1:-1] + r * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - 4 * T[1:-1, 1:-1])
        feito = alvo
        quadros.append(T.copy())
    return quadros


def _mapa(T, lado=4.6):
    n = T.shape[0]
    cel = lado / n
    cores = color_gradient([ManimColor("#1b1f3a"), ManimColor("#7a1f5c"), ManimColor("#e0522a"), ManimColor("#fbe36b")], 64)
    g = VGroup()
    for i in range(n):
        for j in range(n):
            k = int(np.clip((T[i, j] - 20) / 580, 0, 1) * 63)
            g.add(Square(cel, stroke_width=0, fill_color=cores[k], fill_opacity=1).move_to([j * cel, -i * cel, 0]))
    return g.center()


class V10Laje(CursoScene):
    def construct(self):
        t = self.titulo("Laje em incêndio: CPU × GPU", "Aula 2 · vídeo 10")
        with self.voiceover("Uma laje de concreto com a face de baixo num incêndio. Até onde o calor sobe, e quem calcula mais rápido?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Cada ponto vira a média dos vizinhos, a cada passo")
        quadros = _calor()
        mapa = _mapa(quadros[0]).shift(LEFT * 3.2 + DOWN * 0.4)
        lab_f = self.texto("incêndio 600 °C", 22, BAD_C).next_to(mapa, DOWN, buff=0.15)
        est = VGroup(*[Square(0.7, color=c, fill_opacity=0.25) for c in (MUTED, MUTED, MEM_C, MUTED, MUTED)])
        est[0].move_to([0, 0.75, 0]); est[1].move_to([-0.75, 0, 0]); est[2].move_to([0, 0, 0])
        est[3].move_to([0.75, 0, 0]); est[4].move_to([0, -0.75, 0])
        est.shift(RIGHT * 3.3 + UP * 0.8)
        eq = self.texto("novo = atual + 0,24 × (vizinhos − 4 × atual)", 24, MEM_C).next_to(est, DOWN, buff=0.4)
        with self.voiceover(
            "Dividimos o corte da laje numa malha. A cada passo de tempo, cada ponto fica um pouco mais parecido com a "
            "média dos quatro vizinhos. <bookmark mark='p'/>Repetindo isso centenas de vezes, o calor do incêndio sobe."):
            self.play(FadeIn(topo), FadeIn(mapa), FadeIn(lab_f), FadeIn(est), FadeIn(eq))
            self.wait_until_bookmark("p")
            for q in quadros[1:]:
                self.play(mapa.animate.become(_mapa(q).move_to(mapa)), run_time=1.3)
        gpu_t = self.texto("milhões de pontos, a mesma conta,\nos dados parados na GPU", 26, GPU_C).next_to(eq, DOWN, buff=0.5)
        with self.voiceover("Milhões de pontos, a mesma conta, e os dados ficam parados na GPU durante todos os passos. "
                            "É o cenário ideal."):
            self.play(FadeIn(gpu_t))
        self.limpa()

        topo = self.topo("Uma função só, para CPU e GPU")
        cod = self.codigo('''
def simula(T, passos):
    xp = cp.get_array_module(T)      # np ou cp, conforme onde T está
    ...                              # a mesma conta, para os dois

simula(laje_inicial(n), 300)         # roda na CPU
simula(laje_inicial(n, cp), 300)     # roda na GPU
''', largura=11).next_to(topo, DOWN, buff=0.6)
        with self.voiceover(
            "No notebook, a simulação é escrita uma vez só. Ela descobre se a laje está na memória do computador ou na "
            "da GPU, e usa a biblioteca certa. Quem decide onde roda é o lugar onde os dados estão."):
            self.play(FadeIn(topo), FadeIn(cod))
        self.limpa()

        tempos = R.get("calor", {}).get("tempos", {})
        nb = tempos.get("Numba (CPU, todos os núcleos)", {})
        cu = tempos.get("CuPy (GPU)", {})
        if nb and cu:
            pequeno, grande = min(cu, key=int), max(cu, key=int)
            topo = self.topo("300 passos: depende do tamanho da malha")
            def bloco(n):
                b = self.barras([("Numba (CPU)", nb[n], CPU_C), ("CuPy (GPU)", cu[n], GPU_C)],
                                fmt=lambda v: f"{v:.3f} s".replace(".", ","), largura=3.2)
                return VGroup(self.texto(f"malha {n} × {n}", 26, MEM_C), b).arrange(DOWN, buff=0.4)
            b1, b2 = bloco(pequeno), bloco(grande)
            VGroup(b1, b2).arrange(RIGHT, buff=1.0).next_to(topo, DOWN, buff=0.8)
            cabe(VGroup(b1, b2))
            fonte = self.fonte(f"medido · {GPU_NOME} · CPU com {GPU.get('cpu_nucleos', 2)} núcleos")
            ganhou = cu[grande] < nb[grande]
            final = ("a GPU passa à frente. Quanto maior o problema, mais ela aproveita." if ganhou else
                     f"a diferença diminui, mas aqui a CPU, com {GPU.get('cpu_nucleos', 12)} núcleos e Numba, ainda ganha "
                     "de uma GPU de notebook. No Colab, com só dois núcleos e uma GPU maior, a situação se inverte.")
            with self.voiceover(
                f"Os tempos medidos. Na malha pequena, de {pequeno} por {pequeno}, há pouco trabalho para tantos "
                f"operários, e a GPU não ganha. <bookmark mark='g'/>Na malha grande, de {grande} por {grande}, {final}"):
                self.play(FadeIn(topo), FadeIn(fonte), FadeIn(b1[0]))
                self.mostra_barras(b1[1])
                self.wait_until_bookmark("g")
                self.play(FadeIn(b2[0]))
                self.mostra_barras(b2[1])
            self.limpa()

        fecho = VGroup(self.texto("Problema grande + dados parados na GPU → a GPU ganha.", 30, OK_C),
                       self.texto("Problema pequeno → a CPU (Numba) basta.", 30, CPU_C),
                       self.texto("Meça no SEU hardware antes de concluir.", 30, MEM_C)).arrange(DOWN, buff=0.5)
        with self.voiceover("Resumindo: problema grande, com os dados parados na GPU, a GPU ganha. Problema pequeno, a CPU "
                            "com Numba basta. E meça no seu hardware antes de concluir."):
            self.play(LaggedStart(*[FadeIn(f) for f in fecho], lag_ratio=0.7), run_time=4)
        self.wait(1)
