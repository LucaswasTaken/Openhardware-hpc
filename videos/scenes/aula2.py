"""Vídeos da Aula 2 (GPU): 07 CPU×GPU · 08 transferências · 09 kernel · 10 equação do calor.  (08b: v08b_dois_mundos.py)"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, FadeIn, FadeOut, Indicate, LaggedStart, Line, Rectangle,
                   Square, SurroundingRectangle, VGroup, color_gradient, interpolate_color, ManimColor)

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe, results

R = results(2)
GPU = R.get("gpu", {})
GPU_NOME = GPU.get("nome", "GPU")


def vezes(x):
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


# ------------------------------------------------------------------ 07
class V07CpuGpu(CursoScene):
    def construct(self):
        t = self.titulo("CPU × GPU: milhares de núcleos simples", "Aula 2 · vídeo 7")
        with self.voiceover("A placa de vídeo nasceu para desenhar milhões de pixels. Hoje ela também faz engenharia."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Poucos núcleos espertos × milhares de núcleos simples")
        cpu = VGroup(*[Square(1.0, color=CPU_C, fill_opacity=0.35) for _ in range(8)]).arrange_in_grid(2, 4, buff=0.15)
        gpu = VGroup(*[Square(0.11, color=GPU_C, fill_opacity=0.75) for _ in range(40 * 20)]).arrange_in_grid(20, 40, buff=0.035)
        lc = self.texto("CPU", 30, CPU_C).next_to(cpu, UP, buff=0.25)
        lg = self.texto("GPU", 30, GPU_C).next_to(gpu, UP, buff=0.25)
        VGroup(VGroup(lc, cpu), VGroup(lg, gpu)).arrange(RIGHT, buff=1.0).next_to(topo, DOWN, buff=0.5)
        d1 = self.texto("otimizada para LATÊNCIA: terminar uma tarefa rápido", 22, CPU_C)
        d2 = self.texto("otimizada para VAZÃO: terminar muitas tarefas iguais", 22, GPU_C)
        VGroup(d1, d2).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voiceover(
            "A CPU tem poucos núcleos, mas cada um é esperto: prevê desvios, tem cache grande e termina uma tarefa "
            "complicada bem rápido. <bookmark mark='g'/>A GPU tem milhares de núcleos simples. Sozinho, cada um é "
            "fraco; juntos, terminam muitas tarefas iguais ao mesmo tempo. Pense numa equipe de oito engenheiros "
            "sêniores, contra uma obra com milhares de operários."):
            self.play(FadeIn(topo), FadeIn(lc), FadeIn(cpu), FadeIn(d1))
            self.wait_until_bookmark("g")
            self.play(FadeIn(lg), FadeIn(gpu, lag_ratio=0.001), FadeIn(d2), run_time=2.5)
        self.limpa()

        topo = self.topo("Warp: 32 threads, a mesma instrução")
        warp = VGroup(*[Square(0.32, color=GPU_C, fill_opacity=0.8) for _ in range(32)]).arrange(RIGHT, buff=0.06)
        warp.next_to(topo, DOWN, buff=1.0)
        instr = self.texto("y[i] = a · x[i] + b", 30, MEM_C).next_to(warp, DOWN, buff=0.5)
        if_t = self.texto("if x[i] > 0:  caminho A   else:  caminho B", 28, BAD_C).next_to(instr, DOWN, buff=0.7)
        with self.voiceover(
            "Os núcleos da GPU trabalham em grupos de trinta e duas threads, chamados warps, e todas executam a mesma "
            "instrução, cada uma com o seu dado. <bookmark mark='if'/>Se metade do grupo vai por um caminho de um if e "
            "metade por outro, a GPU executa os dois caminhos, um depois do outro, e metade das threads fica parada. "
            "Por isso a GPU gosta de contas uniformes."):
            self.play(FadeIn(topo), FadeIn(warp))
            self.play(FadeIn(instr), warp.animate.set_color(MEM_C), run_time=1.2)
            self.wait_until_bookmark("if")
            self.play(FadeIn(if_t))
            self.play(*[warp[i].animate.set_opacity(0.15 if i % 2 else 1) for i in range(32)])
            self.play(*[warp[i].animate.set_opacity(1 if i % 2 else 0.15) for i in range(32)])
        self.limpa()

        topo = self.topo(f"A GPU da aula: {GPU_NOME}")
        info = VGroup(
            self.texto(f"{GPU.get('sms', 40)} multiprocessadores (SMs)", 30),
            self.texto(f"{GPU.get('nucleos', 2560):,} núcleos CUDA".replace(",", "."), 30, GPU_C),
            self.texto(f"{GPU.get('memoria_gib', 15):.0f} GiB de memória · ~{GPU.get('banda_gbs', 320):.0f} GB/s", 30, MEM_C),
        ).arrange(DOWN, buff=0.4).next_to(topo, DOWN, buff=1.0)
        with self.voiceover(
            f"Na GPU usada na aula, são {GPU.get('nucleos', 2560)} núcleos, organizados em "
            f"{GPU.get('sms', 40)} multiprocessadores. A regra para ela compensar: muito paralelismo de dados, e "
            "contas parecidas para todos."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(i) for i in info], lag_ratio=0.5), run_time=3)
        self.wait(1)


# ------------------------------------------------------------------ 08
class V08Transferencias(CursoScene):
    def construct(self):
        t = self.titulo("O gargalo das transferências", "Aula 2 · vídeo 8")
        with self.voiceover("Por que somar dois vetores na GPU pode ser mais lento do que na CPU?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        sv = R.get("soma_vetores", {"cpu_ms": 29.4, "gpu_ms": 3.4, "gpu_com_copias_ms": 58.4})
        topo = self.topo("Soma de 20 milhões de números")
        bar = self.barras([("CPU (NumPy)", sv["cpu_ms"], CPU_C), ("GPU, só a conta", sv["gpu_ms"], OK_C),
                           ("GPU, com as cópias", sv["gpu_com_copias_ms"], BAD_C)],
                          fmt=lambda v: f"{v:.1f} ms").next_to(topo, DOWN, buff=0.9)
        fonte = self.fonte(f"medido na aula · {GPU_NOME}")
        with self.voiceover(
            f"Somar vinte milhões de números levou {sv['cpu_ms']:.0f} milissegundos na CPU. Na GPU, só a conta, "
            f"{sv['gpu_ms']:.1f}. <bookmark mark='c'/>Mas, incluindo copiar os vetores para a GPU e o resultado "
            f"de volta, {sv['gpu_com_copias_ms']:.0f}: mais lento que a CPU."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar[:2])
            self.wait_until_bookmark("c")
            self.mostra_barras(bar[2:])
        self.limpa()

        topo = self.topo("Quantas contas por byte movido?")
        soma = VGroup(self.texto("soma de vetores", 30, BAD_C), self.texto("1 conta para cada 12 bytes", 26),
                      self.texto("limitada pela memória", 24, MUTED)).arrange(DOWN, buff=0.2)
        mm = VGroup(self.texto("multiplicação de matrizes", 30, OK_C), self.texto("2n³ contas sobre 3n² números", 26),
                    self.texto("limitada pelo cálculo", 24, MUTED)).arrange(DOWN, buff=0.2)
        VGroup(soma, mm).arrange(RIGHT, buff=1.5).next_to(topo, DOWN, buff=0.9)
        with self.voiceover(
            "O segredo é quantas contas você faz para cada byte que move. A soma faz uma conta para cada doze bytes: "
            "quem manda é a memória, e o cano até a GPU é estreito. <bookmark mark='m'/>Já a multiplicação de matrizes "
            "reaproveita cada número muitas vezes. Muita conta por byte, e aí a GPU dispara."):
            self.play(FadeIn(topo), FadeIn(soma))
            self.wait_until_bookmark("m")
            self.play(FadeIn(mm))
        self.limpa()

        linhas = R.get("matmul", {}).get("linhas", [])
        topo = self.topo("Multiplicação de matrizes: GFLOP/s (bilhões de contas por segundo)")
        if linhas:
            n_max = max(l["n"] for l in linhas)
            f32 = next(l for l in linhas if l["n"] == n_max and l["tipo"] == "float32")
            f64 = next(l for l in linhas if l["n"] == n_max and l["tipo"] == "float64")
            gf = lambda t: 2 * n_max**3 / t / 1e9  # noqa: E731
            itens = [("CPU (float32)", gf(f32["cpu_s"]), CPU_C), ("GPU float32", gf(f32["gpu_s"]), OK_C),
                     ("GPU float64", gf(f64["gpu_s"]), BAD_C)]
        else:
            n_max, itens = 4096, [("CPU (float32)", 290, CPU_C), ("GPU float32", 1800, OK_C), ("GPU float64", 40, BAD_C)]
        bar = self.barras(itens, fmt=lambda v: f"{v:.0f}").next_to(topo, DOWN, buff=0.9)
        fonte = self.fonte(f"matrizes {n_max}×{n_max} · {GPU_NOME}")
        with self.voiceover(
            f"Com matrizes de {n_max} por {n_max}, a GPU em precisão simples, float32, foi "
            f"{itens[1][1]/itens[0][1]:.0f} vezes mais rápida que a CPU. <bookmark mark='d'/>Mas, em precisão dupla, "
            "float64, ela despencou: placas de consumo, como a T4 e as GeForce, têm pouquíssimas unidades de precisão "
            "dupla. Confira se float32 basta para o seu problema."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar[:2])
            self.wait_until_bookmark("d")
            self.mostra_barras(bar[2:])
        regra = self.texto("Regra: leve o dado uma vez, faça MUITAS contas lá, traga só o resultado.", 26, MEM_C)
        regra.next_to(bar, DOWN, buff=0.7)
        with self.voiceover("A regra prática: leve o dado uma vez, faça muitas contas por lá, e traga só o resultado."):
            self.play(FadeIn(regra))
        self.wait(1)


# ------------------------------------------------------------------ 09
class V09Kernel(CursoScene):
    def construct(self):
        t = self.titulo("Grid, block e thread: escrevendo um kernel", "Aula 2 · vídeo 9")
        with self.voiceover("Como milhares de threads sabem qual pedaço do problema é delas?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Um kernel é a função que CADA thread executa")
        dados = VGroup(*[Square(0.36, color=DATA_C, fill_opacity=0.6) for _ in range(24)]).arrange(RIGHT, buff=0.05)
        dados.next_to(topo, DOWN, buff=0.7)
        blocos = VGroup()
        for b in range(3):
            r = SurroundingRectangle(dados[b * 8:(b + 1) * 8], color=GPU_C, buff=0.06)
            lab = self.texto(f"bloco {b}", 22, GPU_C).next_to(r, DOWN, buff=0.15)
            blocos.add(VGroup(r, lab))
        grade = self.texto("grade (grid) = 3 blocos × 8 threads", 26, MUTED).next_to(blocos, DOWN, buff=0.4)
        with self.voiceover(
            "Um kernel é uma função que cada thread da GPU executa, uma vez. Cada thread cuida de um elemento. "
            "<bookmark mark='b'/>As threads são organizadas em blocos, e os blocos formam a grade, o grid. Aqui, três "
            "blocos de oito threads. Na prática, usamos blocos de duzentas e cinquenta e seis threads e milhares de "
            "blocos."):
            self.play(FadeIn(topo), FadeIn(dados))
            self.wait_until_bookmark("b")
            self.play(LaggedStart(*[Create(b[0]) for b in blocos], lag_ratio=0.3),
                      LaggedStart(*[FadeIn(b[1]) for b in blocos], lag_ratio=0.3))
            self.play(FadeIn(grade))

        formula = self.texto("i = blockIdx.x · blockDim.x + threadIdx.x", 30, MEM_C).next_to(grade, DOWN, buff=0.6)
        ex = self.texto("bloco 2, thread 3  →  i = 2 · 8 + 3 = 19", 28).next_to(formula, DOWN, buff=0.3)
        with self.voiceover(
            "Cada thread descobre o seu índice global com uma conta: o número do bloco, vezes o tamanho do bloco, "
            "mais a posição dentro do bloco. <bookmark mark='e'/>A thread três do bloco dois cuida do elemento "
            "dezenove."):
            self.play(FadeIn(formula))
            self.wait_until_bookmark("e")
            self.play(FadeIn(ex), Indicate(dados[19], color=MEM_C, scale_factor=1.6))
        self.limpa()

        topo = self.topo("Em Python, com Numba-CUDA")
        cod = self.codigo('''
from numba import cuda

@cuda.jit
def escala(x, a):
    i = cuda.grid(1)          # índice global desta thread
    if i < x.size:            # a última grade passa do fim do array
        x[i] = a * x[i] + 1.0

blocos = (x.size + 255) // 256
escala[blocos, 256](x_gpu, 2.0)   # [grade, bloco]
''').next_to(topo, DOWN, buff=0.4)
        with self.voiceover(
            "Em Python, com o Numba-CUDA, fica assim. cuda ponto grid dá o índice global. <bookmark mark='if'/>O if é "
            "obrigatório: como a grade é um múltiplo de duzentos e cinquenta e seis, as últimas threads passariam do "
            "fim do array. <bookmark mark='l'/>Na chamada, entre colchetes, vão o número de blocos e o tamanho do "
            "bloco."):
            self.play(FadeIn(topo), FadeIn(cod))
            self.wait_until_bookmark("if")
            self.wait_until_bookmark("l")
        pi = R.get("pi_monte_carlo", {})
        if pi:
            nota = self.texto(f"Na aula: Monte Carlo com {pi['amostras']/1e9:.2f} bilhão de amostras em "
                              f"{pi['numba_ms']:.0f} ms", 26, OK_C).to_edge(DOWN, buff=0.4)
            with self.voiceover(
                f"Com esse padrão, a GPU da aula sorteou {pi['amostras']/1e9:.2f} bilhão de pontos de Monte Carlo em "
                f"{pi['numba_ms']:.0f} milissegundos."):
                self.play(FadeIn(nota))
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


class V10Calor(CursoScene):
    def construct(self):
        t = self.titulo("Equação do calor na GPU, passo a passo", "Aula 2 · vídeo 10")
        with self.voiceover("Uma laje de concreto com a face de baixo num incêndio. Como o calor sobe?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Cada ponto: média dos vizinhos, a cada passo")
        quadros = _calor()
        mapa = _mapa(quadros[0]).shift(LEFT * 3.2 + DOWN * 0.4)
        lab_f = self.texto("incêndio 600 °C", 22, BAD_C).next_to(mapa, DOWN, buff=0.15)
        est = VGroup(*[Square(0.7, color=c, fill_opacity=0.25) for c in (MUTED, MUTED, MEM_C, MUTED, MUTED)])
        est[0].move_to([0, 0.75, 0]); est[1].move_to([-0.75, 0, 0]); est[2].move_to([0, 0, 0])
        est[3].move_to([0.75, 0, 0]); est[4].move_to([0, -0.75, 0])
        est.shift(RIGHT * 3.3 + UP * 0.6)
        eq = self.texto("T novo = T + r·(soma dos 4 vizinhos − 4·T)", 24, MEM_C).next_to(est, DOWN, buff=0.4)
        with self.voiceover(
            "Dividimos a laje numa malha. Em cada passo de tempo, a temperatura de cada ponto se aproxima da média dos "
            "quatro vizinhos. <bookmark mark='p'/>Repetindo isso centenas de vezes, o calor do incêndio sobe pela laje."):
            self.play(FadeIn(topo), FadeIn(mapa), FadeIn(lab_f), FadeIn(est), FadeIn(eq))
            self.wait_until_bookmark("p")
            for q in quadros[1:]:
                self.play(mapa.animate.become(_mapa(q).move_to(mapa)), run_time=1.3)
        gpu_t = self.texto("milhões de pontos, todos ao mesmo tempo:\numa thread da GPU por ponto", 26, GPU_C)
        gpu_t.next_to(eq, DOWN, buff=0.5)
        with self.voiceover(
            "Todos os pontos são atualizados ao mesmo tempo, a cada passo. É exatamente o que a GPU gosta: uma thread "
            "para cada ponto da malha."):
            self.play(FadeIn(gpu_t))
        self.limpa()

        topo = self.topo("Uma função, CPU ou GPU (padrão Array API)")
        cod = self.codigo('''
def simula(T, passos):
    xp = array_namespace(T)          # numpy ou cupy, conforme T
    A, B = T, xp.asarray(T, copy=True)
    for _ in range(passos):
        B[1:-1, 1:-1] = A[1:-1, 1:-1] + R * (A[2:, 1:-1] + A[:-2, 1:-1]
                        + A[1:-1, 2:] + A[1:-1, :-2] - 4 * A[1:-1, 1:-1])
        A, B = B, A
    return A
''', largura=11).next_to(topo, DOWN, buff=0.5)
        with self.voiceover(
            "No notebook, escrevemos a simulação uma única vez. A função descobre se recebeu um array do NumPy ou do "
            "CuPy e usa a biblioteca certa. O mesmo código roda na CPU e na GPU."):
            self.play(FadeIn(topo), FadeIn(cod))
        self.limpa()

        tempos = R.get("calor", {}).get("tempos", {})
        n_max = "2048"
        itens = []
        for nome, cor in (("Numba paralelo", CPU_C), ("CuPy (GPU)", MEM_C), ("Numba-CUDA kernel (GPU)", OK_C)):
            v = tempos.get(nome, {}).get(n_max)
            if v:
                itens.append((nome.replace(" (GPU)", " · GPU"), v, cor))
        topo = self.topo("Malha 2048 × 2048, 300 passos")
        if itens:
            bar = self.barras(itens, fmt=lambda v: f"{v:.2f} s").next_to(topo, DOWN, buff=0.9)
            fonte = self.fonte(f"medido na aula · {GPU_NOME}")
            with self.voiceover(
                "Os tempos na aula. O kernel próprio, em que cada thread lê e escreve a memória uma vez só, vence a versão "
                "com fatias do CuPy, que cria arrays temporários. <bookmark mark='b'/>E a comparação com a CPU depende de "
                "um número: largura de banda. Esse problema faz poucas contas por ponto, então ganha quem lê a memória "
                "mais rápido. No Colab, a T4 vence com folga. Num notebook com CPU forte e GPU de entrada, eles empatam."):
                self.play(FadeIn(topo), FadeIn(fonte))
                self.mostra_barras(bar)
                self.wait_until_bookmark("b")
        fecho = self.texto("Meça no SEU hardware antes de concluir.", 30, MEM_C).to_edge(DOWN, buff=0.9)
        with self.voiceover("Moral da história: meça no seu hardware antes de concluir."):
            self.play(FadeIn(fecho))
        self.wait(1)
