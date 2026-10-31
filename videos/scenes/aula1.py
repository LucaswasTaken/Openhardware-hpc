"""Vídeos da Aula 1 (CPU): 01 medir · 02 Amdahl · 03 GIL · 04 Monte Carlo · 05 joblib · 06 Numba."""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Axes, Circle, Create, Dot, FadeIn, FadeOut, Indicate,
                   LaggedStart, Line, Rectangle, Square, Transform, ValueTracker, VGroup, Write, always_redraw)

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe, results

R = results(1)
NUC = R.get("nucleos", 12)
MAQ = "Colab (2 núcleos)" if R.get("maquina") == "colab" else f"notebook com {NUC} núcleos lógicos"


def g(*chaves, padrao=0.0):
    d = R
    for c in chaves:
        d = d.get(c, {}) if isinstance(d, dict) else {}
    return d if isinstance(d, (int, float)) and d else padrao


def vezes(x):
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


# ------------------------------------------------------------------ 01
class V01Medir(CursoScene):
    def construct(self):
        t = self.titulo("Por que paralelizar, e como medir", "Aula 1 · vídeo 1")
        with self.voiceover("Por que engenheiros precisam de computação de alto desempenho? E por onde começar?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Engenharia = muitas contas independentes")
        exemplos = VGroup(*[self.caixa(s, c, w=3.6, h=1.0, fs=24) for s, c in [
            ("casos de carga", CPU_C), ("posições de um veículo", CPU_C), ("amostras de Monte Carlo", GPU_C),
            ("elementos de uma malha", GPU_C), ("vigas de um estudo", MEM_C), ("pontos de integração", MEM_C)]])
        exemplos.arrange_in_grid(2, 3, buff=0.4).next_to(topo, DOWN, buff=0.7)
        with self.voiceover(
            "Boa parte das simulações de engenharia é a mesma conta repetida muitas vezes, com dados diferentes: "
            "casos de carga, posições de um veículo numa ponte, amostras de Monte Carlo, elementos de uma malha. "
            "Contas independentes podem ser feitas ao mesmo tempo. Isso é paralelismo de dados."):
            self.play(FadeIn(topo))
            self.play(LaggedStart(*[FadeIn(e, shift=UP * 0.2) for e in exemplos], lag_ratio=0.3), run_time=4)
        self.limpa()

        topo = self.topo("Regra de ouro: meça antes de otimizar")
        ferr = VGroup(
            self.texto("time.perf_counter()  →  quanto tempo levou", 26),
            self.texto("%timeit  →  repete e mostra a variação", 26),
            self.texto("cProfile  →  quais funções gastam o tempo", 26),
            self.texto("line_profiler  →  quais linhas", 26)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        ferr.next_to(topo, DOWN, buff=0.8)
        with self.voiceover(
            "Antes de paralelizar, meça. O perf counter diz quanto tempo levou. O timeit repete a medida e mostra a "
            "variação. O cProfile mostra quais funções gastam o tempo, e o line profiler, quais linhas. Sem medir, "
            "você corre o risco de acelerar o que não importa."):
            self.play(FadeIn(topo))
            self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.2) for f in ferr], lag_ratio=0.5), run_time=4)
        self.limpa()

        py, npy = g("envoltoria", "python_s", padrao=0.3), g("envoltoria", "numpy_s", padrao=0.03)
        topo = self.topo("Exemplo: envoltória de momentos numa viga")
        cod = self.codigo('''
for x in secoes:                 # 1500 seções
    for a in posicoes_da_carga:  # 1500 posições
        M = momento(x, a, L)     # 2,25 milhões de chamadas
''').next_to(topo, DOWN, buff=0.5)
        barras = self.barras([("Python puro", py, BAD_C), ("NumPy (o laço vai para C)", npy, OK_C)],
                             fmt=lambda v: f"{v*1e3:.0f} ms").next_to(cod, DOWN, buff=0.6)
        with self.voiceover(
            "No nosso exemplo, a envoltória de momentos de uma viga, o perfilador mostrou mais de dois milhões de "
            f"chamadas à mesma função. <bookmark mark='b'/>Só de deixar o NumPy fazer o laço, em C, ficou "
            f"{py/npy:.0f} vezes mais rápido, ainda num único núcleo."):
            self.play(FadeIn(topo), FadeIn(cod))
            self.wait_until_bookmark("b")
            self.mostra_barras(barras)
        fecho = self.texto("Primeiro use bem UM núcleo. Depois, vários.", 32, OK_C).to_edge(DOWN, buff=0.4)
        with self.voiceover("A primeira lição: use bem um núcleo. Depois, vários."):
            self.play(FadeIn(fecho))
        self.wait(1)


# ------------------------------------------------------------------ 02
class V02Amdahl(CursoScene):
    def construct(self):
        t = self.titulo("Speedup, eficiência e a lei de Amdahl", "Aula 1 · vídeo 2")
        with self.voiceover("Quanto mais rápido fica um programa com mais núcleos? Existe um limite."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Speedup e eficiência")
        f1 = self.texto("speedup   S = T₁ / Tₚ", 34)
        f2 = self.texto("eficiência   E = S / p", 34)
        VGroup(f1, f2).arrange(DOWN, buff=0.5).next_to(topo, DOWN, buff=1.0)
        with self.voiceover(
            "Speedup é o tempo com um núcleo dividido pelo tempo com p núcleos. <bookmark mark='e'/>Eficiência é o "
            "speedup dividido por p: a fração do hardware que você está aproveitando. Um é perfeito."):
            self.play(FadeIn(topo), FadeIn(f1))
            self.wait_until_bookmark("e")
            self.play(FadeIn(f2))
        self.limpa()

        # barra serial + paralela encolhendo
        topo = self.topo("Lei de Amdahl: a parte serial não encolhe")
        larg = 10.0
        serial_w = larg * 0.1
        base_y = 1.2
        linhas = VGroup()
        for i, p in enumerate((1, 2, 4, 8, 16)):
            y = base_y - i * 0.85
            s = Rectangle(width=serial_w, height=0.5, color=BAD_C, fill_opacity=0.85, stroke_width=0)
            par = Rectangle(width=(larg - serial_w) / p, height=0.5, color=OK_C, fill_opacity=0.85, stroke_width=0)
            s.move_to([-5.0, y, 0], aligned_edge=LEFT)
            par.next_to(s, RIGHT, buff=0)
            rot = self.texto(f"p = {p}", 22).next_to(s, LEFT, buff=0.25)
            sp = 1 / (0.1 + 0.9 / p)
            val = self.texto(f"S = {sp:.1f}×", 22, MUTED).next_to(par, RIGHT, buff=0.25)
            linhas.add(VGroup(rot, s, par, val))
        leg = VGroup(self.texto("■ parte serial (10%)", 22, BAD_C), self.texto("■ parte paralela (90%)", 22, OK_C))
        leg.arrange(RIGHT, buff=0.8).to_edge(DOWN, buff=0.4)
        with self.voiceover(
            "Imagine um programa em que noventa por cento do tempo pode ser dividido entre os núcleos, e dez por "
            "cento, não. <bookmark mark='p'/>Com dois, quatro, oito, dezesseis núcleos, a parte verde encolhe, mas "
            "a vermelha continua lá. <bookmark mark='lim'/>Mesmo com infinitos núcleos, o speedup nunca passa de dez "
            "vezes. Essa é a lei de Amdahl."):
            self.play(FadeIn(topo), FadeIn(leg))
            self.wait_until_bookmark("p")
            self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.3) for l in linhas], lag_ratio=0.5), run_time=4)
            self.wait_until_bookmark("lim")
            self.play(Indicate(VGroup(*[l[1] for l in linhas]), color=BAD_C))
        self.limpa()

        topo = self.topo("Quanto do código precisa ser paralelo?")
        eixos = Axes(x_range=[0, 7, 1], y_range=[0, 7, 1], x_length=8, y_length=4.6,
                     axis_config={"color": MUTED, "include_ticks": False}).next_to(topo, DOWN, buff=0.4)
        xr = self.texto("log₂ núcleos  (1 → 128)", 20, MUTED).next_to(eixos, DOWN, buff=0.15)
        yr = self.texto("log₂ speedup", 20, MUTED).rotate(np.pi / 2).next_to(eixos, LEFT, buff=0.15)
        ideal = eixos.plot(lambda x: x, color=MUTED, x_range=[0, 7])
        curvas = VGroup()
        nomes = VGroup()
        for f, cor in ((0.5, BAD_C), (0.9, MEM_C), (0.95, CPU_C), (0.99, OK_C)):
            c = eixos.plot(lambda x, f=f: np.log2(1 / ((1 - f) + f / 2**x)), color=cor, x_range=[0, 7])
            curvas.add(c)
            nomes.add(self.texto(f"{f:.0%} paralelo → máx {1/(1-f):.0f}×", 20, cor))
        nomes.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(eixos, RIGHT, buff=0.2).shift(DOWN * 0.6)
        cabe(VGroup(eixos, ideal, curvas, nomes, xr, yr), 12.6).set_x(0)
        with self.voiceover(
            "Veja as curvas. Com metade do código paralelo, o máximo é duas vezes. Com noventa e nove por cento, cem "
            "vezes. Por isso, antes de comprar mais núcleos, reduza a parte serial."):
            self.play(FadeIn(topo), Create(eixos), FadeIn(xr), FadeIn(yr), Create(ideal))
            self.play(LaggedStart(*[Create(c) for c in curvas], lag_ratio=0.4),
                      LaggedStart(*[FadeIn(n) for n in nomes], lag_ratio=0.4), run_time=4)
        self.limpa()

        topo = self.topo("Gustafson: e se o problema crescer junto?")
        a = VGroup(self.texto("Escalabilidade FORTE", 30, CPU_C),
                   self.texto("mesmo problema, mais núcleos → o tempo cai?", 24),
                   self.texto("(limite de Amdahl)", 22, MUTED)).arrange(DOWN, buff=0.2)
        b = VGroup(self.texto("Escalabilidade FRACA", 30, OK_C),
                   self.texto("problema cresce com os núcleos → o tempo se mantém?", 24),
                   self.texto("(lei de Gustafson: malhas mais finas, mais amostras)", 22, MUTED)).arrange(DOWN, buff=0.2)
        VGroup(a, b).arrange(DOWN, buff=0.9).next_to(topo, DOWN, buff=0.7)
        with self.voiceover(
            "Mas, na prática, quando ganhamos núcleos, também aumentamos o problema: malhas mais finas, mais amostras. "
            "<bookmark mark='a'/>Medir com o problema fixo é a escalabilidade forte. <bookmark mark='b'/>Medir com o "
            "problema crescendo junto é a escalabilidade fraca, a lei de Gustafson. Nos exercícios, você vai medir as "
            "duas."):
            self.play(FadeIn(topo))
            self.wait_until_bookmark("a"); self.play(FadeIn(a))
            self.wait_until_bookmark("b"); self.play(FadeIn(b))
        self.wait(1)


# ------------------------------------------------------------------ 03
def _sem_gil():
    out = {}
    for k, v in (R.get("python_sem_gil") or {}).items():
        m = re.search(r"speedup\s+([\d.]+)", v)
        if m:
            out[k] = float(m.group(1))
    return out.get("3.14", 1.0), out.get("3.14t", 2.6)


class V03GIL(CursoScene):
    def construct(self):
        t = self.titulo("Threads × processos, o GIL e o Python sem GIL", "Aula 1 · vídeo 3")
        with self.voiceover("Threads ou processos? E o que é esse tal de GIL?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Processos: cada um com sua memória")
        procs = VGroup(*[VGroup(self.caixa("Python", CPU_C, w=2.0, h=0.8, fs=22),
                                self.caixa("memória", MEM_C, w=2.0, h=0.6, fs=18)).arrange(DOWN, buff=0.1)
                         for _ in range(3)]).arrange(RIGHT, buff=0.8).next_to(topo, DOWN, buff=0.7)
        nota = self.texto("isolados · pesados para criar · dados são copiados", 24, MUTED).next_to(procs, DOWN, buff=0.5)
        with self.voiceover(
            "Um processo é uma cópia independente do interpretador, com a sua própria memória. Eles rodam de verdade "
            "em paralelo, mas custam para criar, e os dados precisam ser copiados entre eles."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(p) for p in procs], lag_ratio=0.3))
            self.play(FadeIn(nota))
        self.limpa()

        topo = self.topo("Threads: memória compartilhada… e um único microfone")
        mem = self.caixa("memória compartilhada", MEM_C, w=8, h=0.8).shift(DOWN * 1.8)
        ths = VGroup(*[self.caixa(f"thread {i+1}", CPU_C, w=2.0, h=0.8, fs=22) for i in range(4)])
        ths.arrange(RIGHT, buff=0.5).shift(UP * 0.3)
        gil = Circle(radius=0.32, color=BAD_C, fill_opacity=0.9)
        gil_t = self.texto("GIL", 20).move_to(gil)
        gil_g = VGroup(gil, gil_t).next_to(ths[0], UP, buff=0.2)
        with self.voiceover(
            "Threads são mais leves e compartilham a mesma memória. Mas no Python tradicional existe o GIL, uma trava "
            "global: <bookmark mark='m'/>como um microfone numa reunião, só quem está com ele executa código Python. "
            "<bookmark mark='p'/>As threads se revezam, e o laço não fica mais rápido."):
            self.play(FadeIn(topo), FadeIn(mem), FadeIn(ths))
            self.wait_until_bookmark("m")
            self.play(FadeIn(gil_g))
            self.wait_until_bookmark("p")
            for i in (1, 2, 3, 0, 1):
                self.play(gil_g.animate.next_to(ths[i], UP, buff=0.2), run_time=0.5)
        self.limpa()

        com, sem = _sem_gil()
        io = g("threads_gil", "io_serial_s", padrao=2.0) / max(g("threads_gil", "io_threads_s", padrao=0.5), 1e-9)
        topo = self.topo("Medido na aula: 4 tarefas em 4 threads")
        bar = self.barras([("laço Python, com GIL", com, BAD_C), ("espera (I/O), com GIL", io, MEM_C),
                           ("laço Python 3.14t, SEM GIL", sem, OK_C)], fmt=vezes).next_to(topo, DOWN, buff=0.9)
        fonte = self.fonte(f"speedup medido · {MAQ}")
        with self.voiceover(
            f"Medimos na aula. Um laço em Python puro com quatro threads: speedup de {com:.1f}, ou seja, nada. "
            f"<bookmark mark='io'/>Já esperando leitura de arquivo ou rede, as threads ajudam, {io:.0f} vezes, porque "
            "quem espera solta o microfone. <bookmark mark='t'/>E a novidade: desde o Python 3.13 existe a versão sem "
            f"GIL. No Python 3.14t, o mesmo laço ficou {sem:.1f} vezes mais rápido."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.play(FadeIn(bar[0][0]), FadeIn(bar[0][1]), FadeIn(bar[0][2]))
            self.wait_until_bookmark("io")
            self.play(FadeIn(bar[1]))
            self.wait_until_bookmark("t")
            self.play(FadeIn(bar[2]), Indicate(bar[2][1], color=OK_C))
        fecho = self.texto("Para cálculo numérico: processos, NumPy, Numba ou GPU.", 28, MUTED).next_to(bar, DOWN, buff=0.6)
        with self.voiceover(
            "Mesmo sem o GIL, cuidado com duas threads escrevendo na mesma variável. Para cálculo numérico, o "
            "caminho principal continua sendo processos, NumPy, Numba ou GPU."):
            self.play(FadeIn(fecho))
        self.wait(1)


# ------------------------------------------------------------------ 04
class V04MonteCarlo(CursoScene):
    def construct(self):
        t = self.titulo("Monte Carlo em paralelo com processos", "Aula 1 · vídeo 4")
        with self.voiceover("Qual é a probabilidade de uma viga falhar? Vamos sortear milhões de vigas."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Falha quando a solicitação S supera a resistência R")
        eixos = Axes(x_range=[100, 500, 100], y_range=[0, 0.02, 0.01], x_length=9, y_length=3.6,
                     axis_config={"color": MUTED, "include_ticks": False}).next_to(topo, DOWN, buff=0.6)
        from math import exp, log, pi, sqrt

        def logn(x, mu=log(350), s=0.1):
            return exp(-(log(x) - mu) ** 2 / (2 * s * s)) / (x * s * sqrt(2 * pi))

        def gumb(x, loc=200, b=25):
            z = (x - loc) / b
            return exp(-(z + exp(-z))) / b
        cR = eixos.plot(logn, x_range=[200, 500], color=CPU_C)
        cS = eixos.plot(gumb, x_range=[120, 400], color=BAD_C)
        lR = self.texto("resistência R", 22, CPU_C).next_to(eixos.c2p(380, 0.013), RIGHT)
        lS = self.texto("solicitação S", 22, BAD_C).next_to(eixos.c2p(140, 0.016), RIGHT)
        with self.voiceover(
            "Uma viga tem resistência R, que varia por causa do material, e recebe uma solicitação S, a maior carga "
            "do ano, que também varia. Ela falha quando S passa de R. <bookmark mark='mc'/>Monte Carlo: sorteie milhões "
            "de pares e conte as falhas."):
            self.play(FadeIn(topo), Create(eixos), Create(cR), Create(cS), FadeIn(lR), FadeIn(lS))
            self.wait_until_bookmark("mc")
            rng = np.random.default_rng(2)
            pts = VGroup()
            for _ in range(60):
                r = rng.lognormal(log(350), 0.1); s = rng.gumbel(200, 25)
                cor = BAD_C if s > r else OK_C
                pts.add(Dot(eixos.c2p(min(max(s, 110), 490), 0.0015 + rng.random() * 0.002), radius=0.05, color=cor))
            self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.05), run_time=2.5)
        self.limpa()

        tempos = R.get("monte_carlo_processos", {}).get("tempos", {"1": 7.1, "2": 4.1, "4": 3.0, "12": 3.2})
        ps = sorted(int(p) for p in tempos)
        t1 = tempos["1"]
        topo = self.topo("160 milhões de amostras em 64 lotes, divididos entre processos")
        lotes = VGroup(*[Square(0.32, color=DATA_C, fill_opacity=0.8) for _ in range(16)]).arrange_in_grid(2, 8, buff=0.08)
        lotes.next_to(topo, DOWN, buff=0.5)
        procs = VGroup(*[self.caixa(f"processo {i+1}", CPU_C, w=2.4, h=0.7, fs=20) for i in range(4)]).arrange(RIGHT, buff=0.3)
        procs.next_to(lotes, DOWN, buff=0.8)
        with self.voiceover(
            "Cada lote de amostras é independente, com a sua própria semente aleatória. <bookmark mark='d'/>O "
            "ProcessPoolExecutor entrega os lotes aos processos, e no fim somamos as falhas."):
            self.play(FadeIn(topo), FadeIn(lotes))
            self.wait_until_bookmark("d")
            self.play(FadeIn(procs))
            anims = [lotes[i].animate.move_to(procs[i % 4]).set_opacity(0) for i in range(16)]
            self.play(LaggedStart(*anims, lag_ratio=0.1), run_time=2.5)
        self.limpa()

        topo = self.topo("Medido na aula")
        bar = self.barras([(f"{p} processo{'s' if p > 1 else ''}", t1 / tempos[str(p)], OK_C if p > 1 else MUTED)
                           for p in ps], fmt=lambda v: f"{v:.2f}×").next_to(topo, DOWN, buff=0.8)
        fonte = self.fonte(f"speedup · {MAQ}")
        sp_max = max(t1 / tempos[str(p)] for p in ps)
        with self.voiceover(
            f"O speedup chegou a {sp_max:.1f} vezes, e não ao número de processos. Criar processos, enviar as tarefas "
            "e juntar resultados custa tempo. E núcleos lógicos não são núcleos físicos inteiros."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar)
        aviso = VGroup(self.texto("No Windows e no macOS, o processo novo começa do zero:", 24),
                       self.texto("a função do trabalhador precisa estar num arquivo .py", 24, MEM_C)).arrange(DOWN, buff=0.15)
        aviso.next_to(bar, DOWN, buff=0.6)
        with self.voiceover(
            "Um detalhe prático: no Windows e no macOS, cada processo novo começa do zero e precisa importar a função "
            "do trabalhador. Por isso, no notebook, gravamos essa função num arquivo ponto py."):
            self.play(FadeIn(aviso))
        self.wait(1)


# ------------------------------------------------------------------ 05
class V05Joblib(CursoScene):
    def construct(self):
        t = self.titulo("joblib: estudo paramétrico em 3 linhas", "Aula 1 · vídeo 5")
        with self.voiceover("Quase dois mil projetos de viga para verificar. Vamos usar todos os núcleos com o joblib."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        casos = int(R.get("joblib_vigas", {}).get("casos", 960))
        topo = self.topo(f"{casos} vigas: vão × altura × carga")
        grade = VGroup(*[Rectangle(width=0.5, height=0.18, color=CPU_C, fill_opacity=0.6, stroke_width=1)
                         for _ in range(16 * 8)]).arrange_in_grid(8, 16, buff=0.06).next_to(topo, DOWN, buff=0.5)
        txt = self.texto("cada uma: análise por elementos finitos + verificação da flecha L/250", 24, MUTED).next_to(grade, DOWN, buff=0.4)
        with self.voiceover(
            "No projeto preliminar, testamos combinações de vão, altura da seção e carga. Cada combinação é uma "
            "análise por elementos finitos, independente das outras. É o caso clássico de problema "
            "embaraçosamente paralelo."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(r) for r in grade], lag_ratio=0.01), run_time=2.5)
            self.play(FadeIn(txt))
        self.limpa()

        topo = self.topo("Do laço serial ao paralelo")
        c1 = self.codigo('''
res = [verifica(c) for c in casos]
''')
        c2 = self.codigo('''
from joblib import Parallel, delayed

res = Parallel(n_jobs=-1)(delayed(verifica)(c) for c in casos)
''')
        VGroup(c1, c2).arrange(DOWN, buff=0.6).next_to(topo, DOWN, buff=0.6)
        with self.voiceover(
            "O laço serial vira, com o joblib, uma única linha: Parallel, com n jobs igual a menos um para usar "
            "todos os núcleos, e delayed na função. O joblib cria os processos, distribui as tarefas e devolve os "
            "resultados na ordem."):
            self.play(FadeIn(topo), FadeIn(c1))
            self.play(FadeIn(c2, shift=UP * 0.2))
        self.limpa()

        topo = self.topo("Cuidado: sobreassinatura")
        procs = VGroup(*[VGroup(self.caixa("processo", CPU_C, w=2.2, h=0.7, fs=20),
                                VGroup(*[Square(0.3, color=BAD_C, fill_opacity=0.8) for _ in range(4)]).arrange(RIGHT, buff=0.08))
                         .arrange(DOWN, buff=0.15) for _ in range(4)]).arrange(RIGHT, buff=0.5).next_to(topo, DOWN, buff=0.6)
        t1 = self.texto("4 processos × 4 threads do BLAS = 16 threads em 4 núcleos → MAIS LENTO", 24, BAD_C)
        t2 = self.texto("solução: limite o BLAS a 1 thread por processo (threadpoolctl)", 24, OK_C)
        VGroup(t1, t2).arrange(DOWN, buff=0.3).next_to(procs, DOWN, buff=0.6)
        with self.voiceover(
            "Um cuidado importante. O NumPy usa o BLAS, que já é paralelo com threads. Quatro processos, cada um com "
            "quatro threads, disputam quatro núcleos e tudo fica mais lento. <bookmark mark='s'/>A solução é limitar o "
            "BLAS a uma thread dentro de cada processo."):
            self.play(FadeIn(topo), FadeIn(procs))
            self.play(FadeIn(t1))
            self.wait_until_bookmark("s")
            self.play(*[p[1][1:].animate.set_opacity(0.1) for p in procs], FadeIn(t2))
        self.limpa()

        jl = R.get("joblib_vigas", {})
        serial = jl.get("serial_s", 8.0)
        tempos = jl.get("tempos", {"2": 4.0, "4": 2.2})
        topo = self.topo("Medido na aula")
        bar = self.barras([("serial", 1.0, MUTED)] + [(f"n_jobs = {k}", serial / v, OK_C) for k, v in tempos.items()],
                          fmt=lambda v: f"{v:.1f}×").next_to(topo, DOWN, buff=0.8)
        fonte = self.fonte(f"speedup · {MAQ}")
        melhor = max(serial / v for v in tempos.values())
        with self.voiceover(
            f"Na aula, o estudo ficou {melhor:.1f} vezes mais rápido. Três linhas de código, e o resultado de "
            "engenharia sai junto: a altura mínima de cada vão."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar)
        self.wait(1)


# ------------------------------------------------------------------ 06
class V06Numba(CursoScene):
    def construct(self):
        t = self.titulo("Numba: do laço lento ao código compilado", "Aula 1 · vídeo 6")
        with self.voiceover("E quando o problema não cabe em operações de array? Compile o laço."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Como o Numba funciona")
        etapas = VGroup(self.caixa("função Python\n@njit", CPU_C, w=2.8, h=1.2, fs=22),
                        self.caixa("LLVM\n(compilador)", MEM_C, w=2.8, h=1.2, fs=22),
                        self.caixa("código de\nmáquina", OK_C, w=2.8, h=1.2, fs=22)).arrange(RIGHT, buff=1.0)
        etapas.next_to(topo, DOWN, buff=1.0)
        setas = VGroup(Arrow(etapas[0].get_right(), etapas[1].get_left(), buff=0.1, color=MUTED),
                       Arrow(etapas[1].get_right(), etapas[2].get_left(), buff=0.1, color=MUTED))
        nota = self.texto("na 1ª chamada (alguns segundos) · cache=True guarda para a próxima vez", 22, MUTED)
        nota.next_to(etapas, DOWN, buff=0.7)
        with self.voiceover(
            "O Numba lê a sua função Python, com laços e condições, e na primeira chamada compila tudo para código de "
            "máquina, usando o LLVM, o mesmo compilador de linguagens como C e Rust. A primeira chamada demora uns "
            "segundos. As seguintes são rápidas."):
            self.play(FadeIn(topo), FadeIn(etapas[0]))
            self.play(Create(setas[0]), FadeIn(etapas[1]))
            self.play(Create(setas[1]), FadeIn(etapas[2]))
            self.play(FadeIn(nota))
        self.limpa()

        topo = self.topo("Paralelo: prange divide o laço entre os núcleos")
        cod = self.codigo('''
@njit(parallel=True, cache=True)
def rigidez(xy, barras, EA):
    k = np.empty((n, 4, 4))
    for e in prange(n):      # cada barra numa thread
        ...                  # matriz local 4×4 da barra e
    return k
''').next_to(topo, DOWN, buff=0.4)
        nucleos = VGroup(*[self.caixa(f"núcleo {i+1}", CPU_C, w=2.0, h=0.6, fs=18) for i in range(4)]).arrange(RIGHT, buff=0.3)
        nucleos.next_to(cod, DOWN, buff=0.5)
        with self.voiceover(
            "Com parallel igual a True e prange no lugar de range, o Numba divide as iterações entre os núcleos, com "
            "threads de verdade: código compilado não precisa do GIL. A condição é que cada iteração seja "
            "independente das outras."):
            self.play(FadeIn(topo), FadeIn(cod))
            self.play(LaggedStart(*[FadeIn(n) for n in nucleos], lag_ratio=0.3))
        self.limpa()

        nb = R.get("numba_trelica", {})
        py, npy, n1, np_ = (nb.get("python_s", 21), nb.get("numpy_s", 0.68), nb.get("numba_s", 0.17),
                            nb.get("numba_par_s", 0.038))
        topo = self.topo("Treliça com 2 milhões de barras: matrizes de rigidez")
        bar = self.barras([("Python puro", py / py, MUTED), ("NumPy", py / npy, CPU_C), ("Numba, 1 núcleo", py / n1, MEM_C),
                           (f"Numba paralelo ({nb.get('threads', NUC)} threads)", py / np_, OK_C)],
                          fmt=vezes, log=True).next_to(topo, DOWN, buff=0.8)
        fonte = self.fonte(f"vezes mais rápido que Python puro (escala logarítmica) · {MAQ}")
        with self.voiceover(
            f"O resultado na aula: o NumPy foi {py/npy:.0f} vezes mais rápido que o Python puro. O Numba em um núcleo, "
            f"{py/n1:.0f} vezes, porque não cria arrays temporários. <bookmark mark='p'/>E o Numba paralelo, "
            f"{py/np_:.0f} vezes."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar[:3])
            self.wait_until_bookmark("p")
            self.mostra_barras(bar[3:])
        self.wait(1)
