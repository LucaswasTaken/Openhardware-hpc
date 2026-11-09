"""Vídeos dos exercícios: 11 relatório (escalabilidade forte) · 12 exemplo da lista com Numba (varrendo threads) ·
13 o mesmo exemplo na GPU (escada de speedup com CuPy) · 14 usando IA sem se perder.

O exemplo é o resolvido em exercicios/lista.md; os números vêm de videos/data/exemplos.json (medir_exemplos.py)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import DOWN, LEFT, RIGHT, UP, Axes, Create, Dot, FadeIn, FadeOut, LaggedStart, Line, VGroup

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe

E = json.loads((Path(__file__).resolve().parents[1] / "data" / "exemplos.json").read_text(encoding="utf-8"))["energia"]
TH = {int(k): v for k, v in E["threads_s"].items()}
PS = sorted(TH)
PMAX = PS[-1]


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def vezes(x):
    return f"{x:,.0f}×".replace(",", ".") if x >= 10 else f"{num(x)}×"


# ------------------------------------------------------------------ 11
class V11Relatorio(CursoScene):
    def construct(self):
        t = self.titulo("Como fazer o relatório", "escalabilidade forte, passo a passo")
        with self.voiceover("Você escolheu um exercício e um caminho, CPU ou GPU. Como mostrar que acelerou de verdade?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Escalabilidade forte: o mesmo problema, mais trabalhadores")
        obra = VGroup(self.texto("obra de área fixa", 28, MEM_C),
                      self.texto("1 pedreiro → 12 dias", 28, CPU_C),
                      self.texto("4 pedreiros → 3 dias?  (se tudo dividir bem)", 28, OK_C)).arrange(DOWN, buff=0.4)
        obra.next_to(topo, DOWN, buff=0.7)
        defs = VGroup(self.texto("speedup = tempo com 1 ÷ tempo com p", 26),
                      self.texto("eficiência = speedup ÷ p   (100% = perfeito)", 26)).arrange(DOWN, buff=0.25)
        defs.to_edge(DOWN, buff=0.6)
        with self.voiceover(
            "O relatório mede uma coisa só: escalabilidade forte. O problema fica do mesmo tamanho, e você aumenta o "
            "número de trabalhadores. É como uma obra de área fixa: se um pedreiro leva doze dias, quatro levariam "
            "três, se tudo se dividir bem. <bookmark mark='d'/>O speedup diz quantas vezes ficou mais rápido. A "
            "eficiência diz quanto do esforço extra foi aproveitado."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(o) for o in obra], lag_ratio=0.6), run_time=3)
            self.wait_until_bookmark("d")
            self.play(FadeIn(defs))
        self.limpa()

        topo = self.topo("Cinco passos (o modelo de relatório faz quase tudo)")
        passos = VGroup(*[self.caixa(s, c, w=11.5, h=0.75, fs=24) for s, c in [
            ("1. Cole a função serial do exercício (está pronta na lista)", MUTED),
            ("2. Acelere: Numba (CPU) ou CuPy (GPU)", CPU_C),
            ("3. Confira: o resultado bate com a serial?", OK_C),
            ("4. Meça: o modelo varia as threads e desenha os gráficos", MEM_C),
            ("5. Analise: o que limitou o ganho?", GPU_C)]]).arrange(DOWN, buff=0.22).next_to(topo, DOWN, buff=0.45)
        with self.voiceover(
            "São cinco passos. Cole a função serial, que já está pronta na lista. Acelere com Numba, na CPU, ou CuPy, na "
            "GPU. <bookmark mark='c'/>Confira que o resultado é o mesmo: rápido e errado vale zero. "
            "<bookmark mark='m'/>Meça; o modelo de relatório varia as threads e desenha os gráficos sozinho. "
            "<bookmark mark='a'/>E, por fim, explique o que limitou o ganho."):
            self.play(FadeIn(topo), FadeIn(passos[0]), FadeIn(passos[1]))
            self.wait_until_bookmark("c"); self.play(FadeIn(passos[2], shift=RIGHT * 0.2))
            self.wait_until_bookmark("m"); self.play(FadeIn(passos[3], shift=RIGHT * 0.2))
            self.wait_until_bookmark("a"); self.play(FadeIn(passos[4], shift=RIGHT * 0.2))
        self.limpa()

        topo = self.topo("Os dois gráficos")
        sp = [TH[1] / TH[p] for p in PS]
        e1 = Axes(x_range=[0, PMAX + 1, 2], y_range=[0, PMAX + 1, 2], x_length=5, y_length=3.6, axis_config={"color": MUTED})
        e2 = Axes(x_range=[0, PMAX + 1, 2], y_range=[0, 1.2, 0.5], x_length=5, y_length=3.6, axis_config={"color": MUTED})
        VGroup(e1, e2).arrange(RIGHT, buff=1.2).next_to(topo, DOWN, buff=0.6)
        ideal = e1.plot(lambda x: x, x_range=[1, PMAX], color=MUTED)
        med = VGroup(*[Dot(e1.c2p(p, s), color=OK_C) for p, s in zip(PS, sp)])
        ef = VGroup(*[Dot(e2.c2p(p, s / p), color=MEM_C) for p, s in zip(PS, sp)])
        t1 = self.texto("speedup × threads (com a reta ideal)", 22).next_to(e1, DOWN, buff=0.2)
        t2 = self.texto("eficiência × threads", 22).next_to(e2, DOWN, buff=0.2)
        with self.voiceover(
            "Os gráficos: speedup em função do número de threads, com a reta ideal para comparar, "
            "<bookmark mark='e'/>e a eficiência. Estes são os do exemplo resolvido da lista. Na GPU, em vez desses "
            "gráficos, você monta uma escada de versões, como mostra o vídeo treze."):
            self.play(FadeIn(topo), Create(e1), Create(ideal), FadeIn(t1))
            self.play(LaggedStart(*[FadeIn(d) for d in med], lag_ratio=0.3))
            self.wait_until_bookmark("e")
            self.play(Create(e2), FadeIn(t2), LaggedStart(*[FadeIn(d) for d in ef], lag_ratio=0.3))
        self.wait(1)


# ------------------------------------------------------------------ 12
class V12Numba(CursoScene):
    def construct(self):
        t = self.titulo("Exemplo da lista: Numba variando as threads", "energia de deformação de 50 milhões de molas")
        with self.voiceover("Vamos resolver o exemplo da lista: a energia de deformação de cinquenta milhões de molas."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Da função serial ao Numba: três mudanças")
        cod = self.codigo('''
from numba import njit, prange, set_num_threads

@njit(parallel=True)                 # 1) compile e paralelize
def energia(k, d):
    u = 0.0
    for i in prange(k.size):         # 2) divide as voltas entre as threads
        u += 0.5 * k[i] * d[i] * d[i]
    return u

for p in (1, 2, 4, 8, 12):
    set_num_threads(p)               # 3) quantas threads nesta medida
''', largura=11).next_to(topo, DOWN, buff=0.4)
        with self.voiceover(
            "A função serial é um laço que soma meio k delta ao quadrado. Para o Numba, são três mudanças: o decorador "
            "njit com parallel igual a true, trocar range por prange, e escolher quantas threads usar em cada medida. "
            "A soma dentro do prange vira automaticamente uma soma em paralelo."):
            self.play(FadeIn(topo), FadeIn(cod))
        self.limpa()

        topo = self.topo(f"Mesmo problema, 1 a {PMAX} threads")
        sel = [p for p in PS if p in (1, 2, 4, 8, PMAX)]
        bar = self.barras([(f"{p} thread" + ("s" if p > 1 else ""), TH[p], CPU_C if p > 1 else MUTED) for p in sel],
                          fmt=lambda v: f"{num(v * 1000, 0)} ms").next_to(topo, DOWN, buff=0.6)
        fonte = self.fonte(f"medido num notebook com {E['nucleos']} núcleos lógicos")
        sp = TH[1] / TH[PMAX]
        with self.voiceover(
            f"Com uma thread, {num(TH[1] * 1000, 0)} milissegundos. Com {PMAX}, {num(TH[PMAX] * 1000, 0)}. "
            f"Um speedup de {num(sp)} vezes, com eficiência de {sp / PMAX:.0%}."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar, run_time=3)
        self.limpa()

        topo = self.topo("Por que não 12×? A memória.")
        a = self.texto("cada mola: 3 multiplicações, mas 16 bytes lidos da memória", 26)
        b = self.texto(f"com {PMAX} threads: {E['banda_gbs']:.0f} GB/s, perto do limite da RAM", 26, MEM_C)
        c = self.texto("as threads ficam ESPERANDO dados, não calculando", 26, BAD_C)
        d = self.texto(f"mesmo assim: {vezes(E['python_s'] / TH[PMAX])} mais rápido que o Python puro", 28, OK_C)
        VGroup(a, b, c, d).arrange(DOWN, buff=0.45).next_to(topo, DOWN, buff=0.7)
        with self.voiceover(
            "Por que não doze vezes? Cada mola faz só três multiplicações, mas precisa ler dezesseis bytes da memória. "
            f"<bookmark mark='b'/>Com todas as threads, o programa lê {E['banda_gbs']:.0f} gigabytes por segundo, perto "
            "do limite da memória do notebook. <bookmark mark='c'/>As threads passam a maior parte do tempo esperando "
            "dados. <bookmark mark='d'/>Essa é exatamente a análise que vai no relatório. E, mesmo assim, o resultado "
            f"é {E['python_s'] / TH[PMAX]:.0f} vezes mais rápido que o Python puro."):
            self.play(FadeIn(topo), FadeIn(a))
            self.wait_until_bookmark("b"); self.play(FadeIn(b))
            self.wait_until_bookmark("c"); self.play(FadeIn(c))
            self.wait_until_bookmark("d"); self.play(FadeIn(d))
        self.wait(1)


# ------------------------------------------------------------------ 13
class V13Escada(CursoScene):
    def construct(self):
        t = self.titulo("O mesmo exemplo na GPU: a escada de speedup", "CuPy")
        with self.voiceover("E se você escolher o caminho da GPU? Lá não há número de threads para variar."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Na GPU: troque np por cp")
        cod = self.codigo('''
import cupy as cp
kg, dg = cp.asarray(k), cp.asarray(d)       # leve: RAM → GPU (uma vez)
u = float(0.5 * cp.dot(kg, dg * dg))        # calcule e traga o número
''', largura=11.5).next_to(topo, DOWN, buff=0.7)
        with self.voiceover(
            "Com CuPy, o código é o do NumPy com c p no lugar de n p: leve os dados para a GPU, calcule, traga o "
            "resultado. Como não dá para variar as threads, comparamos o mesmo problema em cada versão: uma escada."):
            self.play(FadeIn(topo), FadeIn(cod))
        self.limpa()

        degraus = [("Python puro", E["python_s"], MUTED), ("NumPy", E["numpy_s"], MUTED),
                   ("Numba, 1 thread", TH[1], CPU_C), (f"Numba, {PMAX} threads", TH[PMAX], CPU_C)]
        if "cupy_s" in E:
            degraus += [("CuPy, dado na GPU", E["cupy_s"], GPU_C), ("CuPy + viagem", E["cupy_copia_s"], BAD_C)]
        topo = self.topo("A escada: vezes mais rápido que o Python puro")
        bar = self.barras([(n, E["python_s"] / v, c) for n, v, c in degraus], fmt=vezes, log=True).next_to(topo, DOWN, buff=0.5)
        fonte = self.fonte(f"{E['n'] // 10**6} milhões de molas · {E.get('gpu', 'sem GPU')} · CPU com {E['nucleos']} núcleos")
        with self.voiceover(
            "Do Python puro ao NumPy, depois Numba com uma thread e com todas. <bookmark mark='g'/>Com o dado já na GPU, "
            "o CuPy fica no mesmo patamar da CPU. <bookmark mark='v'/>Mas contando a viagem da RAM até a GPU, ele cai "
            "para o fim da escada."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar[:4], run_time=2.5)
            if len(bar) > 4:
                self.wait_until_bookmark("g"); self.mostra_barras(bar[4:5], run_time=1)
                self.wait_until_bookmark("v"); self.mostra_barras(bar[5:], run_time=1)
        self.limpa()

        topo = self.topo("A análise")
        a = self.texto("copiar 800 MB custa ~10× a própria conta", 28, BAD_C)
        b = self.texto("uma conta leve, feita uma vez: a GPU não compensa", 28)
        c = self.texto("dados que ficam na GPU por muitos passos: aí sim", 28, OK_C)
        d = self.texto("concluir isso com números é um ótimo relatório", 28, MEM_C)
        VGroup(a, b, c, d).arrange(DOWN, buff=0.45).next_to(topo, DOWN, buff=0.8)
        with self.voiceover(
            "A análise: copiar oitocentos megabytes custa umas dez vezes a própria conta. Para uma conta leve, feita uma "
            "vez só, a GPU não compensa. Ela compensa quando os dados ficam lá por muitos passos. Chegar a essa "
            "conclusão, com os seus números, é exatamente o que se espera do relatório."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(x) for x in (a, b, c, d)], lag_ratio=0.8), run_time=5)
        self.wait(1)


# ------------------------------------------------------------------ 14
class V14IA(CursoScene):
    def construct(self):
        t = self.titulo("Usando IA sem se perder", "prompt, conferir, medir")
        with self.voiceover("Pode usar inteligência artificial nos exercícios? Pode, e deve. Mas com cuidado."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Um prompt que funciona")
        prompt = self.caixa("“Tenho esta função em Python [cole a serial].\nReescreva com Numba usando @njit(parallel=True)\n"
                            "e prange, sem mudar o resultado.\nExplique cada linha que mudou.”", MEM_C, w=11.5, h=2.6, fs=26)
        prompt.next_to(topo, DOWN, buff=0.7)
        dica = self.texto("para GPU: “…com CuPy, mantendo os dados na GPU; mostre como medir”", 24, GPU_C).next_to(prompt, DOWN, buff=0.5)
        with self.voiceover(
            "Cole a função serial do exercício e peça a versão com Numba, sem mudar o resultado, explicando cada linha "
            "que mudou. <bookmark mark='g'/>Para o caminho da GPU, peça a versão com CuPy, mantendo os dados na GPU, e "
            "peça para mostrar como medir."):
            self.play(FadeIn(topo), FadeIn(prompt))
            self.wait_until_bookmark("g")
            self.play(FadeIn(dica))
        self.limpa()

        topo = self.topo("A IA erra muito em desempenho. Confira sempre:")
        checks = VGroup(*[self.caixa(s, c, w=11.5, h=0.8, fs=25) for s, c in [
            ("1. O resultado bate com a função serial?  (assert np.isclose)", OK_C),
            ("2. Aqueceu antes de medir?  (1ª chamada compila)", CPU_C),
            ("3. Sincronizou a GPU antes de parar o relógio?", GPU_C),
            ("4. A viagem RAM → GPU está ou não no tempo?", MEM_C)]]).arrange(DOWN, buff=0.28)
        checks.next_to(topo, DOWN, buff=0.6)
        with self.voiceover(
            "Antes de acreditar, confira quatro coisas. <bookmark mark='a'/>O resultado bate com a função serial? "
            "<bookmark mark='b'/>Houve aquecimento antes de medir? <bookmark mark='c'/>Na GPU, houve sincronização? "
            "<bookmark mark='d'/>E a viagem da RAM até a GPU está ou não dentro do tempo medido?"):
            self.play(FadeIn(topo))
            for k, c in zip("abcd", checks):
                self.wait_until_bookmark(k)
                self.play(FadeIn(c, shift=RIGHT * 0.2))
        self.limpa()

        topo = self.topo("No relatório: diga o que a IA fez")
        ex = self.texto("“A versão Numba foi sugerida pelo ChatGPT;\neu corrigi a soma e conferi com a serial.”", 30, MEM_C)
        ex.next_to(topo, DOWN, buff=1.0)
        fim = self.texto("A IA escreve. Você confere, mede e explica.", 34, OK_C).next_to(ex, DOWN, buff=1.0)
        with self.voiceover(
            "No relatório, diga em uma linha o que a IA fez e o que você corrigiu. <bookmark mark='f'/>A IA escreve. "
            "Quem confere, mede e explica é você."):
            self.play(FadeIn(topo), FadeIn(ex))
            self.wait_until_bookmark("f")
            self.play(FadeIn(fim))
        self.wait(1)
