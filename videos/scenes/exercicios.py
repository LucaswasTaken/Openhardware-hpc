"""Vídeos dos exercícios: 11 relatório · 12 energia de deformação (Numba) · 13 probabilidade de falha (CuPy) ·
14 busca de nós (joblib: quando paralelizar não ajuda). Os exemplos são PARECIDOS com a lista, não iguais."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import (DOWN, LEFT, RIGHT, UP, Axes, Create, Dot, FadeIn, FadeOut, Indicate, LaggedStart, Rectangle,
                   Square, VGroup)

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe

DADOS = json.loads((Path(__file__).resolve().parents[1] / "data" / "exemplos.json").read_text(encoding="utf-8"))


def vezes(x):
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


# ------------------------------------------------------------------ 11
class V11Relatorio(CursoScene):
    def construct(self):
        t = self.titulo("Como fazer o relatório", "speedup, eficiência e gráficos")
        with self.voiceover("Você escolheu um exercício e uma tecnologia. Como mostrar que acelerou de verdade?"):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Quatro passos")
        passos = VGroup(*[self.caixa(s, c, w=10.5, h=0.9, fs=26) for s, c in [
            ("1. Correção: compare com uma referência (NumPy ou solução analítica)", OK_C),
            ("2. Medida honesta: aquecer, repetir, sincronizar a GPU", CPU_C),
            ("3. Speedup e eficiência, com escalabilidade forte e fraca", MEM_C),
            ("4. Análise: onde está o gargalo?", GPU_C)]]).arrange(DOWN, buff=0.3).next_to(topo, DOWN, buff=0.5)
        with self.voiceover(
            "São quatro passos. <bookmark mark='a'/>Primeiro, correção: um código rápido e errado não vale nada; compare "
            "sempre com uma referência. <bookmark mark='b'/>Segundo, medida honesta: aqueça antes, repita e fique com o "
            "menor tempo, e na GPU, sincronize. <bookmark mark='c'/>Terceiro, speedup e eficiência, variando o número de "
            "núcleos ou o tamanho do problema. <bookmark mark='d'/>Quarto, a análise: o que limitou o ganho?"):
            self.play(FadeIn(topo))
            for k, p in zip("abcd", passos):
                self.wait_until_bookmark(k)
                self.play(FadeIn(p, shift=RIGHT * 0.2))
        self.limpa()

        topo = self.topo("Os gráficos que todo relatório precisa")
        ps = [1, 2, 4, 8]
        sp = [1, 1.9, 3.4, 5.2]
        e1 = Axes(x_range=[0, 9, 2], y_range=[0, 9, 2], x_length=5, y_length=3.6, axis_config={"color": MUTED})
        e2 = Axes(x_range=[0, 9, 2], y_range=[0, 1.2, 0.5], x_length=5, y_length=3.6, axis_config={"color": MUTED})
        VGroup(e1, e2).arrange(RIGHT, buff=1.2).next_to(topo, DOWN, buff=0.7)
        ideal = e1.plot(lambda x: x, x_range=[1, 8], color=MUTED)
        med = VGroup(*[Dot(e1.c2p(p, s), color=OK_C) for p, s in zip(ps, sp)])
        ef = VGroup(*[Dot(e2.c2p(p, s / p), color=MEM_C) for p, s in zip(ps, sp)])
        t1 = self.texto("speedup × núcleos (com a reta ideal)", 22).next_to(e1, DOWN, buff=0.2)
        t2 = self.texto("eficiência × núcleos", 22).next_to(e2, DOWN, buff=0.2)
        with self.voiceover(
            "Dois gráficos são obrigatórios: o speedup em função do número de núcleos, com a reta ideal para comparar, "
            "<bookmark mark='e'/>e a eficiência, que mostra quanto do hardware você aproveitou. O modelo de relatório "
            "do curso já desenha os dois: basta trocar a função de exemplo pela sua."):
            self.play(FadeIn(topo), Create(e1), Create(ideal), FadeIn(t1))
            self.play(LaggedStart(*[FadeIn(d) for d in med], lag_ratio=0.3))
            self.wait_until_bookmark("e")
            self.play(Create(e2), FadeIn(t2), LaggedStart(*[FadeIn(d) for d in ef], lag_ratio=0.3))
        self.limpa()

        topo = self.topo("Perguntas que a análise deve responder")
        qs = VGroup(self.texto("• É limitado pelo cálculo ou pela memória? (calcule os GB/s)", 26),
                    self.texto("• Quanto custou criar processos ou copiar para a GPU?", 26),
                    self.texto("• Qual a fração serial? (Karp–Flatt)", 26),
                    self.texto("• O resultado mudaria em outra máquina? Por quê?", 26)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        qs.next_to(topo, DOWN, buff=0.7)
        with self.voiceover(
            "Na análise, responda: o problema é limitado pelo cálculo ou pela memória? Quanto custou criar processos ou "
            "copiar dados para a GPU? Qual a fração serial? E o resultado mudaria em outra máquina? Os próximos três "
            "vídeos mostram exemplos parecidos com os da lista."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(q, shift=RIGHT * 0.2) for q in qs], lag_ratio=0.5), run_time=4)
        self.wait(1)


# ------------------------------------------------------------------ 12
class V12Energia(CursoScene):
    def construct(self):
        d = DADOS["ex12"]
        t = self.titulo("Exemplo: energia de deformação com Numba", "parecido com o exercício 1 (produto escalar)")
        with self.voiceover("Exemplo resolvido: a energia de deformação de cinquenta milhões de molas, com Numba."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("U = ½ Σ kᵢ δᵢ²   (uma redução: muitos valores → um número)")
        cod = self.codigo('''
@njit(parallel=True)
def energia(k, d):
    u = 0.0
    for i in prange(k.size):
        u += 0.5 * k[i] * d[i] * d[i]   # o Numba reconhece a redução
    return u
''').next_to(topo, DOWN, buff=0.5)
        with self.voiceover(
            "A energia é a soma de meio k delta ao quadrado de cada mola: uma redução, muitos valores viram um número. "
            "Com prange, o Numba divide o laço entre os núcleos. <bookmark mark='r'/>E ele reconhece o mais igual como "
            "redução: cada thread soma a sua parte, e no fim as somas parciais são juntadas. Sem isso, duas threads "
            "escreveriam na mesma variável ao mesmo tempo."):
            self.play(FadeIn(topo), FadeIn(cod))
            self.wait_until_bookmark("r")
            self.play(Indicate(cod, color=MEM_C, scale_factor=1.02))
        self.limpa()

        topo = self.topo(f"{d['n']/1e6:.0f} milhões de molas")
        bar = self.barras([("Python puro", 1.0, MUTED), ("NumPy (np.dot)", d["python_s"] / d["numpy_s"], CPU_C),
                           ("Numba, 1 núcleo", d["python_s"] / d["numba_s"], MEM_C),
                           (f"Numba paralelo ({d['nucleos']} threads)", d["python_s"] / d["numba_par_s"], OK_C)],
                          fmt=vezes, log=True).next_to(topo, DOWN, buff=0.8)
        fonte = self.fonte(f"vezes mais rápido que Python puro (escala log.) · notebook com {d['nucleos']} núcleos lógicos")
        with self.voiceover(
            f"O resultado: Numba paralelo ficou {d['python_s']/d['numba_par_s']:.0f} vezes mais rápido que Python puro "
            f"e {d['numpy_s']/d['numba_par_s']:.0f} vezes mais rápido que o NumPy."):
            self.play(FadeIn(topo), FadeIn(fonte))
            self.mostra_barras(bar)
        an = VGroup(self.texto(f"leitura de memória ≈ {d['banda_gbs']:.0f} GB/s · 3 contas a cada 16 bytes lidos", 26, MEM_C),
                    self.texto("→ quem limita é a memória: compare com a banda da sua máquina", 24))
        an.arrange(DOWN, buff=0.2).next_to(bar, DOWN, buff=0.6)
        with self.voiceover(
            f"E a análise: lendo dois vetores, chegamos a cerca de {d['banda_gbs']:.0f} gigabytes por segundo. São só "
            "três contas para cada dezesseis bytes lidos, então quem limita é a memória, não o cálculo. No seu "
            "relatório, compare esse número com a largura de banda da sua máquina."):
            self.play(FadeIn(an))
        self.wait(1)


# ------------------------------------------------------------------ 13
class V13Falha(CursoScene):
    def construct(self):
        d = DADOS["ex13"]
        t = self.titulo("Exemplo: probabilidade de falha com CuPy", "parecido com o exercício 3 (Monte Carlo para π)")
        with self.voiceover("Exemplo resolvido: a probabilidade de falha de uma viga, sorteando milhões de casos na GPU."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo("Gerar e testar as amostras direto na GPU")
        cod = self.codigo('''
rng = cp.random.default_rng(seed)
R = cp.exp(cp.log(350) + 0.10 * rng.standard_normal(n))   # resistência
S = 200 - 25 * cp.log(-cp.log(rng.random(n)))             # solicitação
pf = int(cp.count_nonzero(R < S)) / n                      # só 1 número volta
''', largura=12).next_to(topo, DOWN, buff=0.5)
        with self.voiceover(
            "As amostras são geradas e testadas na própria GPU. Nada é copiado para lá, e só um número volta: a "
            "probabilidade de falha. É o melhor caso para a GPU: muita conta e quase nenhuma transferência."):
            self.play(FadeIn(topo), FadeIn(cod))
        self.limpa()

        if d.get("gpu"):
            n = "50000000"
            topo = self.topo("50 milhões de amostras")
            bar = self.barras([("CPU (NumPy)", d["cpu"][n], CPU_C), ("GPU (CuPy)", d["gpu"][n], OK_C)],
                              fmt=lambda v: f"{v*1e3:.0f} ms").next_to(topo, DOWN, buff=0.9)
            fonte = self.fonte(f"medido · {d.get('gpu_nome', 'GPU')}")
            with self.voiceover(
                f"Com cinquenta milhões de amostras, a GPU foi {d['cpu'][n]/d['gpu'][n]:.0f} vezes mais rápida que a "
                "CPU, mesmo sendo uma GPU de entrada."):
                self.play(FadeIn(topo), FadeIn(fonte))
                self.mostra_barras(bar)
            self.limpa()

        conv = d.get("convergencia", [])
        if conv:
            topo = self.topo("Quantas amostras bastam? O erro cai com 1/√N")
            # eixo y = log10(erro) + 3, para o eixo x ficar embaixo: 0 → 0,1 %, 1 → 1 %, 2 → 10 %, 3 → 100 %
            ax = Axes(x_range=[3, 7, 1], y_range=[0, 3, 1], x_length=7.5, y_length=4,
                      axis_config={"color": MUTED, "include_ticks": False}).next_to(topo, DOWN, buff=0.6)
            xs = VGroup(*[self.texto(r, 20, MUTED).next_to(ax.c2p(x, 0), DOWN, buff=0.15)
                          for x, r in ((3, "mil"), (5, "100 mil"), (7, "10 milhões"))])
            ys = VGroup(*[self.texto(r, 20, MUTED).next_to(ax.c2p(3, y), LEFT, buff=0.15)
                          for y, r in ((0, "0,1%"), (1, "1%"), (2, "10%"), (3, "100%"))])
            xl = self.texto("amostras N (escala log.)", 20, MUTED).next_to(xs, DOWN, buff=0.1)
            yl = self.texto("erro relativo", 20, MUTED).rotate(np.pi / 2).next_to(ys, LEFT, buff=0.15)
            xl = VGroup(xl, xs); yl = VGroup(yl, ys)
            pts = VGroup(*[Dot(ax.c2p(np.log10(c["n"]), np.log10(c["erro_medio"]) + 3), color=OK_C, radius=0.09)
                           for c in conv])
            c0 = conv[0]
            ref = ax.plot(lambda x: np.log10(c0["erro_medio"]) + 3 - 0.5 * (x - np.log10(c0["n"])), x_range=[3, 7],
                          color=MEM_C)
            lr = self.texto("inclinação −½  (1/√N)", 22, MEM_C).next_to(ax.c2p(5.3, 2.1), RIGHT)
            pf = d.get("pf", 0.0055)
            with self.voiceover(
                f"A probabilidade de falha deu cerca de {pf*1000:.1f} em mil. E quantas amostras bastam? O erro de Monte "
                "Carlo cai com um sobre a raiz de N: para dividir o erro por dez, são cem vezes mais amostras. "
                "<bookmark mark='s'/>Um cuidado: cada processo ou bloco precisa de uma semente diferente, senão você "
                "repete as mesmas amostras e acha que tem mais dados do que tem."):
                self.play(FadeIn(topo), Create(ax), FadeIn(xl), FadeIn(yl))
                self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.3), Create(ref), FadeIn(lr))
                self.wait_until_bookmark("s")
        self.wait(1)


# ------------------------------------------------------------------ 14
class V14Busca(CursoScene):
    def construct(self):
        d = DADOS["ex14"]
        proc, thr = d["joblib_s"]["processos"], d["joblib_s"]["threads"]
        base = d["numpy_s"]
        t = self.titulo("Exemplo: quando paralelizar não ajuda", "parecido com o exercício 5 (busca em array)")
        with self.voiceover("Exemplo resolvido, e com uma surpresa: às vezes paralelizar não ajuda nada."):
            self.play(FadeIn(t))
        self.play(FadeOut(t))

        topo = self.topo(f"Quais dos {d['n']/1e6:.0f} milhões de nós estão dentro de uma região?")
        rng = np.random.default_rng(1)
        pts = VGroup(*[Dot([x, y, 0], radius=0.035,
                           color=OK_C if (x - 0.5) ** 2 + (y + 0.6) ** 2 < 1.2 else MUTED)
                       for x, y in zip(rng.uniform(-5, 5, 500), rng.uniform(-2.6, 1.6, 500))])
        with self.voiceover(
            f"A tarefa: de {d['n']/1e6:.0f} milhões de nós de uma malha, achar quais estão dentro de uma região "
            "circular. O NumPy faz isso numa linha. Vamos dividir o array em blocos e usar o joblib com vários "
            "processos."):
            self.play(FadeIn(topo), LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.002), run_time=2.5)
        self.limpa()

        topo = self.topo("joblib com PROCESSOS")
        ks = list(proc.keys())
        bar = self.barras([("NumPy, 1 núcleo", 1.0, MUTED)] + [(f"{k} processo{'s' if k != '1' else ''}", base / proc[k], BAD_C) for k in ks],
                          fmt=lambda v: f"{v:.2f}×").next_to(topo, DOWN, buff=0.8)
        with self.voiceover(
            "Com processos: nenhum ganho. Mais processos, e o tempo não cai. <bookmark mark='p'/>Por quê? A conta por nó "
            "é minúscula, e cada bloco do array precisa ser copiado para outro processo. Copiar seiscentos megabytes "
            "custa mais do que a própria busca."):
            self.play(FadeIn(topo))
            self.mostra_barras(bar)
            self.wait_until_bookmark("p")
        self.limpa()

        topo = self.topo("A correção: threads compartilham a memória")
        cod = self.codigo('''
Parallel(n_jobs=-1, prefer="threads")(
    delayed(busca)(xy[a:b], a) for a, b in blocos)
''').next_to(topo, DOWN, buff=0.5)
        bar2 = self.barras([("NumPy, 1 núcleo", 1.0, MUTED)] + [(f"{k} thread{'s' if k != '1' else ''}", base / thr[k], OK_C) for k in thr],
                           fmt=lambda v: f"{v:.1f}×").next_to(cod, DOWN, buff=0.6)
        with self.voiceover(
            "A correção é uma palavra: prefer igual a threads. Threads enxergam a mesma memória, então nada é copiado. "
            "E como o NumPy solta o GIL durante as contas, as threads trabalham de verdade em paralelo. "
            f"<bookmark mark='r'/>Resultado: {base/min(thr.values()):.1f} vezes mais rápido."):
            self.play(FadeIn(topo), FadeIn(cod))
            self.wait_until_bookmark("r")
            self.mostra_barras(bar2)
        fecho = self.texto("Pouca conta por byte → evite copiar dados. Meça antes de concluir.", 26, MEM_C)
        fecho.to_edge(DOWN, buff=0.4)
        with self.voiceover(
            "A lição: quando há pouca conta por byte, o custo de mover os dados decide tudo. Isso vale para processos "
            "e para a GPU. No seu relatório, mostre esse tipo de análise."):
            self.play(FadeIn(fecho))
        self.wait(1)
