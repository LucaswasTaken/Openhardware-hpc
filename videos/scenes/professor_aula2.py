"""Vídeo-roteiro da Aula 2 — só para o professor (não vai para o site).

Explica, bloco a bloco, o que abrir, o que mostrar, o que rodar, o que dizer, os números esperados e onde a turma
trava. Cada capítulo é uma cena; videos/professor.py renderiza todas e junta num MP4 só:

    python videos/professor.py            → videos/professor/aula2_professor.mp4

Mantenha alinhado com ROTEIRO_DO_PROFESSOR.md (parte da Aula 2)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import DOWN, LEFT, RIGHT, UP, FadeIn, LaggedStart, Rectangle, VGroup

from common import BAD_C, CPU_C, DATA_C, GPU_C, MEM_C, MUTED, OK_C, CursoScene, cabe, results

R = results(2)
E = json.loads((Path(__file__).resolve().parents[1] / "data" / "exemplos.json").read_text(encoding="utf-8"))["energia"]
GPU_NOME = R.get("gpu", {}).get("nome", "GPU")
CC = R.get("copia_calculo", {"ida_ms": 91, "leve_ms": 18, "pesada_ms": 117, "volta_ms": 127})
VM = R.get("von_mises", {"cpu_ms": 498, "gpu_ms": 40})
MM = R.get("matmul", {}).get("tempos_s", {"CPU float64": 0.96, "GPU float64": 2.04, "GPU float32": 0.076})
ES = R.get("escada", {}).get("tempos_s", {})


def n(x, casas=0):
    return f"{x:.{casas}f}".replace(".", ",")


BLOCOS = [("0:00", "1. Por que GPU?"), ("0:20", "2. Dois mundos"), ("0:50", "3. CuPy em 3 passos"),
          ("1:30", "☕"), ("1:45", "4. Laje em incêndio"), ("2:45", "5. Seu exercício"), ("3:15", "6. Quando usar")]


class Capitulo(CursoScene):
    """Cena genérica: QUADROS = [(tipo, dados, narração)], um quadro por vez."""
    TITULO, SUB, QUADROS = "", "", []

    def linha_tempo(self, atual):
        caixas = VGroup()
        for i, (h, nome) in enumerate(BLOCOS):
            w = 1.0 if nome == "☕" else 1.85
            r = Rectangle(width=w, height=0.75, color=MEM_C if i == atual else MUTED,
                          fill_opacity=0.35 if i == atual else 0.08, stroke_width=3 if i == atual else 1)
            t = VGroup(self.texto(h, 16, MUTED), cabe(self.texto(nome, 17, MEM_C if i == atual else MUTED), w - 0.1))
            caixas.add(VGroup(r, t.arrange(DOWN, buff=0.05).move_to(r)))
        return cabe(caixas.arrange(RIGHT, buff=0.06))

    def visual(self, tipo, d):
        if tipo == "tempo":
            return VGroup(self.linha_tempo(d[0]), self.texto(d[1], 30, MEM_C)).arrange(DOWN, buff=0.8)
        if tipo == "itens":
            titulo, itens = d
            corpo = VGroup(*[self.texto(i, 26) for i in itens]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            return VGroup(self.texto(titulo, 30, MEM_C), corpo).arrange(DOWN, buff=0.5)
        if tipo == "arvore":
            titulo, itens = d
            return VGroup(self.texto(titulo, 28, MEM_C), self.arvore(itens)).arrange(DOWN, buff=0.4)
        if tipo == "slide":
            return self.slide_mini(*d)
        if tipo == "celula":
            rot, num, cod = d
            return VGroup(self.texto(rot, 26, MEM_C), self.celula(num, cod)).arrange(DOWN, buff=0.35)
        if tipo == "diga":
            return VGroup(self.texto("DIGA:", 24, MUTED), self.texto(f"“{d}”", 34, OK_C)).arrange(DOWN, buff=0.4)
        if tipo == "numeros":
            titulo, itens, unidade = d
            b = self.barras(itens, fmt=lambda v: f"{n(v, 1 if v < 10 else 0)} {unidade}", log=(unidade == "×"))
            return VGroup(self.texto(titulo, 28, MEM_C), b).arrange(DOWN, buff=0.5)
        if tipo == "trava":
            itens = VGroup(*[self.texto("⚠ " + i, 26, BAD_C) for i in d]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            return VGroup(self.texto("Onde a turma trava", 30, MEM_C), itens).arrange(DOWN, buff=0.5)
        raise ValueError(tipo)

    def construct(self):
        t = self.titulo(self.TITULO, self.SUB)
        with self.voiceover(self.ABERTURA):
            self.play(FadeIn(t))
        self.limpa()
        for tipo, d, fala in self.QUADROS:
            v = self.visual(tipo, d)
            cabe(v)
            if v.height > 6.8:
                v.scale_to_fit_height(6.8)
            v.move_to([0, 0, 0])
            with self.voiceover(fala):
                if tipo in ("itens", "trava") and len(v[1]) > 1:
                    self.play(FadeIn(v[0]))
                    self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.15) for i in v[1]], lag_ratio=0.8), run_time=2.5)
                elif tipo == "numeros":
                    self.play(FadeIn(v[0]))
                    self.mostra_barras(v[1])
                else:
                    self.play(FadeIn(v))
            self.limpa()


# =================================================================== 0. abertura
class P00Abertura(Capitulo):
    TITULO, SUB = "Aula 2 — roteiro do professor", "GPU na prática · 3h30 · só para você"
    ABERTURA = ("Este vídeo é só para você, professor. Ele percorre a Aula 2 inteira: o que abrir, o que mostrar, o que "
                "rodar, o que dizer, e onde a turma costuma travar.")
    QUADROS = [
        ("itens", ("A mensagem da aula, em uma frase", [
            "A GPU são milhares de operários fazendo a MESMA conta.",
            "Ela só ganha com MUITOS dados…",
            "…e quando os dados FICAM lá.",
            "Ferramenta: uma só — CuPy (troque np por cp)."]),
         "A aula inteira gira em torno de uma frase. A GPU são milhares de operários fazendo a mesma conta. Ela só ganha "
         "quando há muitos dados, e quando os dados ficam lá. E a ferramenta é uma só: o CuPy, que é o NumPy da GPU. "
         "Kernels, memória gerenciada e outras bibliotecas ficam no apêndice ou só são citados."),
        ("itens", ("Para quem é a aula", [
            "engenheiros que NÃO programam",
            "vão pedir ajuda à IA para os exercícios",
            "não ensine sintaxe: ensine O QUE OBSERVAR",
            "cada célula tem “O que faz / O que observar”"]),
         "Lembre do público: engenheiros que não programam, e que vão usar inteligência artificial nos exercícios. "
         "Então não ensine sintaxe. Ensine o que observar em cada número. Cada célula do notebook já vem com uma caixa "
         "dizendo o que ela faz e o que observar. A sua tarefa é ler essa caixa em voz alta, pedir um palpite, e rodar."),
        ("tempo", (-1, "3h30 em 6 blocos + intervalo"),
         "São seis blocos. Por que GPU, vinte minutos. Dois mundos, a memória do computador e a da GPU, meia hora. CuPy "
         "em três passos, quarenta minutos. Intervalo. O projeto da laje em incêndio, uma hora. O exercício de cada um "
         "na GPU, meia hora. E quando usar GPU, quinze minutos no fim."),
    ]


# =================================================================== 1. preparação
class P01Preparacao(Capitulo):
    TITULO, SUB = "Antes da aula", "o que abrir, e onde está cada coisa"
    ABERTURA = "Vamos começar pelo que abrir antes da aula, e onde está cada arquivo no repositório."
    QUADROS = [
        ("arvore", ("Repositório Openhardware-hpc", [
            (0, "notebooks/", False), (1, "Aula2_GPU.ipynb", True), (1, "results/aula2_local.json", False),
            (0, "presentations/", False), (1, "Aula2_GPU.md   (slides, Marp)", True),
            (0, "exercicios/", False), (1, "lista.md  ·  lista.docx", True), (1, "modelo_relatorio.ipynb", True),
            (1, "INSTALACAO.md", True),
            (0, "docs/   (página dos vídeos)", False), (0, "ROTEIRO_DO_PROFESSOR.md", True)]),
         "No repositório, você vai usar cinco arquivos. O notebook da aula, em notebooks, Aula dois GPU. Os slides, em "
         "presentations, Aula dois GPU ponto md, no formato Marp. Na pasta exercicios: a lista, em markdown e em Word, "
         "o modelo de relatório, e o guia de instalação. E o roteiro do professor, na raiz, que é a versão escrita "
         "deste vídeo."),
        ("itens", ("Abas abertas no projetor", [
            "1. Notebook no Colab (botão “Aula 2” do README)",
            "2. Slides (Marp, modo apresentação)",
            "3. Página dos vídeos: lucaswastaken.github.io/Openhardware-hpc",
            "4. exercicios/lista.md no GitHub"]),
         "No projetor, deixe quatro abas abertas: o notebook no Colab, pelo botão do README; os slides em modo "
         "apresentação; a página dos vídeos; e a lista de exercícios no GitHub."),
        ("itens", ("No Colab, antes de todos chegarem", [
            "Ambiente de execução → Alterar o tipo → T4 GPU",
            "Rode a 1ª célula: deve dizer  GPU: True",
            "Peça à turma o mesmo LOGO na chegada (fila de GPU)",
            "Plano B: o notebook publicado já vem executado"]),
         "No Colab, troque o ambiente para T4 GPU e rode a primeira célula. Ela tem que dizer GPU true. Peça à turma "
         "para fazer o mesmo logo na chegada, porque a fila de GPU do Colab pode demorar. Se alguém não conseguir, "
         "acompanha em dupla. E o plano B: o notebook publicado no GitHub já vem executado, com todas as saídas."),
        ("celula", ("1ª célula: preparação", 1, '''
try:
    import cupy as cp
    GPU = cp.cuda.runtime.getDeviceCount() > 0
...
def cronometro(f, *args, repeticoes=5):   # aquece, sincroniza, menor tempo
...
print(f"... · GPU: {GPU}")'''),
         "A primeira célula importa o CuPy, descobre se há GPU, e cria a função cronômetro, que aquece, sincroniza e "
         "devolve o menor tempo. Não precisa explicar o código dela. Só diga: essa função mede do jeito certo, e "
         "vamos entender por que no bloco três."),
    ]


# =================================================================== 2. bloco 1
class P02Bloco1(Capitulo):
    TITULO, SUB = "Bloco 1 · Por que GPU?", "0:00–0:20 · slides 1–3 · vídeo 07 · seção 1"
    ABERTURA = "Bloco um, vinte minutos: por que GPU."
    QUADROS = [
        ("tempo", (0, "0:00–0:20"), "Estamos no começo da aula."),
        ("slide", (3, "1. Por que GPU? Milhares de operários",
                   ["CPU: 8 engenheiros experientes", "GPU: 2 500 operários, a MESMA tarefa ao mesmo tempo",
                    "a pergunta-chave: muitos dados, mesma conta?"]),
         "Abra com o vídeo sete, de um minuto, ou conte a analogia do slide três: a CPU são oito engenheiros "
         "experientes, que resolvem qualquer coisa; a GPU são dois mil e quinhentos operários, que fazem a mesma tarefa "
         "ao mesmo tempo."),
        ("diga", "Quais contas do trabalho de vocês são a MESMA conta em milhões de dados?",
         "Faça esta pergunta à turma e anote duas ou três respostas no quadro. Costumam aparecer tensões em pontos de "
         "integração, nós de malha, amostras de Monte Carlo. Você vai voltar a elas no último bloco."),
        ("celula", ("Seção 1: quem é a GPU", 2, '''
p = cp.cuda.runtime.getDeviceProperties(0)
print(f"{nucleos:,} núcleos  (a CPU tem {os.cpu_count()})")
print(f"{memoria:.1f} GB de memória PRÓPRIA da GPU")'''),
         "Rode a célula da seção um. No Colab, aparece Tesla T4, dois mil quinhentos e sessenta núcleos e uns quinze "
         "gigabytes, contra dois núcleos de CPU. Destaque a expressão memória própria: é o gancho do bloco dois."),
        ("diga", "Então a GPU é mil vezes mais rápida? Não: cada operário é lento, e precisa de material.",
         "Termine com uma pergunta-armadilha: então a GPU é mil vezes mais rápida? Não. Cada operário é lento, e "
         "todos precisam de material, que são os dados. E o material está longe. É a ponte para o bloco dois."),
    ]


# =================================================================== 3. bloco 2
class P03Bloco2(Capitulo):
    TITULO, SUB = "Bloco 2 · Dois mundos", "0:20–0:50 · slides 4–6 · vídeos 08 e 08b · seção 2"
    ABERTURA = "Bloco dois, meia hora: dois mundos, a memória do computador e a da GPU. É o fio condutor da aula."
    QUADROS = [
        ("tempo", (1, "0:20–0:50"), "Vinte minutos de aula."),
        ("slide", (4, "2. Dois mundos: a viagem dos dados",
                   ["RAM ←— cabo PCIe —→ VRAM", "a GPU só calcula o que está na VRAM", "analogia: depósito do outro lado da cidade"]),
         "Mostre o diagrama do slide quatro. Na placa separada, que é o caso do Colab, a GPU tem memória própria, e os "
         "dados viajam pelo cabo PCIe. A analogia: os operários são rápidos, mas o material está num depósito do outro "
         "lado da cidade, e o caminhão demora."),
        ("celula", ("Seção 2: viagem × conta (peça palpites ANTES)", 3, '''
x_gpu = cp.asarray(x_ram)                  # 400 MB: RAM → GPU
t_ida   = cronometro(lambda: cp.asarray(x_ram))
t_leve  = cronometro(lambda: 2.0 * x_gpu + 1.0)
t_volta = cronometro(lambda: cp.asnumpy(x_gpu))'''),
         "Antes de rodar a célula da viagem, pergunte: o que demora mais, levar quatrocentos megabytes até a GPU, ou "
         "multiplicar cinquenta milhões de números? Deixe votarem. Depois rode."),
        ("numeros", (f"Medido no notebook de teste ({GPU_NOME})",
                     [("viagem RAM → GPU", CC["ida_ms"], BAD_C), ("conta leve", CC["leve_ms"], OK_C),
                      ("conta pesada", CC["pesada_ms"], CPU_C), ("viagem GPU → RAM", CC["volta_ms"], BAD_C)], "ms"),
         f"No notebook de teste, com uma GPU de notebook, a ida levou {n(CC['ida_ms'])} milissegundos, e a conta leve "
         f"{n(CC['leve_ms'])}. No Colab, com a T4, espere algo como trinta a quarenta milissegundos de viagem e dois a "
         "quatro de conta leve. Os números mudam; a lição não: a viagem custa várias vezes a conta."),
        ("diga", "Leve os dados UMA vez, faça TODAS as contas lá, traga só o resultado.",
         "Escreva a regra de ouro no quadro, e deixe lá até o fim da aula. Você vai voltar a ela em todos os blocos."),
        ("slide", (6, "Memória unificada: o depósito dentro da obra",
                   ["Apple M1–M5, Grace Hopper, DGX Spark, AMD MI300A", "MUDA: acaba a viagem",
                    "NÃO MUDA: muitos dados, mesma conta, memória rápida"]),
         "Agora passe o vídeo oito b, de dois minutos, ou use o slide seis. Em algumas máquinas novas, CPU e GPU "
         "compartilham a memória. Pergunte: quem tem MacBook com chip M? Esse aluno já usa memória unificada. "
         "Insista no que não muda: continua precisando de muitos dados com a mesma conta."),
        ("trava", ["“Então no Mac não preciso de nada disso?” → a GPU ainda precisa de muitos dados",
                   "confundir memória da GPU com memória do computador",
                   "Se atrasar: o 08b vai para casa; fique no slide 6"],
         "Onde a turma trava: achar que memória unificada resolve tudo, e confundir as duas memórias. Se estiver "
         "atrasado, mande o vídeo oito b para casa e fique só com o slide seis."),
    ]


# =================================================================== 4. bloco 3
class P04Bloco3(Capitulo):
    TITULO, SUB = "Bloco 3 · CuPy em 3 passos", "0:50–1:30 · slides 7–9 · vídeo 09 · seção 3"
    ABERTURA = "Bloco três, quarenta minutos: CuPy em três passos. É o coração prático da aula."
    QUADROS = [
        ("tempo", (2, "0:50–1:30"), "Cinquenta minutos de aula."),
        ("slide", (7, "3. CuPy em 3 passos",
                   ["1. LEVE: cp.asarray(x)", "2. CALCULE: troque np por cp", "3. TRAGA: cp.asnumpy(y) ou float(total)"]),
         "Escreva os três passos no quadro: leve, calcule, traga. Leve com c p ponto asarray. Calcule trocando n p por "
         "c p. Traga só o resultado, com c p ponto asnumpy, ou float, se for um número."),
        ("celula", ("Seção 3.1: von Mises — a MESMA função na CPU e na GPU", 4, '''
def von_mises(sx, sy, txy):
    xp = cp.get_array_module(sx)       # np ou cp, conforme onde estão os dados
    return xp.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)

t_cpu = cronometro(von_mises, sx, sy, txy)          # dados na RAM
t_gpu = cronometro(von_mises, sx_g, sy_g, txy_g)    # dados na GPU'''),
         "Na seção três ponto um, mostre que a função de von Mises é uma só. A linha do xp escolhe a biblioteca "
         "conforme onde estão os dados. Isso é o que os alunos vão reaproveitar no exercício."),
        ("numeros", ("von Mises, 20 milhões de pontos (notebook de teste)",
                     [("NumPy (CPU)", VM["cpu_ms"], CPU_C), ("CuPy (GPU)", VM["gpu_ms"], GPU_C)], "ms"),
         f"No notebook de teste, a GPU foi {n(VM['cpu_ms'] / VM['gpu_ms'])} vezes mais rápida, com os dados já lá. Na T4, "
         "espere de vinte a sessenta vezes. Leia o prompt de inteligência artificial que está logo abaixo da célula, e "
         "incentive a turma a usar."),
        ("celula", ("Seção 3.2: o erro nº 1 — sem sincronizar", 5, '''
print("sem sincronizar:", mede_sem_sincronizar(conta))   # irrealmente pequeno
print("sincronizando:  ", cronometro(conta))             # o tempo real'''),
         "A seção três ponto dois é curta e muito importante. Sem sincronizar, o tempo sai absurdamente pequeno, porque "
         "só mediu o pedido."),
        ("diga", "Resultado de GPU bom demais? Pergunte se sincronizou.",
         "Escreva esta frase no quadro. A inteligência artificial comete esse erro com frequência, e ele vai aparecer "
         "nos relatórios."),
        ("numeros", ("Matrizes 4096 × 4096 (notebook de teste)",
                     [(k, 2 * 4096**3 / v / 1e9, c) for (k, v), c in zip(MM.items(), (CPU_C, BAD_C, OK_C))], "GFLOP/s"),
         "Na seção três ponto três, multiplicação de matrizes, onde a GPU brilha. Repare na diferença entre float32 e "
         "float64: na GPU de teste, precisão dupla foi até mais lenta que a CPU. Na T4, espere uns quatro mil gigaflops "
         "em float32 e uns duzentos em float64. Pergunte: quais cálculos de vocês aceitam sete dígitos?"),
        ("trava", ["achar que trocar np por cp basta (e esquecer a viagem)",
                   "medir sem aquecer: a 1ª chamada é lenta",
                   "Se atrasar: pule as matrizes; o slide 9 mostra o resultado"],
         "Onde a turma trava: achar que trocar n p por c p basta, esquecendo a viagem, e medir sem aquecer. Se estiver "
         "atrasado, pule as matrizes; o slide nove mostra o resultado."),
    ]


# =================================================================== 5. bloco 4
class P05Bloco4(Capitulo):
    TITULO, SUB = "Bloco 4 · Laje em incêndio", "1:45–2:45 · slides 10–11 · vídeo 10 · seção 4"
    ABERTURA = ("Depois do intervalo, bloco quatro, uma hora: o projeto da laje em incêndio. Deixe o vídeo dez tocando "
                "no fim do intervalo.")
    QUADROS = [
        ("tempo", (4, "1:45–2:45"), "Uma hora e quarenta e cinco de aula."),
        ("slide", (10, "4. Projeto: laje em incêndio",
                   ["corte 20 × 20 cm, face de baixo a 600 °C", "cada ponto vira a média dos 4 vizinhos",
                    "simula(T): a MESMA função na CPU e na GPU"]),
         "Apresente o problema: o corte de uma laje, com a face de baixo num incêndio a seiscentos graus. Desenhe no "
         "quadro a cruz de cinco pontos: cada ponto fica um pouco mais parecido com a média dos vizinhos. Não precisa "
         "de equação diferencial."),
        ("celula", ("Seção 4: uma função para CPU e GPU", 6, '''
def simula(T, passos):
    xp = cp.get_array_module(T)          # quem decide é onde T está
    for _ in range(passos):
        B[1:-1, 1:-1] = A[1:-1, 1:-1] + R * (vizinhos - 4 * A[1:-1, 1:-1])
        A, B = B, A

simula(laje_inicial(n), 300)             # CPU
simula(laje_inicial(n, cp), 300)         # GPU'''),
         "Leia a função simula com a turma. Ela não sabe se está na CPU ou na GPU; quem decide é onde está a laje. Rode "
         "a célula: três confere."),
        ("diga", "A partir de que tamanho de malha a GPU ganha?",
         "Antes de rodar a corrida por tamanho de malha, peça palpites. A célula leva um ou dois minutos no Colab, "
         "então use esse tempo para a discussão."),
        ("itens", ("O que esperar no gráfico", [
            "Colab (2 núcleos + T4): GPU perde em 128², ganha a partir de ~512²",
            "malha grande: dados parados na GPU por 300 passos",
            "notebook com 12 núcleos + GPU pequena: o Numba pode ganhar SEMPRE",
            "→ meça no SEU hardware antes de concluir"]),
         "O que esperar: no Colab, com dois núcleos e a T4, a GPU perde na malha pequena e passa à frente a partir de "
         "uns quinhentos por quinhentos. Mas atenção: se você projetar do seu notebook, com doze núcleos e uma GPU de "
         "notebook, o Numba na CPU ganhou em todos os tamanhos no teste. Isso não é erro, é a lição: meça no seu "
         "hardware. Por isso, rode a aula no Colab."),
        ("celula", ("Mapa da laje: a discussão de engenharia", 7, '''
T = simula(laje_inicial(256, cp), 4000)
T = cp.asnumpy(T)                        # passo 3: traga só o resultado
plt.contour(T, levels=[100, 300, 500])'''),
         "Feche com o mapa de temperatura. Pergunte: até onde sobe a curva de quinhentos graus? Com cobrimento de dois "
         "centímetros e meio, a armadura passou de quinhentos graus? É a discussão de engenharia que o cálculo rápido "
         "permite. Se sobrar tempo, o para casa: trocar R por zero vírgula três e ver a instabilidade."),
    ]


# =================================================================== 6. bloco 5
class P06Bloco5(Capitulo):
    TITULO, SUB = "Bloco 5 · O seu exercício na GPU", "2:45–3:15 · slides 12–13 · vídeos 13 e 14 · seção 5"
    ABERTURA = "Bloco cinco, meia hora: o exercício de cada um, e como usar inteligência artificial sem se perder."
    QUADROS = [
        ("tempo", (5, "2:45–3:15"), "Duas horas e quarenta e cinco de aula."),
        ("arvore", ("Mostre a lista no projetor", [
            (0, "exercicios/", False), (1, "lista.md", True), (2, "como fazer, em 5 passos", False),
            (2, "exemplo resolvido: energia de 50 M molas", True), (2, "8 exercícios, função serial pronta", False),
            (2, "usando IA", False), (1, "modelo_relatorio.ipynb   (troque só as células marcadas)", True), (1, "INSTALACAO.md", True)]),
         "Abra a lista no projetor. Mostre a estrutura: cinco passos, um exemplo resolvido completo, oito exercícios "
         "com a função serial pronta, e a seção sobre inteligência artificial. A escolha é CPU com Numba, que é o "
         "recomendado, ou GPU com CuPy. A avaliação é só de escalabilidade forte."),
        ("numeros", ("Escada do exemplo (notebook de teste, 20 M molas)",
                     [(k, ES["Python puro"] / v, c) for (k, v), c in zip(ES.items(), (MUTED, MUTED, CPU_C, CPU_C, GPU_C, BAD_C))]
                     if ES else [("Numba", 600, CPU_C)], "×"),
         "Rode a escada da seção cinco: o mesmo problema da lista, em cada versão. A leitura é a regra de ouro outra "
         "vez: com o dado já na GPU, o CuPy empata com a CPU; contando a viagem, ele cai para o fim da escada, porque a "
         "conta é leve e feita uma vez só. Diga que concluir isso, com números, é um ótimo relatório."),
        ("itens", ("Usando IA: as 4 verificações", [
            "1. o resultado bate com a serial?",
            "2. aqueceu antes de medir?",
            "3. sincronizou a GPU?",
            "4. a viagem RAM → GPU está no tempo?"]),
         "Passe o vídeo catorze, ou mostre o prompt da lista. E repita as quatro verificações: o resultado bate, "
         "aqueceu, sincronizou, e a viagem está ou não no tempo."),
        ("diga", "A IA escreve; vocês conferem, medem e explicam.",
         "Esta é a frase do bloco. E peça que, no relatório, digam em uma linha o que a inteligência artificial fez."),
        ("celula", ("Modelo de relatório: só as células marcadas", "*", '''
def serial(k, d): ...              # <- troque função serial do exercício
@njit(parallel=True)
def acelerada(k, d): ...           # <- troque versão Numba
def acelerada_gpu(kg, dg): ...     # <- troque versão CuPy (opcional)'''),
         "Abra o modelo de relatório e mostre que só as células com o lápis mudam. O resto mede, confere e desenha os "
         "gráficos sozinho. Para quem quer rodar no próprio computador, mostre o guia de instalação."),
    ]


# =================================================================== 7. bloco 6
class P07Bloco6(Capitulo):
    TITULO, SUB = "Bloco 6 · Quando usar GPU", "3:15–3:30 · slides 14–15 · vídeo 11"
    ABERTURA = "Bloco seis, os últimos quinze minutos: quando usar GPU, e as outras ferramentas."
    QUADROS = [
        ("tempo", (6, "3:15–3:30"), "Três horas e quinze."),
        ("itens", ("Checklist em 5 perguntas", [
            "1. muitos dados, mesma conta?", "2. os dados podem ficar na GPU?", "3. muita conta por número?",
            "4. float32 basta?", "5. mediu direito?"]),
         "Leia o checklist e aplique às duas ou três contas que a turma citou no começo da aula. É o fechamento mais "
         "forte: o conteúdo voltando para os problemas deles."),
        ("itens", ("Outras ferramentas: só para saber que existem", [
            "CuPy (a estrela) — resolve quase tudo",
            "Numba-CUDA — kernels próprios (apêndice A)",
            "PyTorch / JAX — otimização, derivadas automáticas",
            "RAPIDS cuDF — tabelas grandes · Apple MLX — Macs"]),
         "Mostre a tabela de ferramentas sem entrar em nenhuma. O CuPy resolve quase tudo. Numba CUDA, PyTorch, JAX, "
         "RAPIDS e MLX são para saber que existem. Quem tiver curiosidade encontra no apêndice do notebook um kernel "
         "e a memória gerenciada."),
        ("diga", "Medir, conferir, achar o gargalo.",
         "Defina o prazo da lista, mostre a rubrica, e encerre com a frase do curso: medir, conferir, achar o gargalo."),
    ]


# =================================================================== 8. plano B e checklist
class P08PlanoB(Capitulo):
    TITULO, SUB = "Plano B, cortes e checklist do dia", ""
    ABERTURA = "Para terminar: o que fazer quando algo dá errado, o que cortar, e o checklist do dia."
    QUADROS = [
        ("trava", ["Colab sem GPU para todos → projete o notebook publicado (já executado)",
                   "Cota de GPU acabou → caminho CPU com Numba (é o recomendado)",
                   "Aluno com Mac M → CuPy não roda; Numba roda muito bem",
                   "“A IA fez 1000× mais rápido” → quase sempre faltou sincronizar"],
         "Se o Colab negar GPU para todos, projete o notebook publicado, que já vem executado, e rode só as células de "
         "CPU. Se a cota de algum aluno acabar, ele faz o caminho CPU com Numba, que é o recomendado. Aluno com Mac: o "
         "CuPy não roda, mas o Numba roda muito bem. E se alguém aparecer com um resultado mil vezes mais rápido, "
         "pergunte se sincronizou."),
        ("itens", ("Se o tempo apertar, corte nesta ordem", [
            "1. vídeo 08b → para casa (fique no slide 6)",
            "2. matrizes float32 × float64 → só o slide 9",
            "3. “para casa” da laje",
            "nunca corte: a viagem (bloco 2) e a escada (bloco 5)"]),
         "Se o tempo apertar, corte nesta ordem: o vídeo oito b, as matrizes, e o para casa da laje. Nunca corte a "
         "medida da viagem, no bloco dois, nem a escada, no bloco cinco. Elas são a aula."),
        ("itens", ("Checklist do dia", [
            "☐ notebook, slides, vídeos e lista abertos",
            "☐ T4 ligada e 1ª célula com GPU: True",
            "☐ regra de ouro e 3 passos no quadro",
            "☐ prazo da lista definido (entrega 24/11)"]),
         "O checklist do dia: as quatro abas abertas; a T4 ligada e a primeira célula rodada; a regra de ouro e os três "
         "passos do CuPy no quadro; e o prazo da lista definido. Boa aula."),
    ]
