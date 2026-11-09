# Lista de exercícios — Computação de Alto Desempenho em Python

**Escolha UM exercício e UM caminho: CPU com Numba (recomendado) ou GPU com CuPy.** Entregue o notebook executado e um **relatório de 1–2 páginas**.

Tudo o que você precisa:

| Arquivo | Para quê |
|---|---|
| esta lista | o problema, a função serial pronta e um **exemplo resolvido** |
| [INSTALACAO.md](INSTALACAO.md) | como rodar no Colab (sem instalar nada) ou no seu computador |
| [modelo_relatorio.ipynb](modelo_relatorio.ipynb) | notebook que **já mede, confere e desenha os gráficos** — você só cola suas funções |
| [vídeos 11–14](https://lucaswastaken.github.io/Openhardware-hpc/#v11) | relatório, exemplo com Numba, exemplo com CuPy e como usar IA (≈ 1 min cada) |

---

## Como fazer, em 5 passos

1. **Escolha** um dos 8 exercícios abaixo.
2. **Abra o modelo** no Colab ([INSTALACAO.md](INSTALACAO.md), passo 1) e cole a **função serial** do exercício na seção 2. Ela já está pronta: é a referência para conferir o resultado.
3. **Acelere** a função: com Numba (`@njit(parallel=True)` + `prange`) **ou** com CuPy (troque `np` por `cp`). Copie o exemplo resolvido abaixo e adapte.
4. **Confira** que o resultado é igual ao da função serial (o modelo já faz isso com `assert`).
5. **Meça a escalabilidade forte** (o modelo faz a tabela e os gráficos) e **escreva a análise**.

## O que é escalabilidade forte (sem fórmula)

O problema fica **do mesmo tamanho** e você aumenta o número de trabalhadores. É como uma obra com área fixa: com 1 pedreiro leva 12 dias; com 4 pedreiros, se tudo dividir bem, 3 dias.

- **Speedup** = quantas vezes ficou mais rápido = tempo com 1 ÷ tempo com p.
- **Eficiência** = speedup ÷ p. 100% é perfeito; 50% quer dizer que metade do esforço extra se perdeu (esperando memória, combinando resultados, pedreiros se atrapalhando).

| Caminho | O que variar | O que comparar |
|---|---|---|
| **CPU — Numba** (recomendado) | número de threads: 1, 2, 4… até os núcleos da máquina (`numba.set_num_threads(p)`) | speedup e eficiência × threads |
| **GPU — CuPy** | na GPU não se escolhe o número de "trabalhadores"; por isso monte uma **escada**: o mesmo problema em cada versão | Python → NumPy → Numba 1 thread → Numba todas → CuPy (dado já na GPU) → CuPy contando a cópia |

> No Colab gratuito a CPU tem só **2 núcleos**: a curva vai até p = 2. Se puder, rode também no seu computador (mais núcleos) e compare.

---

## ✅ Exemplo resolvido — energia de deformação de 50 milhões de molas

**Problema:** a energia elástica de uma estrutura discretizada em molas é $U = \tfrac12 \sum k_i\,\delta_i^2$ (rigidez × deslocamento²). Com 50 milhões de molas, quanto dá para acelerar?

**1. Função serial (a referência):**
```python
def energia_serial(k, d):
    u = 0.0
    for i in range(len(k)):
        u += 0.5 * k[i] * d[i] * d[i]
    return u
```

**2. Versão Numba — só três mudanças:** o decorador, `range` → `prange`, e escolher quantas threads usar. O `u +=` dentro do `prange` vira automaticamente uma soma em paralelo (cada thread soma a sua parte e o Numba junta no fim).
```python
from numba import njit, prange, set_num_threads

@njit(parallel=True)                      # 1) compile e paralelize
def energia_numba(k, d):
    u = 0.0
    for i in prange(k.size):              # 2) prange: as voltas são divididas entre as threads
        u += 0.5 * k[i] * d[i] * d[i]
    return u

rng = np.random.default_rng(0)
k = rng.uniform(1e6, 1e8, 50_000_000)    # rigidez [N/m]
d = rng.normal(0, 1e-3, 50_000_000)      # deslocamento [m]
ref = 0.5 * np.dot(k, d * d)             # referência rápida (NumPy) para conferir

for p in (1, 2, 4, 8, 12):
    set_num_threads(p)                   # 3) quantas threads usar nesta medida
    assert np.isclose(energia_numba(k, d), ref)   # confere ANTES de medir
    # ... medir com o modelo (aquece 1 vez, repete 5, guarda o menor tempo)
```

**3. Resultado** (notebook com 12 núcleos lógicos, Intel; Colab terá 2):

| threads | tempo | speedup | eficiência |
|---|---|---|---|
| 1 | 0,067 s | 1,0× | 100% |
| 2 | 0,061 s | 1,1× | 55% |
| 4 | 0,043 s | 1,6× | 39% |
| 8 | 0,026 s | 2,6× | 33% |
| 12 | 0,019 s | 3,4× | 29% |

**4. Análise (o que escrever):** cada mola faz só 3 multiplicações, mas precisa ler 16 bytes da memória. Com 12 threads o programa lê **41 GB/s** — perto do limite da memória RAM do notebook. As threads ficam **esperando dados**, não calculando: por isso a eficiência cai. Ainda assim, comparado ao Python puro (13 s), a versão final é **≈ 670× mais rápida**; e 1 thread de Numba já é 2× mais rápido que o NumPy (0,14 s), porque não cria o vetor temporário `d*d`.

**5. Mesmo problema na GPU (CuPy) — a escada:**
```python
import cupy as cp
kg, dg = cp.asarray(k), cp.asarray(d)    # copia RAM → GPU (uma vez)
u = float(0.5 * cp.dot(kg, dg * dg))     # mesmo código do NumPy, com cp no lugar de np
```

| versão | tempo | × Python |
|---|---|---|
| Python puro | 13 s | 1× |
| NumPy | 0,14 s | 94× |
| Numba, 1 thread | 0,067 s | 190× |
| Numba, 12 threads | 0,019 s | 670× |
| CuPy, dado já na GPU (GeForce MX570, de notebook) | 0,027 s | 490× |
| CuPy **contando a cópia** RAM → GPU | 0,26 s | 49× |

**Análise:** copiar 800 MB para a GPU custa **10× mais** que a conta. Para uma operação só, não compensa; compensa quando o dado vai para a GPU **uma vez** e é usado muitas vezes (uma simulação com milhares de passos, por exemplo). Numa GPU maior (a T4 do Colab) a conta fica mais rápida — mas a cópia continua lá.

🎬 Este exemplo em vídeo: [Numba variando as threads](https://lucaswastaken.github.io/Openhardware-hpc/#v12) · [a escada até a GPU](https://lucaswastaken.github.io/Openhardware-hpc/#v13).

---

## Os 8 exercícios

Em todos: use **`float64`**, o tamanho sugerido (ou o maior que couber no Colab) e confira sempre com a função serial ou com o NumPy. A **função serial está pronta** — o seu trabalho é acelerá-la e medir.

### 1. Produto escalar
**Aplicação:** trabalho de forças $W = \sum F_i\,u_i$ num modelo com milhões de graus de liberdade. **Tamanho:** $5\times10^7$.
```python
def escalar_serial(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s
```
- **CPU:** igual ao exemplo resolvido. **GPU:** `cp.dot`.
- **Pergunta:** quantos GB/s seu programa lê (bytes lidos ÷ tempo)? Isso explica a eficiência?

### 2. Multiplicação de matrizes densas
**Aplicação:** condensação estática de matrizes de rigidez. **Tamanho:** $n = 1000$ (Python puro só com $n = 200$ — e multiplique o tempo por $(1000/200)^3$).
```python
def matmul_serial(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C
```
- **CPU:** `prange` no laço de fora (`i`): cada thread calcula linhas inteiras de C. **GPU:** `A_gpu @ B_gpu`.
- **Pergunta:** quantos GFLOP/s ($2n^3$ ÷ tempo) você atinge? E o `A @ B` do NumPy? Por que ele é tão mais rápido?

### 3. Monte Carlo para π
**Aplicação:** a mesma técnica da probabilidade de falha em confiabilidade estrutural. **Tamanho:** $10^8$ pontos.
```python
def pi_serial(n):
    dentro = 0
    for _ in range(n):
        x, y = np.random.random(), np.random.random()
        if x * x + y * y <= 1.0:
            dentro += 1
    return 4.0 * dentro / n
```
- **CPU:** `prange` + `dentro += 1` funciona com Numba (cada thread tem seu próprio gerador). **GPU:** gere todos os pontos de uma vez com `cp.random.random(n)`.
- **Conferir:** aqui o resultado não é igual ao serial (é aleatório) — confira que o erro em relação a `np.pi` é menor que $3/\sqrt{n}$.
- **Pergunta:** por que este problema escala melhor que o produto escalar?

### 4. Soma de um vetor (redução)
**Aplicação:** soma de esforços, norma do resíduo num solver iterativo. **Tamanho:** $10^8$.
```python
def soma_serial(x):
    s = 0.0
    for v in x:
        s += v
    return s
```
- **CPU:** `prange` + `s += x[i]`. **GPU:** `cp.sum`.
- **Pergunta:** repita com `float32` (`x.astype(np.float32)` e acumulador `s = np.float32(0.0)`). O tempo muda? E o resultado? (Teste: some $10^7$ valores iguais a 0,1 — o certo é $10^6$; com acumulador `float32` em laço dá ≈ 1 088 000, quase 9% de erro. E o `np.sum`?)

### 5. Busca em vetor
**Aplicação:** encontrar as seções com tensão acima do limite. **Tamanho:** $5\times10^7$, com 10% dos valores acima do limite.
```python
def busca_serial(x, limite):
    achados = []
    for i in range(len(x)):
        if x[i] > limite:
            achados.append(i)
    return np.array(achados)
```
- **CPU:** com `prange`, marque um vetor booleano `acima[i] = x[i] > limite` e depois use `np.flatnonzero(acima)`. **GPU:** `cp.flatnonzero(x_gpu > limite)`.
- **Pergunta:** a parte de juntar os índices é paralela? Quanto do tempo total ela ocupa?

### 6. Ordenação
**Aplicação:** ordenar elementos por tensão para achar os 1% mais solicitados. **Tamanho:** $2\times10^7$.
```python
def ordena_serial(x):
    return np.sort(x)        # aqui a referência é o próprio NumPy
```
- **CPU:** divida o vetor em p blocos, ordene cada bloco num `prange` (`np.sort` funciona dentro do Numba) e no fim junte tudo com `np.sort(..., kind="mergesort")`. **GPU:** `cp.sort`.
- **Pergunta:** meça separado o tempo dos blocos e o da junção final. A junção não paraleliza — é ela que limita o speedup?

### 7. Ponto dentro de polígono
**Aplicação:** quais pontos de um levantamento (lidar, topografia) caem dentro de um lote. **Tamanho:** $10^6$ pontos, polígono com 200 vértices.
```python
def dentro_serial(px, py, vx, vy):          # pontos (px, py); polígono (vx, vy)
    n, m = len(px), len(vx)
    dentro = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        c = False
        for j in range(m):                    # algoritmo do raio: conta quantas arestas o raio cruza
            k = j - 1
            if (vy[j] > py[i]) != (vy[k] > py[i]):
                if px[i] < (vx[k] - vx[j]) * (py[i] - vy[j]) / (vy[k] - vy[j]) + vx[j]:
                    c = not c
        dentro[i] = c
    return dentro
```
- **CPU:** `prange` no laço dos pontos (`i`) — cada ponto é independente. **GPU:** faça o laço das arestas em Python e a conta de todos os pontos de uma vez em CuPy.
- **Pergunta:** com 20 e com 2000 vértices, o speedup muda? Por quê?

### 8. Fecho convexo
**Aplicação:** contorno de uma nuvem de pontos (área de obra, levantamento). **Tamanho:** $10^7$ pontos.
```python
def fecho_serial(P):                          # P: array (n, 2); devolve os vértices do fecho
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    inf, sup = [], []
    for p in P:
        while len(inf) >= 2 and cruz(inf[-2], inf[-1], p) <= 0: inf.pop()
        inf.append(p)
    for p in P[::-1]:
        while len(sup) >= 2 and cruz(sup[-2], sup[-1], p) <= 0: sup.pop()
        sup.append(p)
    return np.array(inf[:-1] + sup[:-1])
```
- **Ideia:** divida os pontos em p blocos, calcule o fecho de cada bloco **em paralelo** (com `joblib` ou `concurrent.futures`, já que listas de tamanho variável são difíceis no Numba) e no fim o fecho dos fechos. Confira com `scipy.spatial.ConvexHull`.
- **Pergunta:** quantos pontos sobram depois da primeira etapa? Por que isso torna o método eficiente?

### Extra: problema proposto por você
Traga um problema da sua área (estruturas, geotecnia, hidráulica, transportes, imagens…) com muitas contas independentes. Mande uma proposta de 1 parágrafo, com a função serial, antes de começar — assunto **"Proposta HPC – seu nome"**.

---

## 🤖 Usando IA (pode e deve — com cuidado)

Pedir ajuda ao ChatGPT, Claude, Gemini ou Copilot é permitido. Funciona bem assim:

> *"Tenho esta função em Python [cole a função serial]. Reescreva com Numba usando `@njit(parallel=True)` e `prange`, sem mudar o resultado. Explique cada linha que mudou."*

> *"Reescreva esta função com CuPy, mantendo os dados na GPU. Mostre como medir o tempo corretamente (aquecer e sincronizar)."*

**A IA erra com frequência em HPC. Antes de acreditar, verifique sempre as três coisas:**
1. **O resultado bate** com a função serial (`assert np.isclose(...)`). Uma versão rápida e errada vale zero.
2. **Aquecimento:** a 1ª chamada de Numba (compila) e de CuPy (inicializa a GPU) é lenta — não meça ela.
3. **Sincronização na GPU:** sem `cp.cuda.Device().synchronize()` você mede só o tempo de *pedir* a conta, não de fazê-la.

O modelo de relatório já faz as três. **No relatório, diga em uma linha o que a IA fez** (ex.: "a versão Numba foi sugerida pelo ChatGPT e eu corrigi a soma").

---

## O que entregar (relatório de 1–2 páginas)

1. **Problema e caminho escolhido** (CPU/Numba ou GPU/CuPy) — 2–3 linhas.
2. **Como acelerou:** o que mudou em relação à função serial.
3. **Correção:** mostre que o resultado confere.
4. **Tabela + gráficos** de escalabilidade forte (CPU: speedup e eficiência × threads; GPU: a escada).
5. **Análise:** o que limitou o ganho? (memória, cópia para a GPU, parte que não paraleliza, poucas contas por dado…)
6. **Máquina:** CPU (núcleos), GPU, se foi no Colab.

### Rubrica (10 pontos)

| Critério | Pontos |
|---|---|
| Versão acelerada correta e conferida com a serial | 3,0 |
| Medidas bem feitas (aquecer, repetir, sincronizar) | 2,0 |
| Tabela e gráficos de escalabilidade forte (ou escada GPU) | 2,5 |
| Análise: o que limitou o ganho, com números | 2,0 |
| Clareza (e dizer o que a IA fez) | 0,5 |
