# Anexo — Como fazer, passo a passo, e um exemplo resolvido

## Tudo o que você precisa

| Arquivo | Para quê |
|---|---|
| esta lista | os exercícios e, neste anexo, um **exemplo resolvido** |
| [lista.md](lista.md) (versão online) | os mesmos exercícios, com a **função serial pronta** de cada um |
| [INSTALACAO.md](INSTALACAO.md) | como rodar no Colab (sem instalar nada) ou no seu computador |
| [modelo_relatorio.ipynb](modelo_relatorio.ipynb) | notebook que **já mede, confere e desenha os gráficos** — você só cola suas funções |
| [vídeos 11–14](https://lucaswastaken.github.io/Openhardware-hpc/#v11) | relatório, exemplo com Numba, exemplo com CuPy e como usar IA (≈ 1 min cada) |

## Como fazer, em 5 passos

- **1. Escolha** um dos exercícios da lista.
- **2. Abra o modelo** no Colab ([INSTALACAO.md](INSTALACAO.md), passo 1) e escreva a **versão serial** (um laço simples; a versão online da lista traz uma pronta para cada exercício) na seção 2: ela é a referência para conferir o resultado.
- **3. Acelere** a função: com Numba (`@njit(parallel=True)` + `prange`) **ou** com CuPy (troque `np` por `cp`). Copie o exemplo resolvido abaixo e adapte.
- **4. Confira** que o resultado é igual ao da versão serial (o modelo já faz isso com `assert`).
- **5. Meça a escalabilidade forte** (o modelo faz a tabela e os gráficos) e **escreva a análise**.

## O que é escalabilidade forte (sem fórmula)

O problema fica **do mesmo tamanho** e você aumenta o número de trabalhadores. É como uma obra com área fixa: com 1 pedreiro leva 12 dias; com 4 pedreiros, se tudo dividir bem, 3 dias.

- **Speedup** = quantas vezes ficou mais rápido = tempo com 1 ÷ tempo com p.
- **Eficiência** = speedup ÷ p. 100% é perfeito; 50% quer dizer que metade do esforço extra se perdeu (esperando memória, combinando resultados, pedreiros se atrapalhando).

| Caminho | O que variar | O que comparar |
|---|---|---|
| **CPU — Numba** (recomendado) | número de threads: 1, 2, 4… até os núcleos da máquina (`numba.set_num_threads(p)`) | speedup e eficiência × threads |
| **GPU — CuPy** | na GPU não se escolhe o número de "trabalhadores"; por isso monte uma **escada**: o mesmo problema em cada versão | Python → NumPy → Numba 1 thread → Numba todas → CuPy (dado já na GPU) → CuPy contando a cópia |

> No Colab gratuito a CPU tem só **2 núcleos**: a curva vai até p = 2. Se puder, rode também no seu computador (mais núcleos) e compare.

## Exemplo resolvido — energia de deformação de 50 milhões de molas

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
