# Lista de exercícios — Computação de Alto Desempenho em Python

**Escolha UM exercício e UMA tecnologia** (pode ser mais de uma, se quiser comparar). Entregue o notebook executado e um **relatório de 1–2 páginas**. O [modelo de relatório](modelo_relatorio.ipynb) já mede tempo, speedup e eficiência e gera os gráficos — comece por ele.

🎬 Antes de começar, assista: [Como fazer o relatório](https://lucaswastaken.github.io/Openhardware-hpc/#v11) e os exemplos resolvidos ([12](https://lucaswastaken.github.io/Openhardware-hpc/#v12), [13](https://lucaswastaken.github.io/Openhardware-hpc/#v13), [14](https://lucaswastaken.github.io/Openhardware-hpc/#v14)) — eles resolvem problemas **parecidos** com os desta lista.

## Tecnologias

| Tecnologia | Aula | Bom para |
|---|---|---|
| NumPy vetorizado | 1 | referência obrigatória: compare sempre com ela |
| `multiprocessing` / `concurrent.futures` | 1 | tarefas independentes e pesadas |
| `joblib` | 1 | laços independentes, estudos paramétricos |
| Numba (`@njit`, `prange`) | 1 | laços com lógica |
| Python 3.14t (sem GIL) + threads | 1 | experimental — compare com processos |
| CuPy | 2 | arrays na GPU |
| Numba-CUDA | 2 | kernels próprios na GPU |
| Outra (Dask, PyTorch, JAX, Polars…) | — | combine com o professor |

## O que todo relatório precisa ter

1. **Problema** (2–3 linhas) e **tecnologia escolhida**, com justificativa.
2. **Implementação:** como você dividiu o trabalho (por elementos, por blocos, por amostras…).
3. **Correção:** compare o resultado com uma referência (NumPy ou solução analítica).
4. **Medidas** (com aquecimento e repetições; na GPU, sincronize):
   - tempo serial e paralelo;
   - **speedup** $S = T_{serial}/T_p$ e **eficiência** $E = S/p$;
   - **escalabilidade forte** (tamanho fixo, varie $p$) e **fraca** (tamanho cresce com $p$) — ou, na GPU, varie o tamanho do problema;
   - para problemas limitados por memória: **largura de banda** atingida (GB/s).
5. **Gráficos:** speedup × $p$ (com a reta ideal) e eficiência × $p$.
6. **Análise:** onde está o gargalo (cálculo, memória, transferências, overhead)? O que limitou o speedup?
7. **Hardware:** CPU (núcleos), GPU, memória — e se rodou no Colab.

### Rubrica (10 pontos)

| Critério | Pontos |
|---|---|
| Implementação correta e validada | 2,5 |
| Medidas bem feitas (aquecer, repetir, sincronizar, separar transferência) | 2,5 |
| Speedup, eficiência e escalabilidade com gráficos | 2,5 |
| Análise do gargalo e conclusões | 2,0 |
| Clareza do relatório | 0,5 |

---

## 1. Produto escalar
**Aplicação:** trabalho de forças, energia ($W = \mathbf{F}\cdot\mathbf{u}$) em sistemas com milhões de graus de liberdade.
- Calcule $\sum_i a_i b_i$ para vetores de $10^6$, $10^7$ e $10^8$ elementos (`float64`).
- Compare: laço Python, `np.dot`, sua versão paralela.
- **Análise:** é limitado por cálculo ou por memória? Calcule a banda atingida (GB/s) e compare com a da sua máquina.

## 2. Multiplicação de matrizes densas
**Aplicação:** montagem e condensação de matrizes de rigidez.
- $C = AB$ com $n$ = 500, 1000, 2000. Implemente o laço triplo e uma versão por **blocos** (*tiling*).
- Meça **GFLOP/s** ($2n^3/t$) e compare com `A @ B` (BLAS).
- **Análise:** por que a versão por blocos é mais rápida (cache)? Na GPU, compare `float32` × `float64`.

## 3. Monte Carlo para π
**Aplicação:** a mesma técnica da probabilidade de falha em confiabilidade estrutural.
- Estime π com $10^6$ a $10^9$ pontos. Mostre o erro × número de amostras (deve cair com $1/\sqrt{N}$).
- **Escalabilidade fraca:** amostras por processo constantes.
- **Cuidado:** cada processo/thread precisa de **sementes diferentes** — mostre o que acontece se forem iguais.

## 4. Redução (soma de um array) — CuPy obrigatório
**Aplicação:** somas de esforços, normas de resíduo em solvers iterativos.
- Some $10^7$ a $5\times10^8$ elementos. Compare `np.sum`, `cp.sum` (dado já na GPU) e `cp.sum` incluindo a cópia.
- **Pergunta central:** existe um tamanho a partir do qual vale a pena **copiar** para a GPU só para somar? E se o dado já estivesse lá (ou numa máquina de memória unificada)?
- Compare `float32` e `float64` (precisão e tempo) e soma ingênua × soma de Kahan.

## 5. Busca em array
**Aplicação:** localizar elementos com tensão acima de um limite, nós numa região.
- Encontre **todos os índices** onde `x > limiar` em $10^7$–$10^8$ elementos, com densidades de 1%, 10% e 50%.
- **Análise:** juntar os resultados de cada parte custa quanto? A densidade muda o balanceamento de carga?

## 6. Ordenação
**Aplicação:** ordenar elementos por tensão, eventos por tempo.
- Implemente uma ordenação paralela: *merge sort* em blocos (cada processo ordena um pedaço e você intercala) **ou** *sample sort*.
- Compare com `np.sort` e `cp.sort` para $10^6$–$10^8$ elementos.
- **Análise:** qual parte não paraleliza (a intercalação final)? Relacione com a lei de Amdahl.

## 7. Ponto dentro de polígono
**Aplicação:** quais pontos de uma nuvem (levantamento topográfico, lidar) caem dentro de um lote ou de uma bacia.
- Algoritmo do **raio** (*ray casting*) para $10^6$–$10^7$ pontos e um polígono com 50–5000 vértices.
- Paralelize por pontos (cada ponto é independente).
- **Análise:** como o número de vértices muda o speedup (mais conta por ponto)?

## 8. Fecho convexo
**Aplicação:** envoltória de uma nuvem de pontos, contorno de uma área de obra.
- Implemente *Monotone Chain* ou *Quickhull* para $10^6$–$10^7$ pontos.
- Paralelize: divida os pontos em blocos, calcule o fecho de cada bloco e depois o fecho dos fechos.
- Compare com `scipy.spatial.ConvexHull`.
- **Análise:** quanto dos pontos sobra após a 1ª etapa? Por que isso torna o método eficiente?

## Extra: problema proposto por você
Traga um problema da sua área (estruturas, geotecnia, hidráulica, transportes, imagens…) que tenha muitas contas independentes. Envie uma proposta de 1 parágrafo antes de começar — assunto **"Proposta HPC – seu nome"**.
