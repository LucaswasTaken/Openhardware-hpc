# Computação de Alto Desempenho em Python
### Curso para Engenharia Civil — 3 Aulas × 2h

---

## 🧭 Visão Geral

**Título:** Computação de Alto Desempenho em Python  
**Público-alvo:** Estudantes de Engenharia Civil com experiência em programação, mas iniciantes em paralelismo.  
**Objetivo:** Compreender princípios de desempenho, paralelismo e aplicações práticas em simulações de engenharia.  
**Duração:** 3 aulas (2h cada) — total de 6 horas.

---

# 🧩 Aula 1 – Pensando em Paralelismo e Performance (CPU)

## 🎯 Objetivos
- Entender o que é paralelismo e onde ele aparece em problemas de engenharia.
- Introduzir conceitos de speedup, escalabilidade e overhead.
- Escrever primeiros exemplos em Python puro e com paralelismo de CPU.

## 🧱 Estrutura de Slides e Exemplos

### Slide 1 – Título e Contexto
- Computação de Alto Desempenho em Python.
- Por que engenheiros civis devem se importar com HPC.
- Lucas Gouveia Omena Lopes — [email/contact].

### Slide 2 – Motivação: O Crescimento dos Problemas
- Simulações estruturais, fluxo, transporte de calor, Monte Carlo.
- Crescimento de malhas → mais tempo computacional.
- Pergunta: o que acontece se eu dobrar o número de nós?

### Slide 3 – Da Computação Serial ao Paralelismo
- CPU = 1 núcleo → 1 tarefa.
- Computadores modernos = múltiplos núcleos → tarefas simultâneas.
- Exemplo visual: “Soma de Vetores Serial vs Paralela”.

### Slide 4 – Exemplo 1: `vector_sum_serial.py`
```python
import time, numpy as np
N = 10_000_000
a, b = np.arange(N), np.arange(N)
start = time.time()
c = np.zeros_like(a)
for i in range(N):
    c[i] = a[i] + b[i]
print("Tempo:", time.time() - start)
```

### Slide 5 – Exemplo 2: `vector_sum_numpy.py`
```python
start = time.time()
c = a + b
print("Tempo:", time.time() - start)
```
- Introdução ao paralelismo implícito via NumPy.
- Como bibliotecas científicas já exploram múltiplos núcleos.

### Slide 6 – Conceitos-Chave
- Speedup, Eficiência, Overhead.
- Amdahl’s Law e Gustafson’s Law.
- Exemplo gráfico 1→8 núcleos.

### Slide 7 – Arquitetura de CPU e Memória
- Núcleos, cache L1/L2/L3, RAM, HD.
- Acesso à memória domina performance.
- Metáfora: “cozinha pequena com muitos cozinheiros”.

### Slide 8 – Threads vs Processos
- Threads compartilham memória.
- Processos têm memória isolada.
- GIL (Global Interpreter Lock) limita threads Python.

### Slide 9 – Exemplo 3: `vector_sum_threading.py`
- Tentar paralelizar soma com `threading.Thread`.
- Mostrar que GIL impede ganho real.

### Slide 10 – Multiprocessamento
- Cada processo tem seu próprio interpretador Python.
- Comunicação via pipe/queue.
- Ideal para tarefas CPU-bound.

### Slide 11 – Exemplo 4: `pi_montecarlo_multiprocessing.py`
- Estimar π por Monte Carlo.
- Cada processo gera N pontos → combina resultados.
- Medir speedup.

### Slide 12 – Exemplo 5: `matrix_multiplication_multiprocessing.py`
- Multiplicação de matrizes grandes por blocos.
- Comparar 1, 2, 4 processos.
- Monitorar uso de CPU.

### Slide 13 – Cuidados e Dicas
- Evite overhead de comunicação.
- Prefira vetorização e operações em lote.
- Reutilize memória.

### Slide 14 – Conexões com Engenharia
- FEM, CFD, DEM.
- Montagem paralela da matriz de rigidez.
- Simulações independentes por elemento.

### Slide 15 – Encerramento da Aula 1
- Conceitos aprendidos: paralelismo, processos, speedup.
- **Tarefa:** paralelizar código próprio.

---

# ⚙️ Aula 2 – Paralelismo Avançado e Escalabilidade

## 🎯 Objetivos
- Usar ferramentas modernas de paralelismo em CPU.
- Medir escalabilidade e compreender overhead.

## 🧱 Estrutura de Slides e Exemplos

### Slide 1 – Revisão e Resultados da Aula 1
- Recapitular Monte Carlo e MatMul paralelos.
- Mostrar speedup obtido.

### Slide 2 – O Problema da Escalabilidade
- Strong vs Weak scaling.
- Gráfico conceitual log-log.

### Slide 3 – Medindo Tempo e Performance
- `%timeit`, `time.perf_counter`, `cProfile`.

### Slide 4 – Exemplo 6: `integrate_trapezoidal_futures.py`
- Integração numérica com `ProcessPoolExecutor`.
- Dividir domínio entre processos e combinar resultados.

### Slide 5 – Biblioteca `joblib`
- Interface simples para loops paralelos.
- Usada no backend do scikit-learn.

### Slide 6 – Exemplo 7: `linear_regression_joblib.py`
- Regressões lineares em paralelo.
- Aplicação em modelos de engenharia.

### Slide 7 – Introdução ao Numba
- Compilação JIT.
- `@njit(parallel=True)`.

### Slide 8 – Exemplo 8: `matrix_mult_numba.py`
- Comparar Python, NumPy, Numba serial e paralelo.
- Mostrar gráfico de speedup.

### Slide 9 – Paralelismo Automático e SIMD
- Vetorização (AVX2/AVX512).

### Slide 10 – Exemplo 9: `scaling_vector_sum.py`
- Rodar com N = 10⁶, 10⁷, 10⁸.
- Plotar tempo vs tamanho.

### Slide 11 – Limites do Paralelismo
- Amdahl revisado, overhead, sincronização.

### Slide 12 – Aplicações de Engenharia
- Integração numérica FEM.
- Confiabilidade via Monte Carlo.

### Slide 13 – Demonstração Final
- Pipeline: profiling → paralelização → medição.

### Slide 14 – Encerramento da Aula 2
- Ferramentas: multiprocessing, futures, joblib, numba.
- **Tarefa:** testar escalabilidade do código.

---

# ⚡ Aula 3 – GPUs em Python e Aplicações em Engenharia

## 🎯 Objetivos
- Introduzir paralelismo massivo com GPU.
- Usar CuPy e Numba CUDA.
- Aplicar HPC a simulações reais.

## 🧱 Estrutura de Slides e Exemplos

### Slide 1 – Revisão CPU vs GPU
- CPU: poucos núcleos complexos.
- GPU: milhares de núcleos simples.

### Slide 2 – Arquitetura GPU
- Threads, blocks, grids.
- Hierarquia de memória.

### Slide 3 – Exemplo 10: `vector_sum_cupy.py`
```python
import cupy as cp
N = 10_000_000
a, b = cp.arange(N), cp.arange(N)
c = a + b
```
- `np → cp`, medir aceleração.

### Slide 4 – Exemplo 11: `matrix_multiplication_cupy.py`
- Multiplicação 4096×4096 GPU vs CPU.

### Slide 5 – Transferência de Dados
- Host→Device e Device→Host.
- Minimizar transferências.

### Slide 6 – Numba CUDA: Primeiros Passos
```python
from numba import cuda
@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]
```
- Conceito de kernel e grid.

### Slide 7 – Exemplo 12: `pi_montecarlo_cuda.py`
- Monte Carlo π em GPU.

### Slide 8 – Exemplo 13: `heat_equation_gpu.py`
- Difusão de calor 2D em GPU.
- Visualização do campo térmico.

### Slide 9 – Integração CPU + GPU
- Pré-processamento no CPU, simulação no GPU.

### Slide 10 – Ferramentas Avançadas
- Dask + CuPy, PyTorch, Singularity.

### Slide 11 – Estudo de Caso
- Simulação de calor em placa metálica.
- Comparação CPU, GPU, Numba.

### Slide 12 – Conclusões e Próximos Passos
- HPC em engenharia civil.
- MPI4Py, clusters, SYCL, oneAPI.

### Slide 13 – Projeto Final
> Paralelize um problema real de engenharia civil e compare performance CPU vs GPU.

---

## 🎓 Resultados de Aprendizagem

1. Entender conceitos de paralelismo e performance.  
2. Paralelizar tarefas CPU-bound com multiprocessing e Numba.  
3. Usar GPU em Python com CuPy e Numba CUDA.  
4. Aplicar HPC a problemas reais de engenharia civil.
