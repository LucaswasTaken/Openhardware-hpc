# Computação de Alto Desempenho em Python
## Curso para Engenharia Civil — 3 Aulas × 2h

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)](https://jupyter.org/)

---

## 🎯 Visão Geral

Este curso oferece uma introdução prática à **Computação de Alto Desempenho (HPC)** em Python, especificamente voltado para estudantes de **Engenharia Civil**. O objetivo é capacitar os alunos a compreender e aplicar conceitos de paralelismo e otimização de performance em problemas de engenharia.

### Público-Alvo
- Estudantes de Engenharia Civil com experiência básica em programação
- Iniciantes em paralelismo e computação de alto desempenho
- Interessados em acelerar simulações e cálculos de engenharia

### Objetivos do Curso
- Compreender princípios fundamentais de paralelismo e performance
- Aplicar técnicas de paralelização em CPU e GPU
- Implementar soluções de HPC para problemas reais de engenharia
- Medir e avaliar ganhos de performance (speedup e escalabilidade)

---

## 📚 Estrutura do Curso

### 🧩 Aula 1 – Pensando em Paralelismo e Performance (CPU)
**Duração:** 2 horas  
**Foco:** Conceitos fundamentais de paralelismo e implementação com multiprocessing

#### Conteúdo Principal:
- Introdução ao paralelismo: da computação serial ao paralela
- Conceitos de speedup, eficiência e overhead
- Amdahl's Law e Gustafson's Law
- Arquitetura de CPU: núcleos, cache e memória
- Diferenças entre threads e processos
- Global Interpreter Lock (GIL) em Python
- Paralelização com `multiprocessing`

#### Exemplos Práticos:
- Soma de vetores: serial vs NumPy vs threading
- Estimativa de π por Monte Carlo paralelo
- Multiplicação de matrizes em blocos
- Aplicações em Finite Element Method (FEM)

### ⚙️ Aula 2 – Paralelismo Avançado e Escalabilidade
**Duração:** 2 horas  
**Foco:** Ferramentas avançadas de paralelização e medição de performance

#### Conteúdo Principal:
- Strong vs Weak scaling
- Medição de performance: `timeit`, `cProfile`
- `ProcessPoolExecutor` e `concurrent.futures`
- Biblioteca `joblib` para loops paralelos
- Compilação JIT com Numba
- Paralelismo automático e vetorização SIMD
- Limites práticos do paralelismo

#### Exemplos Práticos:
- Integração numérica com futures
- Regressões lineares paralelas com joblib
- Multiplicação de matrizes com Numba
- Análise de escalabilidade
- Aplicações em simulações estruturais

### ⚡ Aula 3 – GPUs em Python e Aplicações em Engenharia
**Duração:** 2 horas  
**Foco:** Paralelismo massivo com GPU usando CuPy e Numba CUDA

#### Conteúdo Principal:
- Arquitetura GPU: threads, blocks, grids
- Hierarquia de memória GPU
- CuPy: NumPy para GPU
- Numba CUDA: kernels customizados
- Transferência eficiente de dados Host↔Device
- Integração CPU + GPU
- Ferramentas avançadas: Dask, PyTorch

#### Exemplos Práticos:
- Operações vetoriais em GPU
- Multiplicação de matrizes massivas
- Monte Carlo π em GPU
- Simulação de difusão de calor 2D
- Comparação CPU vs GPU performance

---

## 🛠️ Tecnologias e Ferramentas

### Bibliotecas Principais:
- **NumPy**: Computação científica fundamental
- **Multiprocessing**: Paralelismo em CPU
- **Joblib**: Paralelização de loops
- **Numba**: Compilação JIT e CUDA
- **CuPy**: NumPy para GPU
- **Matplotlib**: Visualização de resultados
- **Jupyter**: Ambiente interativo

### Pré-requisitos de Sistema:
- Python 3.8+ 
- 8GB+ RAM recomendado
- GPU NVIDIA (opcional, para Aula 3)
- CUDA Toolkit (para exemplos GPU)

---

## 📁 Organização dos Arquivos

```
├── README.md                          # Este arquivo
├── HPC_Python_Course_Plan.md         # Plano detalhado do curso
├── exercice_list.md                  # Lista de exercícios
├── notebooks/
│   ├── Aula1_Paralelismo_CPU.ipynb  # Notebook da Aula 1
│   ├── Aula2_Paralelismo_Avancado.ipynb # Notebook da Aula 2
│   └── Aula3_GPU_Computing.ipynb    # Notebook da Aula 3
├── src/
│   ├── aula1/                        # Códigos da Aula 1
│   │   ├── vector_sum_serial.py
│   │   ├── vector_sum_numpy.py
│   │   ├── vector_sum_threading.py
│   │   ├── pi_montecarlo_multiprocessing.py
│   │   └── matrix_multiplication_multiprocessing.py
│   ├── aula2/                        # Códigos da Aula 2
│   │   ├── integrate_trapezoidal_futures.py
│   │   ├── linear_regression_joblib.py
│   │   ├── matrix_mult_numba.py
│   │   └── scaling_vector_sum.py
│   └── aula3/                        # Códigos da Aula 3
│       ├── vector_sum_cupy.py
│       ├── matrix_multiplication_cupy.py
│       ├── pi_montecarlo_cuda.py
│       └── heat_equation_gpu.py
└── deprecated/                       # Códigos legados (mantidos para referência)
```

---

## 🚀 Como Usar Este Material

### 1. Configuração do Ambiente
```bash
# Instalar dependências básicas
pip install numpy matplotlib jupyter joblib numba

# Para exemplos GPU (opcional)
pip install cupy-cuda11x  # ou cupy-cuda12x dependendo da versão CUDA
```

### 2. Executar os Notebooks
```bash
# Navegar até o diretório do curso
cd Openhardware-hpc

# Iniciar Jupyter Notebook
jupyter notebook

# Abrir o notebook da aula desejada em notebooks/
```

### 3. Executar Códigos Standalone
```bash
# Exemplos da Aula 1
python src/aula1/vector_sum_serial.py
python src/aula1/pi_montecarlo_multiprocessing.py

# Exemplos da Aula 2
python src/aula2/matrix_mult_numba.py

# Exemplos da Aula 3 (requer GPU)
python src/aula3/vector_sum_cupy.py
```

---

## 📊 Resultados de Aprendizagem

Ao final do curso, os alunos serão capazes de:

1. **Compreender** conceitos fundamentais de paralelismo e performance
2. **Implementar** soluções paralelas para problemas CPU-bound usando multiprocessing
3. **Utilizar** ferramentas avançadas como joblib e Numba para otimização
4. **Aplicar** computação GPU em Python com CuPy e Numba CUDA
5. **Medir e analisar** ganhos de performance e escalabilidade
6. **Resolver** problemas reais de engenharia civil com HPC

---

## 🎯 Aplicações em Engenharia Civil

### Simulações Estruturais
- Análise de elementos finitos (FEM) paralela
- Montagem paralela de matrizes de rigidez
- Simulações de vibração e dinâmica estrutural

### Análise de Fluxo e Transporte
- Computational Fluid Dynamics (CFD)
- Difusão de calor e umidade
- Transporte de contaminantes

### Métodos Estocásticos
- Simulações Monte Carlo para confiabilidade
- Análise de incertezas em materiais
- Otimização probabilística

### Processamento de Dados
- Análise de grandes datasets de sensores
- Processamento de imagens de inspeção
- Machine learning para predição de falhas

---

## 📞 Contato e Suporte

**Instrutor:** Lucas Gouveia Omena Lopes  
**Repositório:** [GitHub - Openhardware-hpc](https://github.com/LucaswasTaken/Openhardware-hpc)

Para dúvidas e sugestões, abra uma [issue](https://github.com/LucaswasTaken/Openhardware-hpc/issues) no repositório.

---

## 📄 Licença

Este material educacional está disponível sob licença open source. Veja o arquivo `LICENSE` para detalhes.

---

**Última atualização:** Outubro 2025  
**Versão:** 1.0