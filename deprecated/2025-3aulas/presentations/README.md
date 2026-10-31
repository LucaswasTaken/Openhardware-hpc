# 🎯 Apresentações do Curso - Configuração e Uso

Este diretório contém as apresentações teóricas que complementam os notebooks práticos do curso de Computação de Alto Desempenho em Python para Engenharia Civil.

## 📋 Estrutura das Apresentações

### 🧩 [Aula1_Paralelismo_CPU.md](./Aula1_Paralelismo_CPU.md)
**Tópicos Principais:**
- Motivação para HPC em Engenharia Civil
- Conceitos fundamentais de paralelismo
- Python e o Global Interpreter Lock (GIL)
- Multiprocessing e concurrent.futures
- Métricas de performance (speedup, eficiência)
- Aplicações: Monte Carlo, multiplicação de matrizes, análise estrutural

**Recursos Visuais:**
- Performance evolution charts across all 3 classes
- Amdahl's Law speedup curves with real data
- Threading vs multiprocessing architecture diagrams
- Data parallelism visualization for vector operations
- Monte Carlo method visual demonstration

### ⚙️ [Aula2_Paralelismo_Avancado.md](./Aula2_Paralelismo_Avancado.md)
**Tópicos Principais:**
- Medição profissional de performance
- Ferramentas avançadas: joblib, numba
- Strong vs Weak scaling studies
- Otimizações e boas práticas
- Casos reais de otimização
- Hierarchia de otimização

**Recursos Visuais:**
- Overhead comparison charts (multiprocessing vs joblib)
- Numba speedup charts by code type
- Strong/weak scaling efficiency curves with realistic data
- Optimization cascade showing progressive improvements
- Performance comparison across different techniques

### ⚡ [Aula3_GPU_Computing.md](./Aula3_GPU_Computing.md)
**Tópicos Principais:**
- Arquitetura CPU vs GPU
- CUDA Programming Model
- CuPy para computação GPU
- Numba CUDA para kernels customizados
- Memory management e otimização
- Aplicação: simulação de difusão de calor

**Recursos Visuais:**
- Detailed CPU vs GPU architecture comparison
- CUDA execution hierarchy (Grid → Blocks → Threads)
- GPU memory hierarchy pyramid with performance metrics
- Heat equation performance scaling (CPU vs GPU)
- Complete course performance roadmap (1x to 200x+ speedup)

## 🎬 Como Usar as Apresentações

### Fluxo Recomendado de Aula

1. **📊 Apresentação** (15-20 min)
   - Conceitos teóricos
   - Motivação e contexto
   - Exemplos conceituais

2. **💻 Notebook** (30-40 min)
   - Implementação prática
   - Demonstrações ao vivo
   - Exercícios hands-on

3. **🔄 Volta à Apresentação** (10-15 min)
   - Discussão dos resultados
   - Boas práticas
   - Próximos passos

4. **❓ Discussão** (10-15 min)
   - Perguntas e dúvidas
   - Aplicações específicas
   - Desafios propostos

### Ferramentas de Apresentação

#### Opção 1: Visualizador Markdown (Recomendado)
```bash
# Usar extensão do VS Code: Markdown Preview Enhanced
# Ou visualizador online: hackmd.io, dillinger.io
# Suporta Mermaid diagrams nativamente
```

#### Opção 2: Converter para Slides
```bash
# Usando pandoc para reveajs
pandoc Aula1_Paralelismo_CPU.md -t revealjs -s -o aula1_slides.html

# Usando marp (suporta Mermaid)
npm install -g @marp-team/marp-cli
marp Aula1_Paralelismo_CPU.md --pdf

# Mermaid live editor para diagramas individuais
# https://mermaid.live/
```

#### Opção 3: Jupyter RISE (Slides no Notebook)
```bash
pip install RISE
# Usar botão "slideshow" no Jupyter
```

### 📊 Recursos Visuais Incluídos

As apresentações contêm diversos elementos visuais criados com **Mermaid.js**:

#### Tipos de Diagramas
- **Flowcharts**: Fluxos de decisão e algoritmos
- **Gantt Charts**: Comparação temporal serial vs paralelo  
- **XY Charts**: Gráficos de performance e scaling
- **Architecture Diagrams**: CPU vs GPU, memory hierarchy
- **Process Flows**: Workflows de otimização

#### Visualização Online
Os diagramas Mermaid podem ser visualizados em:
- **VS Code** com extensão Markdown Preview Enhanced
- **GitHub** (suporte nativo)
- **Mermaid Live Editor**: https://mermaid.live/
- **HackMD**, **Notion**, **GitLab** (suporte nativo)

## 🚀 Configuração do Ambiente

### Instalação com Conda (Recomendado)

```bash
# Clonar o repositório
git clone https://github.com/LucaswasTaken/Openhardware-hpc.git
cd Openhardware-hpc

# Criar ambiente conda
conda env create -f environment.yml

# Ativar ambiente
conda activate hpc-python-course

# Verificar instalação
python -c "import numpy, numba, joblib; print('✅ Instalação OK')"
```

### Instalação Manual (Alternativa)

```bash
# Criar ambiente Python
python -m venv hpc-env
source hpc-env/bin/activate  # Linux/Mac
# ou
hpc-env\Scripts\activate     # Windows

# Instalar dependências essenciais
pip install numpy scipy matplotlib pandas jupyter
pip install joblib numba psutil tqdm
pip install scikit-learn seaborn

# GPU (opcional - requer CUDA)
pip install cupy-cuda11x  # Para CUDA 11.x
# ou
pip install cupy-cuda12x  # Para CUDA 12.x
```

### Verificação da Instalação

```python
# Executar este script para verificar todas as dependências
import sys

def check_imports():
    packages = {
        'numpy': 'Computação científica',
        'scipy': 'Algoritmos científicos',
        'matplotlib': 'Visualização',
        'pandas': 'Análise de dados',
        'joblib': 'Paralelização simples',
        'numba': 'Compilação JIT',
        'psutil': 'Monitoramento sistema',
        'sklearn': 'Machine learning'
    }
    
    gpu_packages = {
        'cupy': 'NumPy para GPU',
        'numba.cuda': 'CUDA kernels'
    }
    
    print("🔍 Verificando dependências essenciais...")
    for package, desc in packages.items():
        try:
            __import__(package)
            print(f"✅ {package:12} - {desc}")
        except ImportError:
            print(f"❌ {package:12} - {desc} (FALTANDO)")
    
    print("\n🔍 Verificando dependências GPU (opcionais)...")
    for package, desc in gpu_packages.items():
        try:
            if package == 'numba.cuda':
                from numba import cuda
                print(f"✅ {package:12} - {desc}")
            else:
                __import__(package)
                print(f"✅ {package:12} - {desc}")
        except ImportError:
            print(f"⚠️  {package:12} - {desc} (OPCIONAL)")

if __name__ == "__main__":
    check_imports()
    print(f"\n🐍 Python {sys.version}")
```

## 🎯 Dependências Detalhadas

### Essenciais (Obrigatórias)
- **Python 3.9+**: Versão base
- **NumPy**: Operações vetorizadas
- **SciPy**: Algoritmos científicos
- **Matplotlib**: Visualização e plots
- **Jupyter**: Ambiente interativo
- **Joblib**: Paralelização simplificada
- **Numba**: Compilação JIT
- **Psutil**: Monitoramento do sistema

### Recomendadas
- **Pandas**: Análise de dados estruturados
- **Scikit-learn**: Machine learning
- **Seaborn**: Visualização estatística
- **Tqdm**: Barras de progresso

### Opcionais (GPU Computing)
- **CuPy**: NumPy para GPU
- **CUDA Toolkit**: Drivers GPU NVIDIA
- **Numba CUDA**: Kernels customizados

### Desenvolvimento
- **Pytest**: Testes unitários
- **Black**: Formatação de código
- **Memory Profiler**: Análise de memória
- **Line Profiler**: Profiling linha por linha

## 🔧 Solução de Problemas

### Problema: CuPy não instala
```bash
# Verificar CUDA version
nvidia-smi

# Instalar versão correta
conda install cupy cudatoolkit=11.8 -c conda-forge
# ou
pip install cupy-cuda11x  # Para CUDA 11.x
```

### Problema: Numba muito lento
```bash
# Limpar cache do Numba
export NUMBA_CACHE_DIR=/tmp/numba_cache
rm -rf ~/.numba_cache

# Verificar threading
export NUMBA_NUM_THREADS=4
```

### Problema: Out of Memory
```python
# Monitorar uso de memória
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")

# Para GPU
import cupy as cp
free, total = cp.cuda.runtime.memGetInfo()
print(f"GPU: {(total-free)/total*100:.1f}%")
```

## 📊 Performance Tips

### Configurações Otimais
```bash
# Variáveis de ambiente para performance
export OMP_NUM_THREADS=4          # OpenMP threads
export MKL_NUM_THREADS=4          # Intel MKL threads  
export OPENBLAS_NUM_THREADS=4     # OpenBLAS threads
export NUMBA_NUM_THREADS=4        # Numba threads
```

### Jupyter Configuração
```bash
# Aumentar timeout para células longas
jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10

# Usar JupyterLab para melhor performance
jupyter lab --ip=0.0.0.0 --port=8888
```

## 📞 Suporte

### Canais de Ajuda
1. **Issues no GitHub**: Para bugs e problemas técnicos
2. **Discussions**: Para dúvidas conceituais
3. **Email do instrutor**: Para questões específicas
4. **Fórum da disciplina**: Para discussões gerais

### Antes de Pedir Ajuda
1. ✅ Verificar versões das dependências
2. ✅ Executar script de verificação
3. ✅ Tentar soluções da seção "Problemas Comuns"
4. ✅ Incluir detalhes do sistema e erro completo

---

**Sucesso no aprendizado de HPC! 🚀**