# Lista de Exercícios - Computação de Alto Desempenho em Python

## 📝 Instruções Gerais

Estes exercícios são projetados para reforçar os conceitos de paralelismo apresentados nas aulas. Cada exercício explora aspectos práticos do HPC aplicados à engenharia civil.

**Diretrizes:**
- Implemente soluções tanto seriais quanto paralelas
- Meça e compare os tempos de execução
- Calcule speedup e eficiência quando aplicável
- Documente seu código com comentários claros
- Teste com diferentes tamanhos de problema

---

# 🧩 Exercícios da Aula 1 - Paralelismo em CPU

## Exercício 1.1: Produto Escalar Paralelo
**Dificuldade:** ⭐⭐☆☆☆

Implemente o cálculo do produto escalar (dot product) de dois vetores usando multiprocessing.

```python
# Fórmula: dot_product = Σ(a[i] * b[i])
```

**Tarefas:**
1. Implemente versão serial
2. Implemente versão paralela dividindo os vetores em chunks
3. Compare com `np.dot()`
4. Teste com vetores de tamanhos: 1M, 10M, 100M elementos

**Aplicação:** Cálculo de forças internas em estruturas (F = K·u).

---

## Exercício 1.2: Análise de Vigas Paralela
**Dificuldade:** ⭐⭐⭐☆☆

Simule a análise de múltiplas vigas com diferentes carregamentos em paralelo.

**Especificações:**
- Viga simplesmente apoiada, comprimento L
- Carga concentrada P no centro
- Calcular: deflexão máxima, momento máximo, tensão máxima
- Simular 10.000 vigas com parâmetros aleatórios

**Fórmulas:**
```
δ_max = P*L³/(48*E*I)    # Deflexão máxima
M_max = P*L/4            # Momento máximo  
σ_max = M*c/I            # Tensão máxima
```

**Tarefas:**
1. Implemente função para análise de uma viga
2. Paralelizar usando `ProcessPoolExecutor`
3. Gerar estatísticas: média, desvio padrão, histogramas
4. Comparar tempo serial vs paralelo

**Aplicação:** Análise paramétrica para dimensionamento estrutural.

---

## Exercício 1.3: Monte Carlo para Área de Seção
**Dificuldade:** ⭐⭐⭐☆☆

Use Monte Carlo para calcular a área de uma seção transversal complexa (ex: seção T).

**Especificações:**
- Seção T: mesa 20cm × 5cm, alma 5cm × 15cm
- Usar pontos aleatórios em retângulo envolvente
- Contar pontos dentro da seção

**Tarefas:**
1. Implementar teste de ponto dentro da seção T
2. Versão serial do Monte Carlo
3. Versão paralela com múltiplos processos
4. Calcular área, centroide e momento de inércia
5. Estudar convergência com número de amostras

**Aplicação:** Cálculo de propriedades geométricas de seções complexas.

---

## Exercício 1.4: Solução de Sistema Linear Paralelo
**Dificuldade:** ⭐⭐⭐⭐☆

Implemente eliminação de Gauss paralela para sistemas lineares.

**Especificações:**
- Sistema Ax = b, onde A é matriz n×n
- Paralelizar operações de linha da eliminação
- Implementar substituição regressiva

**Tarefas:**
1. Implementar eliminação de Gauss serial
2. Paralelizar usando divisão por linhas
3. Comparar com `np.linalg.solve()`
4. Testar estabilidade numérica
5. Medir speedup para diferentes tamanhos de matriz

**Aplicação:** Solução de sistemas de equações estruturais (Ku = f).

---

# ⚙️ Exercícios da Aula 2 - Paralelismo Avançado

## Exercício 2.1: Interpolação Paralela com Joblib
**Dificuldade:** ⭐⭐☆☆☆

Implemente interpolação de dados experimentais usando múltiplos métodos em paralelo.

**Especificações:**
- Dados de teste de materiais (tensão × deformação)
- Métodos: linear, spline cúbica, polinomial
- Avaliar em 1000 pontos de consulta

**Tarefas:**
1. Gerar dados sintéticos de teste de tração
2. Implementar cada método de interpolação
3. Usar `joblib` para paralelizar diferentes métodos
4. Comparar precisão e tempo de execução
5. Visualizar resultados

**Aplicação:** Processamento de dados experimentais de materiais.

---

## Exercício 2.2: Otimização Paramétrica com Futures
**Dificuldade:** ⭐⭐⭐☆☆

Otimize as dimensões de uma viga para minimizar peso sujeito a restrições.

**Especificações:**
- Viga retangular: altura h, base b
- Restrições: tensão ≤ 250 MPa, deflexão ≤ L/250
- Objetivo: minimizar peso (área da seção)

**Tarefas:**
1. Implementar função objetivo e restrições
2. Usar busca em grade paralela com `concurrent.futures`
3. Encontrar solução ótima
4. Plotar superfície de resposta
5. Comparar com algoritmos de otimização clássicos

**Aplicação:** Dimensionamento ótimo de elementos estruturais.

---

## Exercício 2.3: Análise Modal com Numba
**Dificuldade:** ⭐⭐⭐⭐☆

Calcule frequências naturais de uma viga usando método de diferenças finitas.

**Especificações:**
- Viga engastada-livre discretizada em elementos
- Problema de autovalores: (K - ω²M)φ = 0
- Usar Numba para acelerar montagem de matrizes

**Tarefas:**
1. Implementar montagem de matriz de rigidez K
2. Implementar montagem de matriz de massa M  
3. Usar `@numba.jit` para acelerar loops
4. Comparar primeiras 5 frequências com solução analítica
5. Estudar convergência com refinamento da malha

**Aplicação:** Análise dinâmica de estruturas.

---

## Exercício 2.4: Scaling Study Completo
**Dificuldade:** ⭐⭐⭐⭐☆

Conduza estudo completo de escalabilidade para multiplicação matriz-vetor.

**Especificações:**
- Matrizes esparsas (padrão pentadiagonal)
- Strong scaling: matriz fixa, variar processos
- Weak scaling: elementos por processo fixo

**Tarefas:**
1. Implementar multiplicação matriz-vetor esparsa
2. Medir strong scaling (1-16 processos)
3. Medir weak scaling (mesmo workload por processo)
4. Calcular eficiência e identificar gargalos
5. Plotar curvas de escalabilidade
6. Propor melhorias baseadas nos resultados

**Aplicação:** Dimensionamento de clusters para problemas FEM.

---

# ⚡ Exercícios da Aula 3 - Computação GPU

## Exercício 3.1: Filtro de Imagens com CuPy
**Dificuldade:** ⭐⭐⭐☆☆

Implemente filtros para processamento de imagens de inspeção estrutural.

**Especificações:**
- Filtros: Gaussiano, Sobel, Laplaciano
- Imagens de fissuras em concreto (simular com ruído)
- Comparar performance CPU vs GPU

**Tarefas:**
1. Gerar imagem sintética com "fissuras"
2. Implementar filtros usando NumPy
3. Portar para CuPy (mudança np → cp)
4. Medir speedup para diferentes tamanhos de imagem
5. Visualizar resultados da filtragem

**Aplicação:** Detecção automática de fissuras em estruturas.

---

## Exercício 3.2: FFT para Análise de Vibrações
**Dificuldade:** ⭐⭐⭐☆☆

Use FFT em GPU para analisar sinais de acelerômetros em estruturas.

**Especificações:**
- Sinal simulado: frequências estruturais + ruído
- Calcular espectro de potência
- Identificar picos (frequências naturais)

**Tarefas:**
1. Gerar sinal temporal com múltiplas frequências
2. Implementar FFT usando NumPy e CuPy
3. Calcular densidade espectral de potência
4. Identificar picos automaticamente
5. Comparar performance para sinais longos

**Aplicação:** Monitoramento de saúde estrutural (SHM).

---

## Exercício 3.3: Solver Iterativo GPU
**Dificuldade:** ⭐⭐⭐⭐☆

Implemente método de Jacobi para solução de sistemas lineares em GPU.

**Especificações:**
- Sistema Ax = b com matriz esparsa
- Método iterativo de Jacobi
- Critério de convergência: ||r|| < tol

**Tarefas:**
1. Implementar Jacobi usando CuPy
2. Comparar com versão CPU (NumPy)
3. Estudar convergência para diferentes matrizes
4. Otimizar usando memory patterns eficientes
5. Implementar pré-condicionamento diagonal

**Aplicação:** Solução de grandes sistemas FEM em GPU.

---

## Exercício 3.4: Simulação de Onda 2D com Numba CUDA
**Dificuldade:** ⭐⭐⭐⭐⭐

Simule propagação de ondas sísmicas em meio 2D usando kernels CUDA.

**Especificações:**
- Equação da onda: ∂²u/∂t² = c²∇²u
- Diferenças finitas no tempo e espaço
- Fonte pontual de excitação

**Tarefas:**
1. Implementar kernel CUDA para um passo temporal
2. Implementar condições de contorno absorventes
3. Visualizar propagação da onda (animação)
4. Comparar com versão CPU
5. Medir throughput (pontos de malha por segundo)

**Aplicação:** Simulação de propagação de ondas sísmicas.

---

# 🎯 Projetos Integradores

## Projeto Final A: Simulador FEM Paralelo
**Dificuldade:** ⭐⭐⭐⭐⭐

Desenvolva um simulador de elementos finitos paralelo para treliças 2D.

**Especificações:**
- Elementos de barra (2 nós, 4 graus de liberdade)
- Montagem paralela da matriz global
- Solução paralela do sistema
- Pós-processamento das tensões

**Entregáveis:**
1. Código completo documentado
2. Validação com casos analíticos
3. Análise de performance e escalabilidade
4. Relatório técnico (5-10 páginas)

---

## Projeto Final B: Plataforma de Análise Sísmica
**Dificuldade:** ⭐⭐⭐⭐⭐

Desenvolva plataforma para análise de registros sísmicos.

**Especificações:**
- Processamento de acelerogramas
- Cálculo de espectros de resposta
- Análise estatística de banco de dados
- Interface de visualização

**Entregáveis:**
1. Pipeline completo de processamento
2. Comparação CPU vs GPU
3. Análise de banco de dados sísmicos
4. Dashboard de visualização

---

# 📊 Critérios de Avaliação

## Exercícios Individuais (70%)
- **Correção (40%):** Implementação funciona corretamente
- **Performance (20%):** Speedup e eficiência adequados  
- **Código (10%):** Clareza, documentação, estrutura

## Projeto Final (30%)
- **Funcionalidade (15%):** Atende especificações
- **Inovação (5%):** Soluções criativas e otimizações
- **Relatório (10%):** Análise técnica e conclusões

## Dicas para Sucesso
1. **Comece simples:** Implemente versão serial primeiro
2. **Meça sempre:** Profile antes de otimizar
3. **Documente:** Comente código e resultados
4. **Valide:** Compare com soluções conhecidas
5. **Explore:** Teste diferentes parâmetros e configurações

---

## 📚 Recursos Adicionais

### Dados para Exercícios
- Repositório contém datasets sintéticos
- Scripts de geração de dados incluídos
- Soluções analíticas para validação

### Ferramentas Recomendadas
- **Profiling:** `cProfile`, `line_profiler`
- **Visualização:** `matplotlib`, `seaborn`
- **Análise:** `pandas`, `scipy`
- **GPU:** `cupy`, `numba`

### Suporte
- Issues no repositório GitHub
- Discussões no fórum da disciplina
- Horários de monitoria

**Boa sorte e bom aprendizado em HPC! 🚀**