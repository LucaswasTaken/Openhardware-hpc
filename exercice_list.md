# Lista de Exercícios - Computação de Alto Desempenho em Python
## Escolha UM Exercício e uma Tecnologia

## 📝 Instruções Gerais

Você deve **escolher apenas UM exercício** desta lista e **uma tecnologia** para implementar a solução. Após a implementação, gere um **relatório técnico** (1-2 páginas) com explicações detalhadas e análise completa de performance.

### 🚀 Tecnologias Disponíveis:
- **Joblib**: Paralelização em CPU (múltiplos cores)
- **Numba**: Compilação JIT para performance (código serial otimizado)  
- **CuPy**: Computação em GPU (aceleração massiva)
- **Multiprocessing**: Paralelização tradicional Python
- **Dask**: Computação paralela distribuída
- **Ou qualquer outra tecnologia de HPC que desejar explorar**

### 📋 Estrutura do Relatório:
1. **Problema Escolhido** (2-3 linhas)
2. **Tecnologia Selecionada e Justificativa** (3-4 linhas)
3. **Implementação**: Descrição detalhada da abordagem (4-5 linhas)
4. **Métricas de Performance Obrigatórias**: 
   - Tempo de execução (serial vs paralelo)
   - **Speedup** = T_serial / T_paralelo
   - **Eficiência** = Speedup / N_processos
   - **Escalabilidade** = análise com diferentes números de cores/processos
   - Uso de memória e recursos
5. **Análise dos Resultados** (4-5 linhas)
6. **Gráficos de Escalabilidade** (speedup vs cores, eficiência vs cores)
7. **Conclusões e Limitações** (3-4 linhas)

---

# 🎯 Escolha UM dos 5 Exercícios Abaixo
**Nível de Dificuldade:** ⭐⭐☆☆☆ (Básico-Intermediário)

# 🔢 Exercício 1: Produto Vetorial (Dot Product)
**Aplicação:** Cálculo de forças e energias em sistemas físicos

### Especificações:
- Implementar produto escalar de dois vetores: result = Σ(a[i] * b[i])
- Tamanhos de vetores: 1 milhão, 10 milhões, 100 milhões de elementos
- Comparar com implementação numpy (np.dot)
- Testar escalabilidade com 1, 2, 4, 8 cores/processos

### Requisitos Técnicos:
- Implementar versão serial simples (loop básico)
- Dividir vetor em chunks para paralelização
- Medir tempo total e tempo por elemento
- Validar resultado comparando com numpy

### Análise Obrigatória:
- Strong scaling: vetor fixo, variar processos
- Identificar overhead de paralelização
- Comparar eficiência vs numpy otimizado

---

# 🧮 Exercício 2: Multiplicação de Matrizes Densas
**Aplicação:** Operações básicas em álgebra linear computacional

### Especificações:
- Implementar multiplicação C = A × B (matrizes densas quadradas)
- Tamanhos: 500×500, 1000×1000, 1500×1500
- Algoritmo triplo loop básico: C[i][j] = Σ A[i][k] * B[k][j]
- Testar diferentes estratégias de paralelização

### Requisitos Técnicos:
- Versão serial com loops aninhados
- Paralelizar por linhas, colunas ou blocos
- Medir FLOPS (operações de ponto flutuante por segundo)
- Comparar com numpy.matmul()

### Análise Obrigatória:
- Escalabilidade vs tamanho da matriz
- Eficiência de diferentes estratégias de divisão
- Análise de uso de cache e memória

---

# � Exercício 3: Simulação Monte Carlo para π (joblib ou cupy)
**Aplicação:** Métodos probabilísticos e integração numérica

### Especificações:
- Calcular π usando pontos aleatórios em círculo unitário
- π ≈ 4 × (pontos dentro do círculo / total de pontos)
- Números de pontos: 1M, 10M, 100M, 1B
- Medir convergência e erro relativo

### Requisitos Técnicos:
- Gerar pontos (x,y) aleatórios no quadrado [-1,1]×[-1,1]
- Testar se x² + y² ≤ 1 (dentro do círculo)
- Paralelizar geração e contagem de pontos
- Calcular estatísticas de convergência

### Análise Obrigatória:
- Weak scaling: pontos por processo constante
- Erro vs número de amostras (lei dos grandes números)
- Qualidade dos geradores de números aleatórios

---

# 📊 Exercício 4: Soma de Elementos de Array (cupy obrigatorio)
**Aplicação:** Operação de redução fundamental em computação paralela

### Especificações:
- Calcular soma de todos elementos de um array grande
- Tamanhos: 10M, 50M, 100M, 500M elementos
- Implementar diferentes estratégias de redução
- Comparar com numpy.sum()
- Verificar se em algum momente HAVERÁ GANHO EM PARALELIZAR EM GPU

### Requisitos Técnicos:
- Versão serial: loop simples
- Redução paralela: árvore binária ou divisão em chunks
- Evitar problemas de precisão numérica
- Medir bandwidth de memória

### Análise Obrigatória:
- Escalabilidade limitada por memória vs CPU
- Comparar redução em árvore vs chunks lineares
- Análise de precisão numérica (float32 vs float64)

---

# 🔍 Exercício 5: Busca Linear em Array
**Aplicação:** Busca paralela e processamento de dados

### Especificações:
- Encontrar todas ocorrências de um valor em array grande
- Tamanhos: 10M, 50M, 100M elementos
- Retornar índices de todas as ocorrências encontradas
- Testar com diferentes densidades de ocorrências (1%, 5%, 10%)

### Requisitos Técnicos:
- Versão serial: loop com comparação simples
- Paralelizar busca dividindo array em chunks
- Combinar resultados de diferentes processos
- Medir throughput (elementos processados por segundo)

### Análise Obrigatória:
- Escalabilidade vs densidade de ocorrências
- Overhead de comunicação para combinar resultados
- Load balancing quando ocorrências são irregulares

---

# 🔍 Exercício 6: Ordenação
**Aplicação:** Qualquer algoritmo de ordenação com qualquer método de HPC

### Especificações:
- Sem especificações

### Requisitos Técnicos:
- Versão serial: loop com comparação simples
- Paralelizar busca dividindo array em chunks
- Combinar resultados de diferentes processos
- Medir throughput (elementos processados por segundo)

### Análise Obrigatória:
- Escalabilidade vs densidade de ocorrências
- Overhead de comunicação para combinar resultados
- Load balancing quando ocorrências são irregulares

---

# 🔍 Exercício 7: Algoritmo ponto dentor de polígono
**Aplicação:** Algoritmo do Tiro com qualquer método de HPC

### Especificações:
- Sem especificações

### Requisitos Técnicos:
- Versão serial: loop com comparação simples
- Paralelizar busca dividindo array em chunks
- Combinar resultados de diferentes processos
- Medir throughput (elementos processados por segundo)

### Análise Obrigatória:
- Escalabilidade vs densidade de ocorrências
- Overhead de comunicação para combinar resultados
- Load balancing quando ocorrências são irregulares

---

# 🔍 Exercício 8: Fecho Convexo
**Aplicação:** Algoritmo de fecho convexo com qualquer método de HPC

### Especificações:
- Sem especificações

### Requisitos Técnicos:
- Versão serial: loop com comparação simples
- Paralelizar busca dividindo array em chunks
- Combinar resultados de diferentes processos
- Medir throughput (elementos processados por segundo)

### Análise Obrigatória:
- Escalabilidade vs densidade de ocorrências
- Overhead de comunicação para combinar resultados
- Load balancing quando ocorrências são irregulares

---

# 🎨 Exercício Extra: Problema Proposto pelo Estudante

### 💡 Oportunidade de Criar Seu Próprio Desafio

Se você deseja explorar um problema específico de sua área de interesse ou tem uma aplicação particular em mente, pode **propor seu próprio exercício**!

### Requisitos para Proposta:
1. **Problema Bem Definido**: Descrição clara do problema computacional
2. **Relevância**: Aplicação prática em engenharia, ciências ou computação
3. **Escalabilidade**: Problema deve ser paralelizável/otimizável
4. **Complexidade Adequada**: Nem trivial nem excessivamente complexo

### Exemplos de Problemas Válidos:
- **Processamento de Imagens**: Filtros, segmentação, análise de features
- **Simulação Física**: Dinâmica de fluidos, mecânica dos sólidos, ondas
- **Análise de Dados**: Machine learning, estatística, big data
- **Algoritmos Numéricos**: Solvers, otimização, álgebra linear

### 📧 Como Submeter a Proposta:
Envie por email ou fórum da disciplina com assunto: **"Proposta de Exercício HPC - [Seu Nome]"**

---

**Boa sorte na exploração de HPC! 🚀**