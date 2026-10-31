# Roteiro do professor — como ministrar as 2 aulas

Guia prático para dar o curso **Computação de Alto Desempenho em Python para Engenharia Civil** em **2 aulas presenciais de 3h30**. Para cada bloco: objetivo, o que fazer na sala, o que a turma deve ver nos números, perguntas para a turma e o que cortar se o tempo apertar.

Materiais:
- notebooks: [`notebooks/Aula1_CPU.ipynb`](notebooks/Aula1_CPU.ipynb) e [`notebooks/Aula2_GPU.ipynb`](notebooks/Aula2_GPU.ipynb) (abrem no Colab pelos botões do README);
- slides (Marp): [`presentations/`](presentations/);
- vídeos curtos (~1 min): [lucaswastaken.github.io/Openhardware-hpc](https://lucaswastaken.github.io/Openhardware-hpc/);
- exercícios: [`exercicios/lista.md`](exercicios/lista.md), [rubrica](exercicios/lista.md#rubrica-10-pontos) e [modelo de relatório](exercicios/modelo_relatorio.ipynb).

---

## Antes do curso (1 semana antes)

- [ ] **Rode os dois notebooks no Colab** de ponta a ponta, com a mesma conta que vai usar na sala. A Aula 2 com *T4 GPU*.
- [ ] **Salve os resultados do Colab** (`results/aula1_colab.json`, `results/aula2_colab.json`) no repositório e regere figuras e vídeos — assim slides e vídeos mostram os números da máquina da turma:
  ```bash
  python presentations/figures/make_figures.py
  python videos/render.py && python tools/build_site.py
  ```
- [ ] **Avise a turma:** conta Google para o Colab; assistir aos vídeos 01 e 02 antes da Aula 1 (2 min no total).
- [ ] **Projetor:** teste o tema escuro dos vídeos e o tamanho da fonte do Colab (Ctrl + `+` duas vezes).
- [ ] **Plano B de GPU:** o Colab gratuito às vezes nega a T4 em horário de pico. Tenha uma execução já salva (o notebook publicado vem **executado**, com saídas) para mostrar os resultados mesmo sem GPU.

## Dicas gerais

- **Ritmo:** cada bloco = 3–5 min de conceito (slide ou vídeo) → rodar as células junto com a turma → 1–2 perguntas do "🧠 Teste rápido".
- **Vídeos em sala:** use-os como abertura de bloco quando a turma estiver cansada (após o intervalo, por exemplo). Fora isso, indique-os para revisão.
- **Números reais mudam:** o Colab gratuito tem **2 núcleos**, então speedups de CPU ficam perto de 2×. Diga isso **antes** de rodar — vira lição, não frustração. Peça a quem tem notebook com mais núcleos para rodar localmente e comparar.
- **Pergunta-âncora do curso:** *"onde está o gargalo?"* — volte a ela em todo bloco.

---

## Aula 1 — CPU (3h30)

| Tempo | Bloco | Material |
|---|---|---|
| 0:00–0:20 | 1. Medir antes de otimizar | slides 1–4 · vídeo 01 · seção 1 do notebook |
| 0:20–0:50 | 2. Speedup, Amdahl, Gustafson | slides 5–6 · vídeo 02 · seção 2 |
| 0:50–1:30 | 3. Threads × processos, GIL, Python sem GIL | slides 7–10 · vídeos 03 e 04 · seção 3 |
| 1:30–1:45 | ☕ Intervalo | deixe o vídeo 05 tocando no fim |
| 1:45–2:30 | 4. BLAS e `joblib` | slides 11–12 · vídeo 05 · seção 4 |
| 2:30–3:10 | 5. Numba | slides 13–14 · vídeo 06 · seção 5 |
| 3:10–3:30 | 6. Exercício guiado e lista | slides 15–16 · vídeo 11 · seção 6 |

### 0:00–0:20 · 1. Medir antes de otimizar
**Objetivo:** a turma entende que paralelizar começa por **medir** e que vetorizar costuma render mais que paralelizar.
1. (3 min) Abertura: pergunte "que cálculo da sua área demora?" — anote 3–4 respostas no quadro (casos de carga, Monte Carlo, malhas). Volte a elas no fim do curso.
2. (2 min) Abra o notebook no Colab pelo botão, rode a célula de preparação: mostre que o Colab tem 2 núcleos.
3. (10 min) Envoltória de momentos: `perf_counter` → `%timeit` → `cProfile` → `%lprun`. Pergunte *antes* de mostrar o perfil: "onde vocês acham que está o tempo?".
4. (5 min) Versão NumPy: ~10× no mesmo núcleo. Frase-chave: **"primeiro use bem um núcleo"**.

**Esperado:** Python ~0,3 s, NumPy ~30 ms (varia com a máquina).
**Se atrasar:** pule o `%lprun` (o `cProfile` já mostra a ideia).

### 0:20–0:50 · 2. Speedup, eficiência, Amdahl e Gustafson
**Objetivo:** calcular speedup e eficiência e entender por que a parte serial limita tudo.
1. (8 min) Fórmulas no slide; desenhe no quadro uma barra 10% serial + 90% paralela e "encolha" a parte paralela para p = 2, 4, ∞.
2. (7 min) Rode o gráfico de Amdahl/Gustafson; peça para lerem "com 95% paralelo, qual o máximo?" (20×).
3. (10 min) **Miniexercício** (Karp–Flatt) em duplas, depois rode a célula com a resposta. Discuta: fração serial que **cresce** com p = overhead.
4. (5 min) Escalabilidade forte × fraca: "no seu TCC, o que vocês fariam com o dobro de núcleos? Mesma malha mais rápida ou malha mais fina?"

**Se atrasar:** deixe Gustafson só no slide e indique o vídeo 02.

### 0:50–1:30 · 3. Threads × processos, GIL e Python sem GIL
**Objetivo:** escolher entre threads e processos e saber o que o Python 3.14t muda.
1. (5 min) Slide da tabela threads × processos; analogia do GIL = **um microfone numa reunião**.
2. (7 min) Célula de threads: laço CPU-bound (~1×) e espera/I/O (~4×). Pergunte antes de rodar: "vai acelerar?".
3. (15 min) Monte Carlo de confiabilidade: explique $P_f = P(R < S)$ com as curvas do vídeo 04 (ou desenhe). Mostre o `%%writefile` e explique *spawn* (Windows/macOS). Rode com 1, 2… processos.
4. (10 min) Python sem GIL: a célula baixa o Python 3.14 e 3.14t com `uv` (~30–60 s no Colab — **comece a rodar e explique enquanto baixa**). Resultado esperado: 3.14 ≈ 1×, 3.14t ≈ 1,8–2,6×.
5. (3 min) Mensagem: sem GIL ≠ mágica — corridas de dados, bibliotecas, e para cálculo numérico o caminho principal segue sendo NumPy/Numba/GPU.

**Esperado no Colab:** Monte Carlo ~1,7–1,9× com 2 processos.
**Se atrasar:** rode a demo do 3.14t só pelo professor, sem a turma acompanhar.

### 1:45–2:30 · 4. Vetorização, BLAS e `joblib`
**Objetivo:** perceber que o NumPy já é paralelo e paralelizar um estudo paramétrico sem sobreassinatura.
1. (8 min) `threadpool_info` e `A @ A` com 1 thread × todas. Pergunta: "vocês já usavam vários núcleos sem saber?".
2. (5 min) Mostre `viga_mef.py` e a conferência MEF × analítico (mesma flecha) — **correção antes de velocidade**.
3. (15 min) `joblib` com `inner_max_num_threads=1`. Desenhe a sobreassinatura no quadro (4 processos × 4 threads em 4 núcleos).
4. (7 min) Gráfico da altura mínima × vão, comparado com a regra prática L/12: o resultado de engenharia que o paralelismo destravou.

**Se atrasar:** reduza `casos` (menos alturas) na célula do `joblib`.

### 2:30–3:10 · 5. Numba
**Objetivo:** acelerar laços com lógica e paralelizar com `prange`.
1. (5 min) Slide "como o Numba funciona" (ou vídeo 06): compilação na 1ª chamada, `cache=True`.
2. (20 min) Treliça com 2 milhões de barras: leia `rigidez_python` com a turma, depois NumPy, Numba e Numba paralelo. Esperado (Colab): Numba ~100×, paralelo ~150–200× sobre Python.
3. (5 min) Montagem COO → CSR: por que **não** usar `prange` com `K[i, j] +=` (escrita concorrente).
4. (10 min) Teste rápido em voz alta.

### 3:10–3:30 · 6. Exercício guiado e lista
1. (10 min) Exercício das tensões em 1 milhão de seções: deixe tentarem; a célula "💡 Solução" está logo abaixo — peça para não rodar antes.
2. (8 min) Apresente a lista: **um** exercício, **uma** tecnologia, relatório de 1–2 páginas. Mostre a rubrica e o modelo de relatório. Defina o prazo.
3. (2 min) Para a próxima aula: assistir ao vídeo 07 (CPU × GPU).

---

## Aula 2 — GPU (3h30)

| Tempo | Bloco | Material |
|---|---|---|
| 0:00–0:25 | 1. CPU × GPU | slides 1–4 · vídeo 07 · seção 1 |
| 0:25–0:50 | 2. CPU e GPU ainda são mundos separados? | slides 5–7 · vídeo 08b · seção 2 |
| 0:50–1:05 | 3. CuPy e como medir direito | slides 8–9 · vídeo 08 · seção 3 |
| 1:05–1:45 | 4. Kernels | slide 10 · vídeo 09 · seção 4 |
| 1:45–2:00 | ☕ Intervalo | |
| 2:00–3:00 | 5. Projeto: equação do calor | slides 11–12 · vídeo 10 · seção 5 |
| 3:00–3:30 | 6. Quando usar GPU? Exercícios | slides 13–14 · seção 6 |

**Antes de começar:** peça que todos troquem o ambiente para **T4 GPU** e rodem a célula de preparação — a fila de GPU do Colab pode levar alguns minutos. Quem não conseguir GPU acompanha pelo notebook publicado (com saídas) ou em dupla.

### 0:00–0:25 · 1. CPU × GPU
1. (8 min) Tabela CPU × GPU; analogia: 8 engenheiros sêniores × obra com 2 500 operários.
2. (7 min) Warps e divergência: peça a 8 alunos que "executem a mesma instrução" levantando a mão; depois um `if` divide o grupo — metade espera. (Funciona muito bem.)
3. (10 min) Célula das propriedades da GPU: núcleos, memória, banda teórica.

### 0:25–0:50 · 2. CPU e GPU ainda são mundos separados?
**Objetivo:** entender memória unificada sem cair no marketing — a pergunta passa a ser *onde o dado mora e quem o processa melhor*.
1. (5 min) Comece com o **vídeo 08b** (2 min) — ele apresenta os três modelos de hardware.
2. (5 min) Slide dos três modelos: GPU discreta (T4), coerente com link rápido (Grace Hopper/Blackwell, DGX Spark), memória física única (Apple M, MI300A do El Capitan, Ryzen AI Max). Pergunte: "quem aqui tem Mac com chip M?" — ele já usa memória unificada.
3. (10 min) Demo: cópia explícita × memória gerenciada × CPU tocando os dados × prefetch. Leitura esperada: 1º kernel gerenciado lento (migração), 2º rápido, toque da CPU traz tudo de volta, `prefetch` recupera.
4. (5 min) O que **não** muda: paralelismo de dados, banda, localidade. Frase final no quadro.

**Atenção:** no Colab (Linux) há migração sob demanda e `prefetch`; em Windows local, não — os números mudam, e isso é parte da lição.

### 0:50–1:05 · 3. CuPy e como medir direito
1. (5 min) As três regras: aquecer, sincronizar, separar transferência. Mostre o que acontece **sem** sincronizar (tempo absurdo de rápido).
2. (5 min) Soma de vetores: GPU vence só na conta e perde com as cópias.
3. (5 min) Matmul: GFLOP/s, `float32` × `float64`. Pergunte: "quais problemas de vocês aceitam `float32`?".

### 1:05–1:45 · 4. Kernels
1. (10 min) Grid, block, thread no quadro (ou vídeo 09). Índice global: bloco 2, thread 3 → i = 19.
2. (10 min) `ElementwiseKernel` de von Mises: 1 passada na memória × expressão com temporários.
3. (15 min) Monte Carlo com Numba-CUDA: leia o kernel linha a linha; o `if i < saida.size`; gerador por thread.
4. (5 min) Teste rápido.

**Se atrasar:** mostre o von Mises só rodando, sem detalhar a sintaxe do kernel em C.

### 2:00–3:00 · 5. Projeto: equação do calor numa laje
**Objetivo:** levar um problema de engenharia do Python puro à GPU e interpretar escalabilidade.
1. (10 min) Física e discretização: média dos vizinhos, estabilidade $r \le 0{,}25$. Mostre o vídeo 10 se a turma estiver cansada.
2. (10 min) Leia a função `simula` (Array API): **o mesmo código** roda em NumPy e CuPy.
3. (5 min) Kernel 2D com Numba-CUDA (grade 16×16).
4. (20 min) Rode a comparação por tamanho de malha (~1–2 min no Colab). Enquanto roda, peça palpites: "a partir de que malha a GPU ganha?". Leia o gráfico: pequena → GPU não ganha; grande → dispara; kernel próprio > fatias.
5. (10 min) Mapa de temperatura da laje e discussão de engenharia (isoterma de 500 °C e armadura, por exemplo). **Para casa:** convecção na face superior ou $r > 0{,}25$.
6. (5 min) Fecho: problema limitado por memória → ganha quem tem mais banda.

### 3:00–3:30 · 6. Quando usar GPU? Exercícios
1. (10 min) Checklist em 5 perguntas; aplique a 2–3 problemas que a turma citou no começo da Aula 1.
2. (10 min) Exercícios: tire dúvidas de escolha; mostre os vídeos 12–14 (exemplos **parecidos**, não iguais à lista).
3. (10 min) Encerramento: o que levar — medir, encontrar o gargalo, escolher a ferramenta.

---

## Depois das aulas

- **Prazo sugerido** do relatório: 2 semanas. Corrija com a [rubrica](exercicios/lista.md#rubrica-10-pontos).
- **Erros comuns nos relatórios:** medir sem aquecer; esquecer de sincronizar a GPU; comparar com Python puro e não com NumPy; speedup "superlinear" por cache sem explicação; usar a mesma semente em todos os processos.
- **Atualizar o curso:** edite `tools/build_aula1.py`/`build_aula2.py`, rode `python tools/run_nb.py notebooks/…`, e regere figuras e vídeos. A CI do GitHub executa os notebooks em CPU a cada mudança.
