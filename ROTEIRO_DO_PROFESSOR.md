# Roteiro do professor — como ministrar as 2 aulas

Guia prático para dar o curso **Computação de Alto Desempenho em Python para Engenharia Civil** em **2 aulas presenciais de 3h30**. Para cada bloco: objetivo, o que fazer na sala, o que a turma deve ver nos números, perguntas para a turma e o que cortar se o tempo apertar.

Materiais:
- notebooks: [`notebooks/Aula1_CPU.ipynb`](notebooks/Aula1_CPU.ipynb) e [`notebooks/Aula2_GPU.ipynb`](notebooks/Aula2_GPU.ipynb) (abrem no Colab pelos botões do README);
- slides (Marp): [`presentations/`](presentations/);
- vídeos curtos (~1 min): [lucaswastaken.github.io/Openhardware-hpc](https://lucaswastaken.github.io/Openhardware-hpc/);
- exercícios: [`exercicios/lista.md`](exercicios/lista.md), [rubrica](exercicios/lista.md#rubrica-10-pontos), [modelo de relatório](exercicios/modelo_relatorio.ipynb) e [guia de instalação](exercicios/INSTALACAO.md);
- **vídeo-roteiro da Aula 2** (só para o professor): `python videos/professor.py` gera `videos/professor/aula2_professor.mp4`.

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


## Aula 2 — GPU na prática (3h30)

**Ideia da aula, em uma frase:** *a GPU são milhares de operários fazendo a mesma conta; ela só ganha quando há muitos dados e quando os dados ficam lá.* A ferramenta é uma só, o **CuPy** (`np` → `cp`). O resto (kernels, memória gerenciada, PyTorch…) fica no apêndice ou só é citado.

**Público:** engenheiros que não programam e que vão pedir ajuda à IA. Não ensine sintaxe: mostre **o que observar** em cada resultado. Cada célula do notebook já traz "▶ O que esta célula faz / 👀 O que observar". Leia essa caixa em voz alta e rode.

| Tempo | Bloco | Slides | Vídeo | Notebook |
|---|---|---|---|---|
| 0:00–0:20 | 1. Por que GPU? | 1–3 | 07 | seção 1 |
| 0:20–0:50 | 2. Dois mundos: RAM e VRAM | 4–6 | 08, 08b | seção 2 |
| 0:50–1:30 | 3. CuPy em 3 passos | 7–9 | 09 | seção 3 |
| 1:30–1:45 | ☕ Intervalo | — | deixe o 10 tocando | — |
| 1:45–2:45 | 4. Projeto: laje em incêndio | 10–11 | 10 | seção 4 |
| 2:45–3:15 | 5. O seu exercício na GPU | 12–13 | 13, 14 | seção 5 |
| 3:15–3:30 | 6. Quando usar GPU + outras ferramentas | 14–15 | 11 | seção 6 |

### Antes de começar (10 min antes)
- [ ] Abra no projetor, em abas: o **notebook** (`notebooks/Aula2_GPU.ipynb`, botão Colab do README), os **slides** (`presentations/Aula2_GPU.md`, Marp), a **página dos vídeos** e a **lista** (`exercicios/lista.md`).
- [ ] No Colab: *Ambiente de execução → Alterar o tipo → T4 GPU*. Rode a 1ª célula: deve dizer `GPU: True`.
- [ ] Peça à turma para fazer o mesmo **logo na chegada** (a fila de GPU pode levar minutos). Quem não conseguir: segue em dupla ou pelo notebook publicado, que já vem executado.
- [ ] Plano B: se o Colab negar GPU para todos, projete o notebook publicado (já tem as saídas) e rode só as células de CPU.

### 0:00–0:20 · 1. Por que GPU? Milhares de operários
**Mensagem:** CPU = 8 engenheiros experientes; GPU = 2 500 operários que fazem **a mesma tarefa ao mesmo tempo**.
1. (3 min) Vídeo 07, ou conte a analogia no slide 3.
2. (5 min) Pergunte: "quais contas do trabalho de vocês são *a mesma conta em milhões de dados*?" Anote 2–3 respostas (tensão em pontos de integração, nós de malha, amostras de Monte Carlo) e volte a elas no bloco 6.
3. (5 min) Seção 1 do notebook: rode a célula da GPU. **Esperado no Colab:** Tesla T4, 2 560 núcleos, ~15 GB, contra 2 núcleos de CPU.
4. (5 min) Pergunta-armadilha: "então a GPU é 1 000× mais rápida?" Não: cada operário é lento, e eles precisam de material (dados). É o gancho para o bloco 2.

### 0:20–0:50 · 2. Dois mundos: a memória do computador e a da GPU
**Mensagem:** **leve os dados uma vez.** A viagem RAM → GPU costuma custar mais que a conta.
1. (5 min) Slide 4 (diagrama). Analogia: o depósito do outro lado da cidade, e o caminhão é o cabo PCIe.
2. (10 min) Seção 2, célula da viagem × conta. **Peça palpites antes de rodar:** "o que demora mais, levar 400 MB ou multiplicar 50 milhões de números?" **Esperado (T4, aproximado):** ida ~30–40 ms, conta leve ~2–4 ms, conta pesada ~10–20 ms, volta ~30–40 ms. A ida custa ~10× a conta leve.
3. (10 min) Vídeo 08b + slide 6 (memória unificada). Pergunte: "quem tem Mac com chip M?" Esse aluno já usa memória unificada. Explique o que **não** muda: continua precisando de muitos dados com a mesma conta.
4. (5 min) Teste rápido (2 perguntas no notebook). Resposta da 2ª: com 10 000 passos, a viagem é paga uma vez só e a GPU compensa.

**Se atrasar:** passe o 08b para casa e fique só com o slide 6.

### 0:50–1:30 · 3. CuPy em 3 passos
**Mensagem:** **Leve → Calcule (troque `np` por `cp`) → Traga.** E para medir: aqueça e sincronize.
1. (5 min) Vídeo 09 ou slide 7. Escreva no quadro os 3 passos, que vão ficar lá até o fim da aula.
2. (10 min) Von Mises: mostre que a função é **a mesma** para CPU e GPU (`cp.get_array_module`). **Esperado:** GPU 20–60× mais rápida que o NumPy, com os dados já na GPU. Leia em voz alta o prompt "💬 Peça à IA" e incentive a turma a usá-lo.
3. (10 min) "O erro nº 1": rode a célula sem sincronizar. O tempo sai absurdamente pequeno. Frase para o quadro: **"resultado de GPU bom demais? pergunte se sincronizou."**
4. (10 min) Matrizes 4096×4096. **Esperado (T4, aproximado):** CPU float64 ~50–100 GFLOP/s; GPU float64 ~200–250; GPU float32 ~4 000. Pergunte: "quais cálculos de vocês aceitam 7 dígitos?"
5. (5 min) Teste rápido.

**Se atrasar:** pule as matrizes (o slide 9 mostra o resultado).

### 1:45–2:45 · 4. Projeto: laje em incêndio
**Mensagem:** a GPU ganha quando **o problema é grande** e **os dados ficam lá** por muitos passos. E o Numba na CPU é um adversário sério.
1. (10 min) O problema (slide 10): face inferior a 600 °C, "cada ponto vira a média dos vizinhos". Desenhe a cruz de 5 pontos no quadro. **Não** precisa de equação diferencial.
2. (10 min) Leia a função `simula` com a turma: ela não sabe onde está (CPU ou GPU); quem decide é o array. Rode a célula: 3 ✓.
3. (20 min) A corrida por tamanho de malha (1–2 min no Colab). **Enquanto roda, peça palpites:** "a partir de que malha a GPU ganha?" **Esperado (Colab):** em 128² a GPU empata ou perde; a partir de ~512² ela passa à frente; em 2048² ganha com folga do Numba de 2 núcleos. Num notebook com 12+ núcleos e GPU pequena, o Numba chega perto, e isso também é uma lição.
4. (10 min) Mapa da laje: onde está a isoterma de 500 °C? Com cobrimento de 2,5 cm, a armadura passou de 500 °C? É a discussão de engenharia que o cálculo rápido permite.
5. (10 min) Folga, perguntas, ou o "para casa" (`R = 0.3` → instabilidade).

### 2:45–3:15 · 5. O seu exercício na GPU
**Mensagem:** na GPU não há "número de threads" para variar. Monte a **escada**: o mesmo problema em cada versão.
1. (5 min) Abra `exercicios/lista.md` no projetor. Mostre que a escolha é **CPU com Numba (recomendado) ou GPU com CuPy** e que o **exemplo resolvido** é o mesmo da célula a seguir.
2. (10 min) Rode a escada (seção 5). **Leitura:** contando a viagem RAM → GPU, a GPU **perde** para o Numba, porque a conta é leve e feita uma vez só. Com o dado já lá, ela compete. É a regra de ouro outra vez.
3. (10 min) Vídeo 14 e "Usando IA". Mostre o prompt e as **4 verificações**. Frase: "a IA escreve; vocês conferem".
4. (5 min) Abra o **modelo de relatório**: mostre as células ✏️ e que o resto é automático. Mostre `exercicios/INSTALACAO.md` para quem quer rodar no próprio computador.

### 3:15–3:30 · 6. Quando usar GPU + outras ferramentas
1. (8 min) Checklist de 5 perguntas (slide 14). Aplique aos 2–3 problemas anotados no bloco 1.
2. (4 min) Tabela de ferramentas: CuPy ⭐ resolve quase tudo; Numba-CUDA, PyTorch/JAX, RAPIDS e MLX são "para saber que existem". O apêndice do notebook tem kernel e memória gerenciada para os curiosos.
3. (3 min) Prazo da lista e rubrica. Encerramento: **medir, conferir, achar o gargalo.**

### Perguntas que a turma costuma fazer
- *"Meu notebook não tem placa NVIDIA, posso fazer o caminho GPU?"* Sim, no Colab (INSTALACAO.md, parte 1).
- *"O Colab disse que acabou a cota de GPU."* Faça o caminho CPU com Numba (é o recomendado) ou tente no dia seguinte.
- *"A IA me deu um código CuPy 1 000× mais rápido."* Quase sempre faltou sincronizar. Rode a conferência.
- *"Por que a GPU perdeu no meu exercício?"* Provavelmente a conta é leve e a viagem está no tempo. Isso é uma **ótima** análise para o relatório.
- *"Tenho Mac M1/M2…"* O CuPy não roda, mas o caminho CPU com Numba roda muito bem. Quem quiser explorar a GPU do Mac pode usar o Apple MLX (fora da avaliação).

---

## Depois das aulas

- **Prazo** da lista: 2 semanas. Corrija com a [rubrica](exercicios/lista.md#rubrica-10-pontos).
- **Erros comuns nos relatórios:** não conferir o resultado com a função serial; medir sem aquecer; esquecer de sincronizar a GPU; não dizer se a viagem RAM → GPU está no tempo; não dizer o que a IA fez.
- **Atualizar o curso:**
  - Edite `tools/build_aula2.py`, `tools/build_modelo.py` e `exercicios/lista.md`.
  - Rode `python tools/run_nb.py notebooks/Aula2_GPU.ipynb` e `python tools/build_lista_docx.py`.
  - Regere figuras e vídeos.
  - A CI do GitHub executa os notebooks em CPU a cada mudança.
