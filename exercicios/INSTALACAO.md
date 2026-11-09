# Como rodar os exercícios

Há dois caminhos. **Comece pelo Colab**: não precisa instalar nada, e tem GPU de graça.

| | Colab (navegador) | Seu computador |
|---|---|---|
| Instalação | nenhuma | ~10 minutos, uma vez |
| CPU | 2 núcleos (curva de escalabilidade vai só até 2) | todos os seus núcleos (4–16, em geral) |
| GPU | NVIDIA T4, grátis (algumas horas por dia) | só se você tiver placa NVIDIA |
| Indicado para | caminho GPU (CuPy); primeiros testes | caminho CPU (Numba), para ter uma curva de verdade |

O ideal para o caminho CPU: **fazer funcionar no Colab e medir no seu computador.**

---

## 1. Colab (recomendado para começar)

1. Abra o modelo: [modelo_relatorio.ipynb no Colab](https://colab.research.google.com/github/LucaswasTaken/Openhardware-hpc/blob/master/exercicios/modelo_relatorio.ipynb).
2. **Arquivo → Salvar uma cópia no Drive.** (Sem isso, você perde o que fizer.)
3. **Só no caminho GPU:** *Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU → Salvar*.
4. *Ambiente de execução → Executar tudo*. A primeira célula imprime algo como:
   ```
   Python 3.12 · 2 núcleos lógicos · GPU: True · Colab: True
   ```
   `GPU: True` só aparece se você escolheu a T4 no passo 3.

Numba, NumPy e CuPy já vêm instalados no Colab.

**Problemas comuns no Colab**
- *"Não é possível se conectar a um back-end de GPU"*: a cota grátis do dia acabou. Use CPU hoje (o caminho Numba funciona igual) ou tente amanhã.
- *O ambiente reiniciou e as variáveis sumiram*: normal após ~90 min parado. Rode tudo de novo (*Executar tudo*).
- *Tempos muito diferentes entre execuções*: o Colab é compartilhado. Repita a medida (o modelo já repete 5 vezes e pega a menor).

---

## 2. No seu computador (Windows, Linux ou Mac)

Vamos usar o **uv**, um instalador de Python rápido que não mexe em nada do seu sistema.

### Passo 1 — instalar o uv
**Windows** (abra o *PowerShell* pelo menu Iniciar):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
**Linux / Mac** (abra o *Terminal*):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Feche e abra o terminal de novo. Teste: `uv --version` deve mostrar um número de versão.

### Passo 2 — baixar o material
Se tiver o Git: `git clone https://github.com/LucaswasTaken/Openhardware-hpc`
Se não: no [GitHub do curso](https://github.com/LucaswasTaken/Openhardware-hpc), botão verde **Code → Download ZIP**, e descompacte.

Depois entre na pasta:
```bash
cd Openhardware-hpc
```

### Passo 3 — criar o ambiente e instalar as bibliotecas
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```
O uv baixa o próprio Python 3.12 se você não tiver. Leva 1–3 minutos.

### Passo 4 — abrir o Jupyter
```bash
uv run jupyter lab
```
O navegador abre sozinho. Navegue até `exercicios/modelo_relatorio.ipynb` e rode a primeira célula. Deve aparecer o número de núcleos da sua máquina (ex.: `12 núcleos lógicos`).

**Prefere VS Code?** Abra a pasta `Openhardware-hpc`, abra o notebook e, no canto superior direito, escolha o kernel `.venv (Python 3.12)`.

### Passo 5 (opcional) — GPU NVIDIA no seu computador
Só se você tem placa **NVIDIA** (GeForce, RTX, Quadro…). Placas AMD/Intel e Macs: use o Colab para o caminho GPU.

1. Atualize o **driver** da NVIDIA (site da NVIDIA ou GeForce Experience). Teste no terminal: `nvidia-smi` deve mostrar a sua placa.
2. Instale o CuPy (ele já traz o CUDA, não precisa instalar o CUDA Toolkit à parte):
   ```bash
   uv pip install "cupy-cuda12x[ctk]"
   ```
3. Teste:
   ```bash
   uv run python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
   ```
   Deve imprimir o nome da sua placa.

---

## Erros comuns no computador

| Mensagem | O que fazer |
|---|---|
| `uv: command not found` / `'uv' não é reconhecido` | Feche e abra o terminal depois de instalar o uv. |
| `No module named 'numba'` | O notebook está usando outro Python. No Jupyter/VS Code, escolha o kernel do `.venv`. |
| `CUDA driver version is insufficient` | Atualize o driver da NVIDIA. |
| `cudaErrorNoDevice` / `GPU: False` | Sem placa NVIDIA visível: use o Colab para o caminho GPU. |
| Numba demora vários segundos na 1ª chamada | Normal: é a compilação. Por isso o modelo aquece antes de medir. |
| `MemoryError` ou o computador trava | Diminua o tamanho do problema (ex.: de $10^8$ para $10^7$) e diga isso no relatório. |
| Windows: `execução de scripts foi desabilitada` | Rode no PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

Travou em outra coisa? Copie a mensagem de erro **inteira** e pergunte à IA ("estou instalando numba com uv no Windows e apareceu: …") — ou abra uma [issue](https://github.com/LucaswasTaken/Openhardware-hpc/issues).
