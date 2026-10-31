# Slides

Dois decks em [Marp](https://marp.app/) (Markdown com `---` entre slides):

- [`Aula1_CPU.md`](Aula1_CPU.md)
- [`Aula2_GPU.md`](Aula2_GPU.md)

**Ver/apresentar:** abra no VS Code com a extensão *Marp for VS Code*, ou exporte:

```bash
npx @marp-team/marp-cli@latest presentations/Aula1_CPU.md --pdf --allow-local-files
npx @marp-team/marp-cli@latest presentations/Aula2_GPU.md --html
```

**Figuras:** `figures/*.png` são geradas a partir dos números medidos nos notebooks (`notebooks/results/*.json`):

```bash
python presentations/figures/make_figures.py
```

Rode os notebooks no Colab, salve os `results/aula*_colab.json` e regere as figuras para que os slides mostrem os números da máquina da turma.
