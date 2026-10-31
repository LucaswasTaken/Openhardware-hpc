"""Execute a notebook in place (cwd = notebooks/) and print each cell's wall time and errors."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}},
                        allow_errors=True)
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t = time.perf_counter()
        client.execute_cell(cell, i)
        dt = time.perf_counter() - t
        errs = [o for o in cell.outputs if o.output_type == "error"]
        head = cell.source.splitlines()[0][:60]
        print(f"[{i:02d}] {dt:7.1f}s {'ERRO ' + errs[0].ename + ': ' + errs[0].evalue[:200] if errs else ''} | {head}", flush=True)
nbformat.write(nb, path)
