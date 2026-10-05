"""Converts a `# %%` percent-format script into a Jupyter notebook.

`# %% [markdown]` cells become markdown (leading `# ` stripped from each line); every
other `# %%` cell becomes a code cell. A minimal stand-in for `jupytext --to ipynb`.

Usage: python py_to_ipynb.py notebooks/verify_between_person_signal.py
"""

import sys
from pathlib import Path

import nbformat


def convert(src: Path) -> Path:
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        elif kind == "code" and text:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = src.with_suffix(".ipynb")
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    print(convert(Path(sys.argv[1])))
