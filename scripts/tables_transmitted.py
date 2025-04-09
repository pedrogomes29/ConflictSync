# tables.py
# Reads tables and outputs clean LaTeX tables for appendix/reference
import argparse
from io import TextIOWrapper
import pathlib


def read(f: TextIOWrapper, *, name: str) -> dict[str, list[str]]:
    """Reads the ratios from the input file"""
    values = {}
    while True:
        fptr = f.tell()
        parts = f.readline().rstrip().split(maxsplit=1)

        if not parts:
            return values

        ctx, algo = parts
        assert ctx in ["total", "metadata", "redundancy"]
        if ctx != name:
            f.seek(fptr)
            return values

        raw = f.readline().rstrip().split()
        grouped = [f"{raw[i]} {raw[i+1]}" for i in range(0, len(raw), 2)]
        values[algo] = grouped


def latex_si_format(val: str) -> str:
    """Formats a value like '2.3 MB' as a LaTeX siunitx command with rounding"""
    num, unit = val.split()
    num = float(num)

    # Round to 3 significant digits
    if num == 0:
        rounded = "0"
    elif num < 0.01:
        rounded = f"{num:.2e}"
    elif num < 10:
        rounded = f"{num:.3g}"
    else:
        rounded = f"{num:.4g}"

    prefix = {
        "B": r"\byte",
        "kB": r"\kilo\byte",
        "MB": r"\mega\byte",
    }[unit]

    return f"\\SI{{{rounded}}}{{{prefix}}}"

def textable(name: str, points: list[int], values: dict[str, list[str]]) -> str:
    assert all(0 <= x <= 100 for x in points)
    indexes = [p // 5 for p in points]
    cols = "l" + "c" * len(points)

    # Table header
    header = (
        "\t\t\\textbf{Algorithm} & "
        + " & ".join(f"\\textbf{{{p}\\%}}" for p in points)
        + " \\\\"
    )

    # Extract only needed percentages per algorithm
    rows = []
    for algo, vals in values.items():
        selected = [latex_si_format(vals[i]) for i in indexes]
        rows.append(f"\t\t{algo} & {' & '.join(selected)} \\\\")

    def rule(kind: str) -> str:
        return f"\t\t\\{kind}rule"

    centering = "\t\\centering"
    caption = f"\t\\caption{{{name.replace('_', ' ').title()}}}"
    label = f"\t\\label{{tab:{name}}}"

    return "\n".join(
        ["\\begin{table*}[h]", centering]
        + [f"\t\\begin{{tabular}}{{{cols}}}", rule("top"), header, rule("mid")]
        + rows
        + [rule("bottom"), "\t\\end{tabular}"]
        + [caption, label, "\\end{table*}"]
    )


def main():
    """Produces tables in tex format"""
    parser = argparse.ArgumentParser(prog="plotter")
    parser.add_argument("file", nargs="?", default=("-"), type=argparse.FileType("r"))
    args = parser.parse_args()

    percentages = [0, 25, 50, 75, 90, 95, 100]
    dtype = pathlib.Path(args.file.name).stem

    # Read the ratios
    total = read(args.file, name="total")
    metadata = read(args.file, name="metadata")
    redundancy = read(args.file, name="redundancy")

    # Emit the tables in tex
    total_table = textable(f"{dtype}_total", percentages, total)
    metadata_table = textable(f"{dtype}_metadata", percentages, metadata)
    redundancy_table = textable(f"{dtype}_redundancy", percentages, redundancy)

    print(total_table, metadata_table, redundancy_table, sep="\n\n")


if __name__ == "__main__":
    main()
