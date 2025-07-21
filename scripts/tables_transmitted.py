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


def parse_num(val: str) -> float:
    num, unit = val.split()
    num = float(num)
    scale = {"B": 1, "kB": 1e3, "MB": 1e6}[unit]
    return num * scale


def latex_si_format(val: str, bold: bool = False) -> str:
    """Formats a value like '2.3 MB' as a LaTeX siunitx command with rounding"""
    num, unit = val.split()
    num = float(num)

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

    content = f"\\SI{{{rounded}}}{{{prefix}}}"
    return f"\\textbf{{{content}}}" if bold else content


def textable(name: str, points: list[int], values: dict[str, list[str]], bold_min: bool = False) -> str:
    assert all(0 <= x <= 100 for x in points)
    indexes = [p // 5 for p in points]
    cols = "l" + "c" * len(points)

    header = (
        "\t\t\\textbf{Algorithm} & "
        + " & ".join(f"\\textbf{{{p}\\%}}" for p in points)
        + " \\\\"
    )

    # Prepare matrix of values for comparisons
    matrix = []
    for algo, vals in values.items():
        selected = [vals[i].replace("−", "-") for i in indexes]
        matrix.append((algo, selected))

    # Find min values per column
    min_per_col = []
    for i in range(len(indexes)):
        col_vals = [parse_num(row[1][i]) for row in matrix]
        min_val = min(col_vals)
        min_per_col.append(min_val)

    # Build rows
    rows = []
    for algo, vals in matrix:
        formatted = []
        for i, val in enumerate(vals):
            num = parse_num(val)
            is_min_of_col = num == min_per_col[i]
            formatted.append(latex_si_format(val, is_min_of_col and bold_min))
        rows.append(f"\t\t{algo} & {' & '.join(formatted)} \\\\")

    def rule(kind: str) -> str:
        return f"\t\t\\{kind}rule"

    centering = "\t\\centering"
    font_size = "\t\\footnotesize"
    caption = f"\t\\caption{{{name.replace('_', ' ').title()}}}"
    label = f"\t\\label{{tab:{name}}}"

    return "\n".join(
        ["\\begin{table*}[h]", font_size, centering]
        + [f"\t\\begin{{tabular}}{{{cols}}}", rule("top"), header, rule("mid")]
        + rows
        + [rule("bottom"), "\t\\end{tabular}"]
        + [caption, label, "\\end{table*}"]
    )


def main():
    """Produces tables in tex format"""
    parser = argparse.ArgumentParser(prog="tables_transmitted")
    parser.add_argument("file", nargs="?", default=("-"), type=argparse.FileType("r"))
    parser.add_argument("--data_type_name", help="Optional name of the data type", default=None)

    args = parser.parse_args()

    percentages = [0, 25, 50, 75, 90, 95, 100]
    dtype = args.data_type_name or pathlib.Path(args.file.name).stem

    total = read(args.file, name="total")
    metadata = read(args.file, name="metadata")
    redundancy = read(args.file, name="redundancy")

    #total_table = textable(f"{dtype}_transmitted_total", percentages, total, True)
    metadata_table = textable(f"{dtype}_transmitted_metadata", percentages, metadata, True)
    #redundancy_table = textable(f"{dtype}_transmitted_redundancy", percentages, redundancy)
    print(metadata_table, sep="\n\n")

    #print(total_table, metadata_table, redundancy_table, sep="\n\n")


if __name__ == "__main__":
    main()
