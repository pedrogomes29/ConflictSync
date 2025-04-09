# tables.py
# Reads tables and makes tables from it
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
        assert ctx in ["metadata", "redundancy"]

        if ctx != name:
            f.seek(fptr)
            return values

        values[algo] = [p.replace("%", "\\%") for p in f.readline().rstrip().split()]


def textable(name: str, points: list[int], values: dict[str, list[str]]) -> str:
    assert all(0 <= x <= 100 for x in points)

    indexes = [p // 5 for p in points]
    cols = "l" + "c" * len(points)

    def bold(s: str) -> str:
        return f"\\textbf{{{s}}}"

    percentages = " &".join(bold(f"{p}\\%") for p in points)
    algorithm = bold("Algorithm")
    header = f"\t\t{algorithm} & {percentages} \\\\"

    rows = []
    for a, v in values.items():
        vals = [p for i, p in enumerate(v) if i in indexes]
        assert len(points) == len(vals)

        rows.append(f'\t\t{a} & {" &".join(vals)} \\\\' )

    def rule(kind: str) -> str:
        assert kind in ["top", "mid", "bottom"]
        return f"\t\t\\{kind}rule"

    centering = "\t\\centering"
    caption = "\t\\caption{}"
    label = f"\t\\label{{tab:{name}_ratios}}"

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
    metadata = read(args.file, name="metadata")
    redundancy = read(args.file, name="redundancy")

    # Emit the tables in tex
    metadata_table = textable(f"{dtype}_metadata", percentages, metadata)
    redundancy_table = textable(f"{dtype}_redundancy", percentages, redundancy)

    print(metadata_table, redundancy_table, sep="\n\n")


if __name__ == "__main__":
    main()
