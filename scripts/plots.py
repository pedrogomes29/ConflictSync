# plots.py
# Plots the data gathered from experiements

import argparse
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import NamedTuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.typing import ColorType
from matplotlib import colormaps


class Header(NamedTuple):
    avg_size: int
    upload: int
    download: int


class Metrics(NamedTuple):
    state: int
    metadata: int
    duration: float


class Algorithm(NamedTuple):
    name: str
    params: dict[str, str]

    def __hash__(self) -> int:
        return hash((self.name, frozenset(self.params.items())))

    @property
    def lf(self) -> float:
        return float(self.params["f_{ld}"])

    def is_blbu(self) -> bool:
        return self.name == "Bloom+Bucketing"


class Experiment(NamedTuple):
    env: Header
    runs: dict[Algorithm, list[Metrics]]


similarities = range(0, 101, 5)
percent_formatter = ticker.PercentFormatter()
byte_formatter = ticker.EngFormatter(unit="B")
bit_formatter = ticker.EngFormatter(unit="b")


def read_algorithm(k: str) -> Algorithm:
    """
    Parses an algorithm key.
    This function assumes that the input is not malformed.
    """
    name, *params = k.replace("[", " ").replace(",", " ").removesuffix("]").split()
    formatted = {}

    for param in params:
        pname, value = param.split("=")
        if pname == "fpr":
            formatted["\\epsilon"] = value.replace("%", "\\%")
        elif pname == "lf":
            formatted["f_{ld}"] = value

    return Algorithm(name, formatted)


def read_experiments(f: TextIOWrapper, include: set[str] = None, exclude: set[str] = None, min_similarity:int = 0, max_similarity:int = 100) -> list[Experiment]:
    """
    Reads an experiment from the input source.
    This function assumes that the input is not malformed.
    """
    # Ignore the first empty line
    _ = f.readline()

    headers = []
    collector = [
        defaultdict(list[Metrics]),
        defaultdict(list[Metrics]),
        defaultdict(list[Metrics]),
    ]
    global similarities

    for s in similarities:
        for i, m in enumerate(collector):
            header = Header(*map(int, f.readline().rstrip().split()))
            if s == 0:
                headers.append(header)
            assert headers[i].upload == header.upload
            assert headers[i].download == header.download

            while parts := f.readline().rstrip().split():
                algo, *metrics = parts
                algo = read_algorithm(algo)
                                
                if include and algo.name not in include:
                    continue
                if exclude and algo.name in exclude:
                    continue
                
                
                metrics = Metrics(int(metrics[0]), int(metrics[1]), float(metrics[2]))
                if min_similarity <= s <= max_similarity:
                    m[algo].append(metrics)
                    
    similarities = range(min_similarity, max_similarity + 1, 5)
    
    
    assert len(headers) == 3
    assert all(
        all(len(v) == len(list(similarities)) for v in c.values()) for c in collector
    )
    return [Experiment(*p) for p in zip(headers, collector)]


def fmt_label(label: Algorithm) -> str:
    """Simple label format to be displayed in legend"""
    if not label.params:
        return label.name

    name = "".join(p[:2] for p in label.name.split("+"))
    params = f'[{", ".join(f"${k} = {v}$" for k, v in label.params.items())}]'
    return f"{name} {params}"


def plot_transmitted(exp: Experiment, colors: dict[Algorithm, ColorType]) -> Figure:
    """Plots the transmitted data (total, metadata, redundancy) over the network for each protocol."""
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw={'wspace': 0.23})  # Adjust spacing
    ax1, ax2, ax3 = axes  # Unpack subplots

    # Adjust layout to make space for the legend
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.3)  # Increased bottom space

    markers = ['o', '^', 'v', 's', 'D', '*', 'p', 'H', 'X', '+']
    marker_dict = {algo: markers[i % len(markers)] for i, algo in enumerate(exp.runs)}


    # Format axes
    labels = ["Total", "Metadata", "Redundancy"]
    for ax, label in zip(axes, labels):
        ax.xaxis.set_major_formatter(percent_formatter)
        ax.yaxis.set_major_formatter(byte_formatter)
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.75)
        ax.set_xlabel("Similarity", fontsize=25)
        ax.set_ylabel(f"{label} (Bytes)", fontsize=25, labelpad=8)
        ax.tick_params(axis="both", labelsize=20)


    legend_handles = []  # Store legend handles

    # Plot data
    for _, (algo, metrics) in enumerate(exp.runs.items()):
        color = colors[algo]
        label = fmt_label(algo)
        marker = marker_dict[algo]

        line_handle, = ax1.plot(similarities, [m.state + m.metadata for m in metrics], marker=marker, color=color, lw=2, label=label,  markersize=8)
        ax2.plot(similarities, [m.metadata for m in metrics],  marker=marker, c=color, lw=2, markersize=8)
        base_pts = [2 * (1 - (s / 100)) * exp.env.avg_size for s in similarities]
        ax3.plot(similarities, [max(m.state - nr, 0) for m, nr in zip(metrics, base_pts)],  marker=marker, c=color, lw=2, markersize=8)

        legend_handles.append(line_handle)  # Store one handle per algorithm


    fig.legend(
        handles=legend_handles,       # Uses the stored line handles for consistency
        loc="lower center",           # Places the legend below the graphs, centered
        ncol=(len(exp.runs) + 1) // 2,
        frameon=False,                # Removes the box around the legend,
        fontsize=30,                  # Increases legend text size
        title_fontsize=40             # Increases legend title size
    )

    return fig



def print_transmitted(exp: Experiment, what: str) -> Figure:
    """Prints the actual values of total, metadata, or redundancy transmitted (in bytes)."""
    for algo, metrics in exp.runs.items():
        label = fmt_label(algo)
        if what == "total":
            values = [m.state + m.metadata for m in metrics]
        elif what == "metadata":
            values = [m.metadata for m in metrics]
        elif what == "redundancy":
            base_points = [2 * (1 - (s / 100)) * exp.env.avg_size for s in similarities]
            values = [max(m.state - nr, 0) for m, nr in zip(metrics, base_points)]
        else:
            raise ValueError(f"Unknown value parameter {what} for 'what'")

        formatted = [byte_formatter(v) for v in values]
        print(f"{what} {label}", " ".join(formatted), sep="\n")

def print_transmission_ratios(exp: Experiment, what: str):
    """Prints the ratios of metadata and redundancy against the total transmitted."""
    for algo, metrics in exp.runs.items():
        label = fmt_label(algo)
        total = [m.state + m.metadata for m in metrics]

        if what == "metadata":
            collected = [m.metadata for m in metrics]
        elif what == "redundancy":
            base_points = [2 * (1 - (s / 100)) * exp.env.avg_size for s in similarities]
            collected = [max(m.state - nr, 0) for m, nr in zip(metrics, base_points)]
        else:
            raise ValueError(f"Unknown value parameter {what} for what")

        rts = [f"{m / t:.1%}" for m, t in zip(collected, total)]
        print(f"{what} {label}", " ".join(rts), sep="\n")


def plot_time_to_sync(exp: Experiment, colors: dict[Algorithm, ColorType]) -> Figure:
    """Plots the time to sync on different link configurations"""
    fig, ax = plt.subplots(layout="constrained")

    up, down = bit_formatter(exp.env.upload), bit_formatter(exp.env.download)
    ylabel = f"Time to Sync (s)\n{up}/s up, {down}/s down"

    ax.xaxis.set_major_formatter(percent_formatter)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.75)
    ax.set(xlabel="Similarity", xmargin=0, ylabel=ylabel)

    for algo, metrics in exp.runs.items():
        color = colors[algo]
        label = fmt_label(algo)
        time = [m.duration for m in metrics]
        ax.plot(similarities, time, "o-", c=color, lw=0.8, label=label)

    ax.legend(title="Algorithms")
    return fig


def main():
    """Script that extracts relevant data from logs and produces the plots for each experiment"""
    parser = argparse.ArgumentParser(prog="plotter")
    parser.add_argument("files", nargs="*", default=("-"), type=argparse.FileType("r"))
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output_data", action="store_true", help="Output the transmitted data that was input")
    parser.add_argument("--output_ratios", action="store_true", help="Output metadata and redundancy transmission ratios")
    parser.add_argument("--include", nargs="*", help="Algorithms to include")
    parser.add_argument("--exclude", nargs="*", help="Algorithms to exclude")
    parser.add_argument("--min_similarity", type=int, default=0, help="Minimum similarity to plot (default: 0)")
    parser.add_argument("--max_similarity", type=int, default=100, help="Maximum similarity to plot (default: 100)")
    args = parser.parse_args()

    include_algorithms = set(args.include) if args.include else None
    exclude_algorithms = set(args.exclude) if args.exclude else None
    
    # Set global configs for plotting
    plt.style.use("seaborn-v0_8-paper")
    plt.rc("font", family="serif")

    # Setup the out directory
    out_dir = Path("results/")
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    def save_or_show(fig: Figure, fname: str):
        if args.save:
            fig.savefig(out_dir / fname, dpi=600)
            plt.close(fig)
        if args.show:
            plt.show()

    for file in args.files:
        # File reading
        exps = read_experiments(file, include_algorithms, exclude_algorithms, args.min_similarity, args.max_similarity)
        #get number of algorithms
                
        nr_algorithms = len(exps[1].runs)
        colormap = colormaps.get_cmap("tab10") if nr_algorithms<=10 else colormaps.get_cmap("tab20")
        colors = [colormap(i) for i in range(nr_algorithms)]

        colors = {
            a: colors[i]
            for i, a in enumerate(exps[1].runs.keys())
        }


        # Display the ratios
        if args.output_ratios:
            for k in ("metadata", "redundancy"):
                print_transmission_ratios(exps[1], k)

        if args.output_data:
            for k in ("total", "metadata", "redundancy"):
                print_transmitted(exps[1], k)



        runs = {
            k: v
            for k, v in exps[1].runs.items()
        }
        core = Experiment(exps[1].env, runs)

        transmitted = plot_transmitted(core, colors)
        name = f"{Path(file.name).stem}_transmitted.pdf"
        save_or_show(transmitted, name)

        for exp, k in zip(exps, ["up", "symm", "down"]):
            # Plot the core time experiments
            runs = {
                k: v for k, v in exp.runs.items()
            }
            core = Experiment(exp.env, runs)

            time = plot_time_to_sync(core, colors)
            name = f"{Path(file.name).stem}_time_{k}.pdf"
            save_or_show(time, name)


if __name__ == "__main__":
    main()
