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
import numpy as np
from matplotlib import lines
import math


class Header(NamedTuple):
    upload: int
    download: int


class Metrics(NamedTuple):
    state: int
    metadata: int
    redundancy: int
    duration: float


class Algorithm(NamedTuple):
    name: str
    params: dict[str, str]
    hidden: bool

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


similarities = []
percent_formatter = ticker.PercentFormatter()
byte_formatter = ticker.EngFormatter(unit="B")
bit_formatter = ticker.EngFormatter(unit="b")
NR_EXPERIMENTS = 3

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
        elif pname == "m_ratio":
            if math.isclose(float(value), 1 / math.log(2), rel_tol=1e-3):
                formatted["m\\_ratio"] = "1/ln(2)"
            else:
                formatted["m\\_ratio"] = value
        elif pname == "angle":
            formatted["angle"] = value
        elif pname == "sim":
            formatted["sim"] = value

    return Algorithm(name, formatted, False)


def read_experiments(f: TextIOWrapper, nr_experiments:int, include: set[str] = None, exclude: set[str] = None, min_similarity:int = 0, max_similarity:int = 100) -> list[Experiment]:
    """
    Reads an experiment from the input source.
    This function assumes that the input is not malformed.
    """


    headers = []
    collector = [defaultdict(list) for _ in range(nr_experiments)]
    start_percentage, end_percentage, nr_steps = map(int, f.readline().rstrip().split())
    step_size = (end_percentage - start_percentage) / nr_steps
    
    global similarities
    similarities = [start_percentage + i * step_size for i in range(nr_steps+1)]
    # Ignore the first empty line
    _ = f.readline()

    for s in similarities:
        for i, m in enumerate(collector):
            header_vals = f.readline().rstrip().split()
            theoretical_minimum = header_vals.pop(0)
            header = Header(*map(int, header_vals))
            if s == start_percentage:
                headers.append(header)
            assert headers[i].upload == header.upload
            assert headers[i].download == header.download

            while parts := f.readline().rstrip().split():
                algo, *metrics = parts
                algo = read_algorithm(algo)
                if include and algo.name not in include:
                    algo = algo._replace(hidden=True)
                if exclude and algo.name in exclude:
                    algo = algo._replace(hidden=True)
                
                """
                is_best = False
                if algo.name=="Bloom+Bucketing" and algo.params.get('\\epsilon')=='1\\%' and algo.params.get('f_{ld}') =='0.2':
                    is_best = True
                if algo.name=="Bloom+Rateless" and algo.params.get('\\epsilon')=='1\\%':
                    is_best = True
                if algo.name=='Rateless':
                    is_best = True
                if algo.name=='Baseline':
                    is_best = True
                
                if not is_best:
                    algo = algo._replace(hidden=True)    
                """            
                
                
                metrics = Metrics(
                    int(metrics[0]), # state
                    int(metrics[1]), # metadata
                    int(metrics[0]) - int(theoretical_minimum), #redundancy
                    float(metrics[2])
                )
                
                if min_similarity <= s <= max_similarity:
                    m[algo].append(metrics)
                        
    
    
    assert len(headers) == nr_experiments
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

def plot_transmitted_with_surface(exp: Experiment, colors: dict[Algorithm, ColorType], marker_dict: dict[Algorithm, str]) -> Figure:
    """Plot only Metadata transmitted with Bloom+Rateless as a background surface (y-lim fixed 0 to 300kB)."""

    visible_algos = [algo for algo in exp.runs if not algo.hidden]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    ax.xaxis.set_major_formatter(percent_formatter)
    ax.yaxis.set_major_formatter(byte_formatter)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.75)
    ax.set_xlabel("Similarity", fontsize=25)
    ax.set_ylabel("Metadata (Bytes)", fontsize=25, labelpad=8)
    ax.tick_params(axis="both", labelsize=20)

    legend_handles = []

    # Extract Bloom+Rateless runs
    blra_runs = {
        algo: metrics for algo, metrics in exp.runs.items()
        if not algo.hidden and algo.name == "Bloom+Rateless"
    }

    if blra_runs:
        metadata_matrix = np.array([[m.metadata for m in metrics] for metrics in blra_runs.values()])
        ymin = np.min(metadata_matrix, axis=0)
        ymax = np.max(metadata_matrix, axis=0)
        ax.fill_between(similarities, ymin, ymax, color="lightblue", alpha=0.3, label="BlRa")

    # Plot other algorithms normally
    for algo, metrics in exp.runs.items():
        if algo.hidden or algo.name == "Bloom+Rateless":
            continue

        color = colors[algo]
        label = fmt_label(algo)
        marker = marker_dict[algo]

        line_handle, = ax.plot(similarities, [m.metadata for m in metrics], marker=marker, color=color, lw=2, label=label, markersize=8)
        legend_handles.append(line_handle)

    # Fix y-axis to [0, 300kB]
    ax.set_ylim(0, 255_000)

    # Custom legend: includes lines + surface patch
    fig.legend(
        handles=legend_handles + [lines.Line2D([], [], color='lightblue', alpha=0.3, lw=10, label='BlRa')],
        loc="lower center",
        ncol=(len(visible_algos) + 2) // 3,
        frameon=False,
        fontsize=30,
        title_fontsize=40
    )

    return fig


def plot_transmitted(exp: Experiment, colors: dict[Algorithm, ColorType], marker_dict: dict[Algorithm, str]) -> Figure:
    """Plots the transmitted data (total, metadata, redundancy) over the network for each protocol."""
    
    visible_algos = [algo for algo in exp.runs if not algo.hidden]

    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw={'wspace': 0.23})  # Adjust spacing
    ax1, ax2, ax3 = axes  # Unpack subplots

    # Adjust layout to make space for the legend
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.3)

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
        if(algo.hidden):
            continue
        color = colors[algo]
        label = fmt_label(algo)
        marker = marker_dict[algo]

        line_handle, = ax1.plot(similarities, [m.state + m.metadata for m in metrics], marker=marker, color=color, lw=2, label=label,  markersize=8)
        ax2.plot(similarities, [m.metadata for m in metrics],  marker=marker, c=color, lw=2, markersize=8)
        ax3.plot(similarities, [m.redundancy for m in metrics],  marker=marker, c=color, lw=2, markersize=8)

        legend_handles.append(line_handle)  # Store one handle per algorithm


    fig.legend(
        handles=legend_handles,       # Uses the stored line handles for consistency
        loc="lower center",           # Places the legend below the graphs, centered
        ncol=(len(visible_algos) + 2) // 3,
        frameon=False,                # Removes the box around the legend,
        fontsize=30,                  # Increases legend text size
        title_fontsize=40             # Increases legend title size
    )

    return fig



def print_transmitted(exp: Experiment, what: str) -> Figure:
    """Prints the actual values of total, metadata, or redundancy transmitted (in bytes)."""
    for algo, metrics in exp.runs.items():
        if(algo.hidden):
            continue
        label = fmt_label(algo)
        if what == "total":
            values = [m.state + m.metadata for m in metrics]
        elif what == "metadata":
            values = [m.metadata for m in metrics]
        elif what == "redundancy":
            values = [m.redundancy for m in metrics]
        else:
            raise ValueError(f"Unknown value parameter {what} for 'what'")

        formatted = [byte_formatter(v) for v in values]
        print(f"{what} {label}", " ".join(formatted), sep="\n")

def print_transmission_ratios(exp: Experiment, what: str):
    """Prints the ratios of metadata and redundancy against the total transmitted."""
    for algo, metrics in exp.runs.items():
        if(algo.hidden):
            continue
        label = fmt_label(algo)
        total = [m.state + m.metadata for m in metrics]

        if what == "metadata":
            collected = [m.metadata for m in metrics]
        elif what == "redundancy":
            collected = [m.redundancy for m in metrics]
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
        if(algo.hidden):
            continue
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
        exps = read_experiments(file, NR_EXPERIMENTS, include_algorithms, exclude_algorithms, args.min_similarity, args.max_similarity)
        exps.insert(0, {}) #TODO: Remove

        #get number of algorithms
                
        colormap = colormaps.get_cmap("tab10")
                
        colors = {
            a: colormap(i%10)
            for i, a in enumerate(exps[1].runs.keys())
        }
        markers = ['o', '^']
        
        marker_dict = {}
        for i, algo in enumerate(exps[1].runs.keys()):
            if i < 10:
                marker_dict[algo] = markers[0]
            else:
                marker_dict[algo] = markers[1]



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

        transmitted = plot_transmitted(core, colors, marker_dict)
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
