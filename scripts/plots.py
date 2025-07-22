# plots.py
# Plots the data gathered from experiements

import argparse
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, NamedTuple, Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.typing import ColorType
from matplotlib import colormaps
import numpy as np
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
EXPERIENCES = ["up","symm","down"]
"""
let links = [
    each experience corresponds to a link
    (Bandwidth::Mbps(10.0), Bandwidth::Mbps(1.0)), --> up
    (Bandwidth::Mbps(10.0), Bandwidth::Mbps(10.0)), --> sym
    (Bandwidth::Mbps(1.0), Bandwidth::Mbps(10.0)), -->down
];
"""

#abbreviations that display in the plot
algorithm_abbreviations = {
    "Baseline": "Baseline",
    "Bucketing": "Bu",
    "Rateless": "Rs",
    "Bloom+Rateless": "BlRs",
    "Bloom+Bucketing": "BlBu",
}


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

    return Algorithm(name, formatted, False)


def read_experiments(f: TextIOWrapper, nr_experiments:int, include: set[str] = None, exclude: set[str] = None) -> list[Experiment]:
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
                                
                metrics = Metrics(
                    int(metrics[0]), # state
                    int(metrics[1]), # metadata
                    int(metrics[0]) - int(theoretical_minimum), #redundancy
                    float(metrics[2])
                )
                
                m[algo].append(metrics)    
    
    assert len(headers) == nr_experiments
    assert all(
        all(len(v) == len(list(similarities)) for v in c.values()) for c in collector
    )
    return [Experiment(*p) for p in zip(headers, collector)]


def fmt_label(label: Algorithm) -> str:
    """Simple label format to be displayed in legend"""

    name = algorithm_abbreviations[label.name]

    if not label.params:
        return name

    params = f'[{", ".join(f"${k} = {v}$" for k, v in label.params.items())}]'
    return f"{name} {params}"

def plot_metric(exp: Experiment, colors: dict[Algorithm, ColorType], marker_dict: dict[Algorithm, str], line_style_dict: dict[Algorithm, str], metric_function: Callable[[Metrics],int], metric_name: str) -> Figure:
    """Plot the result of applying metric_function to the measured metrics"""
    visible_algos = [algo for algo in exp.runs if not algo.hidden]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.3)

    ax.xaxis.set_major_formatter(percent_formatter)
    ax.yaxis.set_major_formatter(byte_formatter)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.75)
    ax.set_xlabel("Similarity", fontsize=25)
    ax.set_ylabel(metric_name, fontsize=25, labelpad=8)
    ax.tick_params(axis="both", labelsize=20)

    legend_handles = []

    for algo, metrics in exp.runs.items():
        color = colors[algo]
        label = fmt_label(algo)
        marker = marker_dict[algo]
        line_style = line_style_dict[algo]
        line_handle, = ax.plot(similarities, [metric_function(m) for m in metrics], marker=marker, linestyle=line_style, color=color, lw=2, label=label, markersize=8)
        legend_handles.append(line_handle)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=(len(visible_algos) + 1) // 2,
        frameon=False,
        fontsize=15,
        title_fontsize=30
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
            values = [m.redundancy for m in metrics]
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
            collected = [m.redundancy for m in metrics]
        else:
            raise ValueError(f"Unknown value parameter {what} for what")

        rts = [f"{m / t:.1%}" for m, t in zip(collected, total)]
        print(f"{what} {label}", " ".join(rts), sep="\n")


def plot_time_to_sync(exp: Experiment, colors: dict[Algorithm, ColorType],  marker_dict: dict[Algorithm, str], line_style_dict: dict[Algorithm, str]) -> Figure:
    """Plots the time to sync on different link configurations"""
    fig, ax = plt.subplots(layout="constrained")

    up, down = bit_formatter(exp.env.upload), bit_formatter(exp.env.download)
    ylabel = f"Time to Sync (s)\n{up}/s up, {down}/s down"

    ax.xaxis.set_major_formatter(percent_formatter)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.75)
    ax.set(xlabel="Similarity", xmargin=0, ylabel=ylabel)

    for algo, metrics in exp.runs.items():
        color = colors[algo]
        marker = marker_dict[algo]
        line_style = line_style_dict[algo]
        label = fmt_label(algo)
        time = [m.duration for m in metrics]
        ax.plot(similarities, time, marker=marker, linestyle=line_style, c=color, lw=0.8, label=label)

    ax.legend(title="Algorithms")
    return fig


def filter_runs(
    runs: dict[Algorithm, Metrics],
    name: str | None = None,
    params_match: dict[str, str] | None = None
) -> dict[Algorithm, Metrics]:
    return {
        algo: metric
        for algo, metric in runs.items()
        if (name is None or algo.name == name) and all(algo.params.get(k) == v for k, v in (params_match or {}).items())
    }

def main():
    """Script that extracts relevant data from logs and produces the plots for each experiment"""
    parser = argparse.ArgumentParser(prog="plots")
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
        exps = read_experiments(file, len(EXPERIENCES), include_algorithms, exclude_algorithms)
        
        
        symm_idx = EXPERIENCES.index("symm")
        line_styles = ['solid','dotted','dashdot']
        markers = ['.', 'v', '*', 'D', 's', 'X', 'o']
        colormap = colormaps.get_cmap("tab10")
        colormap_len = 10

        marker_dict = {}
        line_style_dict = {}
        colors_dict = {}
        for i, algo in enumerate(exps[symm_idx].runs.keys()):
            marker_dict[algo] = markers[i % len(markers)]
            line_style_dict[algo] = line_styles[i % len(line_styles)]
            colors_dict[algo] = colormap(i % colormap_len)


        # Display the ratios
        if args.output_ratios:
            for k in ("metadata", "redundancy"):
                print_transmission_ratios(exps[symm_idx], k)

        if args.output_data:
            for k in ("total", "metadata", "redundancy"):
                print_transmitted(exps[symm_idx], k)

        runs = exps[symm_idx].runs
        env = exps[symm_idx].env

        # Bucketing
        bucketing_runs = filter_runs(runs, name="Bucketing")
        bucketing_experiment = Experiment(env, bucketing_runs)

        bucketing_total = plot_metric(
            bucketing_experiment, colors_dict, marker_dict, line_style_dict,
            lambda m: m.state + m.metadata, "Total"
        )
        save_or_show(bucketing_total, "bucketing_total.pdf")

        bucketing_redundancy = plot_metric(
            bucketing_experiment, colors_dict, marker_dict, line_style_dict,
            lambda m: m.redundancy, "Redundancy"
        )
        save_or_show(bucketing_redundancy, "bucketing_redundancy.pdf")

        # Bloom+Bucketing
        bloom_bucketing_runs = filter_runs(runs, name="Bloom+Bucketing")
        bloom_bucketing_experiment = Experiment(env, bloom_bucketing_runs)

        bloom_bucketing_total = plot_metric(
            bloom_bucketing_experiment, colors_dict, marker_dict, line_style_dict,
            lambda m: m.state + m.metadata, "Total"
        )
        save_or_show(bloom_bucketing_total, "bloom_bucketing_total.pdf")

        # Bloom+Rateless
        bloom_rateless_runs = filter_runs(runs, name="Bloom+Rateless")
        bloom_rateless_experiment = Experiment(env, bloom_rateless_runs)

        bloom_rateless_total = plot_metric(
            bloom_rateless_experiment, colors_dict, marker_dict, line_style_dict,
            lambda m: m.state + m.metadata, "Total"
        )
        save_or_show(bloom_rateless_total, "bloom_rateless_total.pdf")

        # Overall Comparison
        overall_comparison_filters = [
            ("Bloom+Bucketing", {'\\epsilon': '1\\%', 'f_{ld}': '0.2'}),
            ("Bloom+Rateless", {'\\epsilon': '1\\%'}),
            ("Rateless", None),
            ("Baseline", None),
            ("Bucketing", {'f_{ld}': '0.2'}),
            ("Bucketing", {'f_{ld}': '1'}),
        ]

        overall_comparison_runs = {
            algo: metric
            for algo, metric in runs.items()
            for name, params in overall_comparison_filters
            if algo.name == name and all(algo.params.get(k) == v for k, v in (params or {}).items())
        }
        overall_comparison_experiment = Experiment(env, overall_comparison_runs)

        overall_comparison_total = plot_metric(
            overall_comparison_experiment, colors_dict, marker_dict, line_style_dict,
            lambda m: m.state + m.metadata, "Total"
        )
        save_or_show(overall_comparison_total, "overall_comparison_total.pdf")

        # Best Comparison Metadata
        best_runs = {
            algo: metric
            for algo, metric in runs.items()
            if (algo.name == "Bloom+Bucketing" and algo.params.get('\\epsilon') == '1\\%' and algo.params.get('f_{ld}') == '0.2')
            or (algo.name == "Bloom+Rateless" and algo.params.get('\\epsilon') == '1\\%')
            or (algo.name == "Baseline")
        }
        best_experiment = Experiment(env, best_runs)

        best_metadata_plot = plot_metric(
            best_experiment, colors_dict, marker_dict, line_style_dict,
            lambda m: m.metadata, "Metadata"
        )
        save_or_show(best_metadata_plot, "overall_comparison_metadata.pdf")



if __name__ == "__main__":
    main()
