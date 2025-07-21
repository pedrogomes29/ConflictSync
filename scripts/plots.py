# plots.py
# Plots the data gathered from experiements

import argparse
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, NamedTuple
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
EXPERIENCES = ["up","symm","down"]
algorithm_abbreviations = {
    "Baseline": "Baseline",
    "Bucketing": "Bu",
    "Rateless": "Rs",
    "Bloom+Rateless": "BlRs",
    "Bloom+Bucketing": "BlBu",
    "Bucketing+Rateless": "BuRs",
    "Bloom+Bucketing+Rateless": "BlBuRs",
    "RBloom+Rateless+Heuristic": "RbRsAn",
    "RBloom+Rateless+Similarity": "RbRsSi",
    "RBloom+Rateless+NoParams": "RbRsCo"
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
        elif pname == "m":
            if math.isclose(float(value), 1 / math.log(2), rel_tol=1e-3):
                formatted["m"] = "opt"
            else:
                formatted["m"] = value
        elif pname == "angle":
            formatted["angle"] = value
        elif pname == "sim":
            formatted["sim"] = value

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
                if include and algo.name not in include:
                    algo = algo._replace(hidden=True)
                if exclude and algo.name in exclude:
                    algo = algo._replace(hidden=True)
                                
                  
                visible = False #True if the algorithm + configuration is to be displayed

                if algo.name=="Bloom+Bucketing" and algo.params.get('\\epsilon')=='1\\%' and algo.params.get('f_{ld}') =='0.2':
                    visible = True
                if algo.name=="Bloom+Rateless" and algo.params.get('\\epsilon')=='1\\%':
                    visible = True
                if algo.name=='Rateless':
                    visible = True
                if algo.name=='Baseline':
                    visible = True
                if algo.name=='Bucketing' and algo.params.get('f_{ld}') in ['0.2','1']:
                    visible = True
              
                """
                visible = False
                    
                if algo.name=="RBloom+Rateless+Heuristic" and algo.params.get('angle')=='0.5':
                    visible = True
                if algo.name=="RBloom+Rateless+Similarity" and algo.params.get('sim')=='0.99':
                    visible = True
                if algo.name=='RBloom+Rateless+NoParams':
                    visible = True      
                """                                        
                if not visible:
                    algo = algo._replace(hidden=True)
  
                                
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

def plot_transmitted_with_surface(exp: Experiment, colors: dict[Algorithm, ColorType], marker_dict: dict[Algorithm, str], min_similarity: int = 0, max_similarity: int = 100) -> Figure:
    """Plot only Metadata transmitted with Bloom+Rateless minimum values as a line (y-lim fixed 0 to 300kB)."""

    blra_percentages_to_plot = []

    not_hidden = filter(lambda algo: not algo.hidden, exp.runs)
    bloom_rateless_to_show = list(filter(lambda algo: algo.name == "Bloom+Rateless" and algo.params.get('\\epsilon') in [f"{x}\\%" for x in blra_percentages_to_plot], not_hidden))
    not_hidden = filter(lambda algo: not algo.hidden, exp.runs)
    other_algos_to_show = filter(lambda algo: algo.name != "Bloom+Rateless", not_hidden)
        
    visible_algos = list(bloom_rateless_to_show) + list(other_algos_to_show)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.25)

    ax.xaxis.set_major_formatter(percent_formatter)
    ax.yaxis.set_major_formatter(byte_formatter)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.75)
    ax.set_xlabel("Similarity", fontsize=25)
    ax.set_ylabel("Metadata", fontsize=25, labelpad=8)
    ax.tick_params(axis="both", labelsize=20)

    legend_handles = []

    # Extract Bloom+Rateless runs
    blra_runs = {
        algo: metrics for algo, metrics in exp.runs.items()
        if not algo.hidden and algo.name == "Bloom+Rateless"
    }

    # Plot other algorithms normally
    for algo, metrics in exp.runs.items():
        if algo.hidden or (algo.name == "Bloom+Rateless" and algo.params.get('\\epsilon') not in [f"{x}\\%" for x in blra_percentages_to_plot]):
            continue

        color = colors[algo]
        label = fmt_label(algo)
        marker = marker_dict[algo]

        if algo.name.startswith("RBloom"):
            # Use transparency, dashed line, and marker for RBloom
            line_handle, = ax.plot(similarities, [m.metadata for m in metrics], color="darkblue", lw=3, ls='--', alpha=0.7, marker=marker, markersize=5, label=label)
        else:
            line_handle, = ax.plot(similarities, [m.metadata for m in metrics], color=color, lw=2, ls='--', marker=marker, markersize=4, label=label)
        legend_handles.append(line_handle)

    if blra_runs:
        metadata_matrix = np.array([[m.metadata for m in metrics] for metrics in blra_runs.values()])
        ymin = np.min(metadata_matrix, axis=0)
        # Plot BlRa Min with solid line, marker, and increased width
        ax.plot(similarities, ymin, color="red", lw=3, ls='-', alpha=0.7, label="BlRa Min")

    ax.set_ylim(0, 255_000)

    total_legend_items = len(visible_algos) + 1  # +1 for 'BlRa Min' line

    fig.legend(
        loc="lower center",
        ncol=math.ceil(total_legend_items / 2),
        frameon=False,
        fontsize=18,
        title_fontsize=30
    )

    return fig

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
        if algo.hidden:
            continue

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


def plot_transmitted(exp: Experiment, colors: dict[Algorithm, ColorType], marker_dict: dict[Algorithm, str]) -> Figure:
    """Plots the transmitted data (total, metadata, redundancy) over the network for each protocol."""
    
    visible_algos = [algo for algo in exp.runs if not algo.hidden]

    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw={'wspace': 0.23})
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
        ncol=(len(visible_algos) + 1) // 2,
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
        
        
        colormap = colormaps.get_cmap("tab10")
        symm_idx = EXPERIENCES.index("symm")
        maintain_colors:bool = True 
        #the same algorithm always has the same color, across plots with different combinations of algorithms 
                
        if maintain_colors:
            colors = {
                a: colormap(i%10)
                for i, a in enumerate(exps[symm_idx].runs.keys())
            }
            line_styles = ['solid','dotted','dashdot']
            markers = ['.', 'v', '*', 'D', 's', 'X', ',', 'o']

            marker_dict = {}
            line_style_dict = {}
            for i, algo in enumerate(exps[symm_idx].runs.keys()):
                marker_dict[algo] = markers[i % len(markers)]
                line_style_dict[algo] = line_styles[i % len(line_styles)]
        else:
            algos_to_plot = []
            for algo in exps[symm_idx].runs.keys():
                if algo.hidden:
                    continue
                if algo.name == "Bloom+Rateless":
                    epsilon = algo.params.get("\\epsilon")
                    if epsilon not in [f"{x}\\%" for x in [1, 10, 25]]:
                        continue
                algos_to_plot.append(algo)

            # Assign colors without i % 10
            colors = {
                algo: colormap(i / max(1, len(algos_to_plot) - 1))  # Spread evenly in colormap
                for i, algo in enumerate(algos_to_plot)
            }

            # Assign markers
            markers = ['o', '^']
            marker_dict = {
                algo: markers[0] if i < len(markers) else markers[i % len(markers)]
                for i, algo in enumerate(algos_to_plot)
            }


        # Display the ratios
        if args.output_ratios:
            for k in ("metadata", "redundancy"):
                print_transmission_ratios(exps[symm_idx], k)

        if args.output_data:
            for k in ("total", "metadata", "redundancy"):
                print_transmitted(exps[symm_idx], k)



        runs = {
            k: v
            for k, v in exps[symm_idx].runs.items()
        }
        core = Experiment(exps[symm_idx].env, runs)

        #transmitted = plot_transmitted(core, colors, marker_dict)
        transmitted = plot_metric(core, colors, marker_dict, line_style_dict, lambda metric: metric.state + metric.metadata, "Total")
        #transmitted = plot_transmitted_with_surface(core, colors, marker_dict)
        name = f"{Path(file.name).stem}_transmitted.pdf"
        save_or_show(transmitted, name)

        """
        for exp_name, exp in zip(EXPERIENCES, exps):
            # Plot the core time experiments
            runs = {
                k: v for k, v in exp.runs.items()
            }
            core = Experiment(exp.env, runs)

            time = plot_time_to_sync(core, colors)
            name = f"{Path(file.name).stem}_time_{exp_name}.pdf"
            save_or_show(time, name)
        """


if __name__ == "__main__":
    main()
