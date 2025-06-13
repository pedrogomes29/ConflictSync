import argparse
from pathlib import Path
from matplotlib.figure import Figure
from typing import NamedTuple, TextIO
import matplotlib.pyplot as plt
import math

class Header(NamedTuple):
    similarity:int
    
class Metrics(NamedTuple):
    run_nr: int
    positives: int


class Experiment(NamedTuple):
    header: Header
    runs: list[Metrics]

SET_CARDINALITY = 100_000


def read_experiment(f: TextIO) -> list[Experiment]:
    start_percentage, end_percentage, nr_steps = map(int, f.readline().split())
    step_size = (end_percentage - start_percentage) / nr_steps
    similarities = [start_percentage + i * step_size for i in range(nr_steps + 1)]

    f.readline()

    experiments = []
    for similarity in similarities:
        _ = f.readline().split()
        header=Header(similarity=similarity)
        metrics = []
        while True:
            line = f.readline()
            if not line.strip():  # blank line ends the block
                break
            run_nr, metadata = map(int, line.split())
            metrics.append(Metrics(run_nr=run_nr, positives=metadata))

        experiments.append(Experiment(header=header, runs=metrics))

    return experiments


def detect_convergence(positives: list[int], angle_threshold_deg: float = 1.0, window: int = 1) -> int:
    """
    Detects when the average angle (in degrees) of the curve over a window is below a threshold.
    Normalizes by dividing by the given nr_positives.
    """
    if len(positives) < window + 1:
        return -1

    for i in range(window, len(positives)):
        angles = []
        for j in range(window):
            dy = positives[i - j] - positives[i - j - 1]
            angle_rad = math.atan(abs(dy))  # dx = 1
            angle_deg = math.degrees(angle_rad)
            angles.append(angle_deg)
        avg_angle = sum(angles) / window
        if avg_angle < angle_threshold_deg:
            return i
    return -1

def plot_experiments(experiments: list[Experiment]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    for experiment in experiments:
        run_nrs = [m.run_nr for m in experiment.runs]
        positives = [m.positives for m in experiment.runs]
        normalized_positives = [p / SET_CARDINALITY for p in positives]
        similarity = experiment.header.similarity


        label = f"{similarity:.1f}%"
        ax.plot(run_nrs, normalized_positives, marker='o', label=label)

        # Detect convergence
        convergence_idx = detect_convergence(normalized_positives)
        if convergence_idx != -1:
            convergence_run = run_nrs[convergence_idx]
            ax.scatter(convergence_run, normalized_positives[convergence_idx], color='red', s=100, marker='*', zorder=5)
            ax.annotate(
                f"{convergence_run + 1}", 
                (convergence_run, normalized_positives[convergence_idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                color='red'
            )
            

    ax.set_title("Positives vs Slice")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Positives (divided by set size)")
    ax.grid(True)
    ax.legend(title="Similarity")
    fig.tight_layout()

    return fig
        
def main():
    """Script that extracts experiment data and produces a single plot"""
    parser = argparse.ArgumentParser(prog="plotter")
    parser.add_argument("files", nargs="*", default=("-"), type=argparse.FileType("r"))
    parser.add_argument("--save", action="store_true", help="Save plot to disk")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-paper")
    plt.rc("font", family="serif")

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
        experiments = read_experiment(file)
        fig = plot_experiments(experiments)
        base_name = Path(file.name).stem
        fname = f"{base_name}_positives_all_similarities.pdf"
        save_or_show(fig, fname)


if __name__ == "__main__":
    main()