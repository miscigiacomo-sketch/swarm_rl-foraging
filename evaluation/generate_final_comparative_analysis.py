import csv
import os

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = [
    {
        "scenario": "2 agents\nfixed obstacles\n5x5",
        "experiment": "Exp. 21",
        "environment": "fixed_obstacles",
        "agents": 2,
        "grid_size": 5,
        "random_success": 76.40,
        "greedy_success": 93.10,
        "ppo_deterministic_success": 92.80,
        "ppo_stochastic_success": 100.00,
        "random_length": 24.63,
        "greedy_length": 5.83,
        "ppo_deterministic_length": 6.08,
        "ppo_stochastic_length": 3.32,
    },
    {
        "scenario": "2 agents\nrandom obstacles\n5x5",
        "experiment": "Exp. 23",
        "environment": "random_obstacles",
        "agents": 2,
        "grid_size": 5,
        "random_success": 81.50,
        "greedy_success": 93.50,
        "ppo_deterministic_success": 82.40,
        "ppo_stochastic_success": 99.40,
        "random_length": 22.24,
        "greedy_length": 5.45,
        "ppo_deterministic_length": 10.77,
        "ppo_stochastic_length": 3.91,
    },
    {
        "scenario": "2 agents\nrandom obstacles\n10x10",
        "experiment": "Exp. 24",
        "environment": "random_obstacles",
        "agents": 2,
        "grid_size": 10,
        "random_success": 37.60,
        "greedy_success": 95.10,
        "ppo_deterministic_success": 86.30,
        "ppo_stochastic_success": 99.98,
        "random_length": 38.81,
        "greedy_length": 7.03,
        "ppo_deterministic_length": 11.51,
        "ppo_stochastic_length": 7.15,
    },
    {
        "scenario": "3 agents\nrandom obstacles\n10x10",
        "experiment": "Exp. 25",
        "environment": "random_obstacles",
        "agents": 3,
        "grid_size": 10,
        "random_success": 50.00,
        "greedy_success": 96.20,
        "ppo_deterministic_success": 65.00,
        "ppo_stochastic_success": 99.82,
        "random_length": 35.12,
        "greedy_length": 5.66,
        "ppo_deterministic_length": 21.72,
        "ppo_stochastic_length": 9.92,
    },
]


METHODS = [
    ("random", "Random baseline"),
    ("greedy", "Greedy baseline"),
    ("ppo_deterministic", "PPO deterministic"),
    ("ppo_stochastic", "PPO stochastic"),
]


def ensure_output_directories():
    os.makedirs("results/final_analysis", exist_ok=True)
    os.makedirs("plots", exist_ok=True)


def save_csv_table():
    output_path = "results/final_analysis/final_comparative_results.csv"

    fieldnames = [
        "experiment",
        "scenario",
        "environment",
        "agents",
        "grid_size",
        "method",
        "success_rate_percent",
        "average_episode_length",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for experiment in EXPERIMENTS:
            for method_key, method_name in METHODS:
                writer.writerow(
                    {
                        "experiment": experiment["experiment"],
                        "scenario": experiment["scenario"].replace("\n", " "),
                        "environment": experiment["environment"],
                        "agents": experiment["agents"],
                        "grid_size": experiment["grid_size"],
                        "method": method_name,
                        "success_rate_percent": experiment[
                            f"{method_key}_success"
                        ],
                        "average_episode_length": experiment[
                            f"{method_key}_length"
                        ],
                    }
                )

    return output_path


def save_text_summary():
    output_path = "results/final_analysis/final_comparative_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("FINAL COMPARATIVE ANALYSIS\n")
        file.write("==========================\n\n")

        file.write("Included scenarios:\n")
        for experiment in EXPERIMENTS:
            file.write(
                f"- {experiment['experiment']}: "
                f"{experiment['scenario'].replace(chr(10), ', ')}\n"
            )

        file.write("\nSuccess rate comparison:\n")

        for experiment in EXPERIMENTS:
            file.write(f"\n{experiment['experiment']} - ")
            file.write(f"{experiment['scenario'].replace(chr(10), ', ')}\n")
            file.write(
                f"Random baseline: {experiment['random_success']:.2f}%\n"
            )
            file.write(
                f"Greedy baseline: {experiment['greedy_success']:.2f}%\n"
            )
            file.write(
                "PPO deterministic: "
                f"{experiment['ppo_deterministic_success']:.2f}%\n"
            )
            file.write(
                "PPO stochastic: "
                f"{experiment['ppo_stochastic_success']:.2f}%\n"
            )

        file.write("\nAverage episode length comparison:\n")

        for experiment in EXPERIMENTS:
            file.write(f"\n{experiment['experiment']} - ")
            file.write(f"{experiment['scenario'].replace(chr(10), ', ')}\n")
            file.write(
                f"Random baseline: {experiment['random_length']:.2f}\n"
            )
            file.write(
                f"Greedy baseline: {experiment['greedy_length']:.2f}\n"
            )
            file.write(
                "PPO deterministic: "
                f"{experiment['ppo_deterministic_length']:.2f}\n"
            )
            file.write(
                "PPO stochastic: "
                f"{experiment['ppo_stochastic_length']:.2f}\n"
            )

        file.write("\nMain findings:\n")
        file.write(
            "1. PPO stochastic deployment achieved the highest success rate "
            "in all included scenarios.\n"
        )
        file.write(
            "2. PPO stochastic is not equivalent to random exploration. "
            "Actions are sampled from the learned PPO policy distribution, "
            "not uniformly from the action space.\n"
        )
        file.write(
            "3. Deterministic PPO became less robust in the most complex "
            "3-agent 10x10 scenario, where the joint action space increased "
            "from 16 to 64 actions.\n"
        )
        file.write(
            "4. The greedy obstacle-aware baseline remained strong, especially "
            "in terms of episode length, but it is a hand-coded local heuristic "
            "rather than a learned controller.\n"
        )
        file.write(
            "5. The 10x10 experiments increased the spatial search space and "
            "supported the analysis of grid-size generalization.\n"
        )
        file.write(
            "6. The 3-agent experiment strengthened the swarm-inspired framing "
            "by testing scalability to a larger multi-agent system.\n"
        )

    return output_path


def grouped_bar_plot(metric_suffix, ylabel, title, output_path):
    labels = [experiment["scenario"] for experiment in EXPERIMENTS]
    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 6))

    for method_index, (method_key, method_name) in enumerate(METHODS):
        values = [
            experiment[f"{method_key}_{metric_suffix}"]
            for experiment in EXPERIMENTS
        ]

        offset = (method_index - 1.5) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=method_name,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if metric_suffix == "success":
        ax.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def ppo_deployment_plot():
    output_path = "plots/ppo_deterministic_vs_stochastic_success.png"

    labels = [experiment["scenario"] for experiment in EXPERIMENTS]
    deterministic_values = [
        experiment["ppo_deterministic_success"]
        for experiment in EXPERIMENTS
    ]
    stochastic_values = [
        experiment["ppo_stochastic_success"]
        for experiment in EXPERIMENTS
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        deterministic_values,
        width,
        label="PPO deterministic",
    )

    ax.bar(
        x + width / 2,
        stochastic_values,
        width,
        label="PPO stochastic",
    )

    ax.set_title("Deterministic vs stochastic PPO deployment")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def ppo_deployment_gap_plot():
    output_path = "plots/ppo_stochastic_gain_over_deterministic.png"

    labels = [experiment["scenario"] for experiment in EXPERIMENTS]
    gains = [
        experiment["ppo_stochastic_success"]
        - experiment["ppo_deterministic_success"]
        for experiment in EXPERIMENTS
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(labels, gains)

    ax.set_title("Success-rate gain from stochastic PPO deployment")
    ax.set_ylabel("Success-rate gain (percentage points)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def main():
    ensure_output_directories()

    csv_path = save_csv_table()
    summary_path = save_text_summary()

    success_plot_path = grouped_bar_plot(
        metric_suffix="success",
        ylabel="Success rate (%)",
        title="Final success rate comparison",
        output_path="plots/final_success_rate_comparison.png",
    )

    length_plot_path = grouped_bar_plot(
        metric_suffix="length",
        ylabel="Average episode length",
        title="Final average episode length comparison",
        output_path="plots/final_episode_length_comparison.png",
    )

    ppo_plot_path = ppo_deployment_plot()
    ppo_gain_plot_path = ppo_deployment_gap_plot()

    print("===== FINAL COMPARATIVE ANALYSIS GENERATED =====")
    print(f"CSV table saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Success rate plot saved to: {success_plot_path}")
    print(f"Episode length plot saved to: {length_plot_path}")
    print(f"PPO deployment plot saved to: {ppo_plot_path}")
    print(f"PPO stochastic gain plot saved to: {ppo_gain_plot_path}")


if __name__ == "__main__":
    main()