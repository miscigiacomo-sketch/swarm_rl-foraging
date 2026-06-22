import os

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = [
    "Random",
    "Greedy",
    "PPO det.",
    "PPO stoch.",
]


FIXED_5X5 = [76.40, 93.10, 92.80, 100.00]
RANDOM_5X5 = [81.50, 93.50, 82.40, 99.40]

RANDOM_10X10_2AGENTS = [37.60, 95.10, 86.30, 99.98]
RANDOM_10X10_3AGENTS = [50.00, 96.20, 65.00, 99.82]

DEPLOYMENT_GAIN_LABELS = [
    "Fixed 5x5\n2 agents",
    "Random 5x5\n2 agents",
    "Random 10x10\n2 agents",
    "Random 10x10\n3 agents",
]

DEPLOYMENT_GAINS = [
    100.00 - 92.80,
    99.40 - 82.40,
    99.98 - 86.30,
    99.82 - 65.00,
]


def ensure_output_directories():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results/final_analysis", exist_ok=True)


def grouped_comparison_subplot(
    ax,
    before_values,
    after_values,
    before_label,
    after_label,
    title,
):
    x = np.arange(len(METHOD_LABELS))
    width = 0.35

    ax.bar(x - width / 2, before_values, width, label=before_label)
    ax.bar(x + width / 2, after_values, width, label=after_label)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=15)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)


def deployment_gain_subplot(ax):
    x = np.arange(len(DEPLOYMENT_GAIN_LABELS))

    ax.bar(x, DEPLOYMENT_GAINS)

    ax.set_title("Deployment mode effect", fontsize=11)
    ax.set_ylabel("Stochastic gain (percentage points)")
    ax.set_xticks(x)
    ax.set_xticklabels(DEPLOYMENT_GAIN_LABELS, rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def save_summary_file():
    output_path = "results/final_analysis/sensitivity_analysis_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("SENSITIVITY ANALYSIS SUMMARY\n")
        file.write("============================\n\n")

        file.write("1. Obstacle configuration effect (2 agents, 5x5):\n")
        file.write("Fixed obstacles vs random obstacles\n")
        for method, fixed_value, random_value in zip(
            METHOD_LABELS,
            FIXED_5X5,
            RANDOM_5X5,
        ):
            difference = random_value - fixed_value
            file.write(
                f"- {method}: fixed {fixed_value:.2f}%, "
                f"random {random_value:.2f}%, "
                f"difference {difference:+.2f} percentage points\n"
            )

        file.write("\n2. Grid-size effect (2 agents, random obstacles):\n")
        file.write("5x5 vs 10x10\n")
        for method, small_value, large_value in zip(
            METHOD_LABELS,
            RANDOM_5X5,
            RANDOM_10X10_2AGENTS,
        ):
            difference = large_value - small_value
            file.write(
                f"- {method}: 5x5 {small_value:.2f}%, "
                f"10x10 {large_value:.2f}%, "
                f"difference {difference:+.2f} percentage points\n"
            )

        file.write("\n3. Agent-count effect (10x10, random obstacles):\n")
        file.write("2 agents vs 3 agents\n")
        for method, two_agent_value, three_agent_value in zip(
            METHOD_LABELS,
            RANDOM_10X10_2AGENTS,
            RANDOM_10X10_3AGENTS,
        ):
            difference = three_agent_value - two_agent_value
            file.write(
                f"- {method}: 2 agents {two_agent_value:.2f}%, "
                f"3 agents {three_agent_value:.2f}%, "
                f"difference {difference:+.2f} percentage points\n"
            )

        file.write("\n4. PPO deployment-mode effect:\n")
        for label, gain in zip(DEPLOYMENT_GAIN_LABELS, DEPLOYMENT_GAINS):
            file.write(
                f"- {label.replace(chr(10), ', ')}: "
                f"stochastic gain {gain:.2f} percentage points\n"
            )

        file.write("\nMain findings:\n")
        file.write(
            "1. Randomized obstacles mainly reduced deterministic PPO "
            "performance, while stochastic PPO remained highly robust.\n"
        )
        file.write(
            "2. Increasing grid size strongly penalized the random baseline, "
            "while stochastic PPO generalized very well.\n"
        )
        file.write(
            "3. Increasing the number of agents substantially reduced "
            "deterministic PPO performance, but stochastic PPO remained "
            "near-perfect.\n"
        )
        file.write(
            "4. The success-rate gain of stochastic over deterministic PPO "
            "increased with scenario complexity.\n"
        )

    return output_path


def main():
    ensure_output_directories()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    grouped_comparison_subplot(
        ax=axes[0, 0],
        before_values=FIXED_5X5,
        after_values=RANDOM_5X5,
        before_label="Fixed obstacles",
        after_label="Random obstacles",
        title="Obstacle configuration (2 agents, 5x5)",
    )

    grouped_comparison_subplot(
        ax=axes[0, 1],
        before_values=RANDOM_5X5,
        after_values=RANDOM_10X10_2AGENTS,
        before_label="5x5",
        after_label="10x10",
        title="Grid size (2 agents, random obstacles)",
    )

    grouped_comparison_subplot(
        ax=axes[1, 0],
        before_values=RANDOM_10X10_2AGENTS,
        after_values=RANDOM_10X10_3AGENTS,
        before_label="2 agents",
        after_label="3 agents",
        title="Agent count (10x10, random obstacles)",
    )

    deployment_gain_subplot(axes[1, 1])

    fig.suptitle("Sensitivity to Key Experimental Factors", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_path = "plots/sensitivity_to_key_factors.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    summary_path = save_summary_file()

    print("===== SENSITIVITY ANALYSIS GENERATED =====")
    print(f"Plot saved to: {plot_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()