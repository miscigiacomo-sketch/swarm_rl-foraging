import os
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


AGENT_COMPARISON = {
    "agents": ["Random Agent", "PPO Agent"],
    "success_rates": [63, 100],
    "average_rewards": [0.63, 1.00],
    "average_steps": [30.55, 4.26],
}


GRID_SIZE_COMPARISON = {
    "grids": ["5x5", "10x10"],
    "success_rates": [100, 96],
    "average_rewards": [1.00, 0.96],
    "average_steps": [3.86, 12.88],
}


def create_results_folder():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_bar_plot(labels, values, ylabel, title, filename, ylim=None):
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(ylim)

    output_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_agent_comparison():
    save_bar_plot(
        labels=AGENT_COMPARISON["agents"],
        values=AGENT_COMPARISON["success_rates"],
        ylabel="Success Rate (%)",
        title="Success Rate Comparison",
        filename="success_rate_comparison.png",
        ylim=(0, 110),
    )

    save_bar_plot(
        labels=AGENT_COMPARISON["agents"],
        values=AGENT_COMPARISON["average_steps"],
        ylabel="Average Episode Length",
        title="Average Steps Comparison",
        filename="average_steps_comparison.png",
    )


def plot_grid_size_comparison():
    save_bar_plot(
        labels=GRID_SIZE_COMPARISON["grids"],
        values=GRID_SIZE_COMPARISON["success_rates"],
        ylabel="Success Rate (%)",
        title="Grid Size Generalization - Success Rate",
        filename="grid_size_success_rate.png",
        ylim=(0, 110),
    )

    save_bar_plot(
        labels=GRID_SIZE_COMPARISON["grids"],
        values=GRID_SIZE_COMPARISON["average_steps"],
        ylabel="Average Episode Length",
        title="Grid Size Generalization - Average Steps",
        filename="grid_size_average_steps.png",
    )


def save_metrics_summary():
    output_path = os.path.join(RESULTS_DIR, "report_metrics.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("===== SWARM RL RESULTS =====\n\n")

        file.write("Agent Comparison:\n\n")

        for index, agent in enumerate(AGENT_COMPARISON["agents"]):
            file.write(f"{agent}:\n")
            file.write(
                f"Success Rate: {AGENT_COMPARISON['success_rates'][index]:.2f}%\n"
            )
            file.write(
                f"Average Reward: {AGENT_COMPARISON['average_rewards'][index]:.2f}\n"
            )
            file.write(
                f"Average Steps: {AGENT_COMPARISON['average_steps'][index]:.2f}\n\n"
            )

        file.write("Conclusion:\n")
        file.write(
            "The PPO agent significantly outperforms the random baseline, "
            "achieving a perfect success rate and requiring far fewer steps "
            "to collect the resource.\n\n"
        )

        file.write("===== GRID SIZE GENERALIZATION =====\n\n")

        for index, grid in enumerate(GRID_SIZE_COMPARISON["grids"]):
            file.write(f"{grid} Grid:\n")
            file.write(
                f"Success Rate: {GRID_SIZE_COMPARISON['success_rates'][index]:.2f}%\n"
            )
            file.write(
                f"Average Reward: {GRID_SIZE_COMPARISON['average_rewards'][index]:.2f}\n"
            )
            file.write(
                f"Average Steps: {GRID_SIZE_COMPARISON['average_steps'][index]:.2f}\n\n"
            )

        file.write("Conclusion:\n")
        file.write(
            "The PPO agent trained on the original 5x5 environment generalizes "
            "well to a larger 10x10 grid. However, the average episode length "
            "increases, showing that larger environments reduce collection efficiency.\n"
        )


def main():
    create_results_folder()
    plot_agent_comparison()
    plot_grid_size_comparison()
    save_metrics_summary()

    print("Results generated successfully.")
    print(f"Files saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()