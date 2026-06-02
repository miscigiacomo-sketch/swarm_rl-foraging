import os
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def create_results_folder():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_success_rate():
    agents = ["Random Agent", "PPO Agent"]
    success_rates = [63, 100]

    plt.figure()
    plt.bar(agents, success_rates)
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate Comparison")
    plt.ylim(0, 110)

    output_path = os.path.join(RESULTS_DIR, "success_rate_comparison.png")
    plt.savefig(output_path)
    plt.close()


def plot_average_steps():
    agents = ["Random Agent", "PPO Agent"]
    average_steps = [30.55, 4.26]

    plt.figure()
    plt.bar(agents, average_steps)
    plt.ylabel("Average Episode Length")
    plt.title("Average Steps Comparison")

    output_path = os.path.join(RESULTS_DIR, "average_steps_comparison.png")
    plt.savefig(output_path)
    plt.close()


def save_metrics_summary():
    output_path = os.path.join(RESULTS_DIR, "report_metrics.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("===== SWARM RL RESULTS =====\n\n")
        file.write("Random Agent:\n")
        file.write("Success Rate: 63%\n")
        file.write("Average Reward: 0.63\n")
        file.write("Average Steps: 30.55\n\n")

        file.write("PPO Agent:\n")
        file.write("Success Rate: 100%\n")
        file.write("Average Reward: 1.00\n")
        file.write("Average Steps: 4.26\n\n")

        file.write("Conclusion:\n")
        file.write(
            "The PPO agent significantly outperforms the random baseline, "
            "achieving perfect success rate and requiring far fewer steps "
            "to collect the resource.\n"
        )


def main():
    create_results_folder()
    plot_success_rate()
    plot_average_steps()
    save_metrics_summary()

    print("Results generated successfully.")
    print(f"Files saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()