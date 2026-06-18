# Experiment 27 - Learned-Behavior and Failure Analysis

## Objective

The objective of this analysis is to interpret the final policy behavior observed in the multi-agent foraging experiments, with particular attention to the difference between deterministic and stochastic PPO deployment.

This experiment does not introduce a new training run. Instead, it analyzes the already completed experiments and explains why the stochastic PPO policy consistently achieved higher success rates than the deterministic PPO policy, especially in the more complex random-obstacle and 3-agent scenarios.

---

## Background

The multi-agent environments use centralized joint-action control. For `n` agents, each individual agent has four possible movement actions, which leads to a joint action space of size:

```text
4^n
```

Therefore:

```text
2 agents: 4^2 = 16 joint actions
3 agents: 4^3 = 64 joint actions
```

As the number of agents increases, the policy must select from a much larger set of joint actions. This makes the task harder because the selected action must coordinate multiple agents while avoiding collisions, invalid obstacle moves, repeated positions, and inefficient loops.

---

## Deterministic PPO vs Stochastic PPO

PPO learns a probability distribution over actions. During evaluation, this learned distribution can be used in two different ways:

1. **Deterministic deployment**: select the action with the highest probability.
2. **Stochastic deployment**: sample an action from the learned probability distribution.

The stochastic PPO policy is not equivalent to the random baseline. The random baseline samples uniformly from the full action space, without using any learned information. In contrast, stochastic PPO samples from a trained policy distribution, meaning that actions with higher learned value are more likely to be selected.

This distinction is important because the final results show that stochastic sampling helped the learned policy escape repeated or locally suboptimal action patterns.

---

## Final Comparative Results

The final comparative analysis showed the following PPO deployment behavior across the main multi-agent scenarios:

| Scenario | PPO Deterministic Success | PPO Stochastic Success |
|---|---:|---:|
| 2 agents, fixed obstacles, 5x5 | 92.80% | 100.00% |
| 2 agents, random obstacles, 5x5 | 82.40% | 99.40% |
| 2 agents, random obstacles, 10x10 | 86.30% | 99.98% |
| 3 agents, random obstacles, 10x10 | 65.00% | 99.82% |

The gap between deterministic and stochastic PPO became largest in the most complex setting: the 3-agent 10x10 random-obstacle environment.

This suggests that deterministic argmax action selection became more fragile as the joint action space increased from 16 to 64 actions.

---

## Failure Pattern Interpretation

Previous failure-case analysis showed that deterministic PPO failures were dominated by repeated-position and no-move patterns. These failures are consistent with a policy that repeatedly selects the same locally preferred action even when that action does not resolve the current spatial configuration.

In environments with obstacles and multiple agents, the highest-probability joint action is not always globally useful. A deterministic policy may repeatedly choose an action that seems locally best according to the learned policy but causes one or more agents to remain blocked, collide, or cycle between positions.

Stochastic deployment reduces this issue because it occasionally samples alternative actions that still have high probability under the learned policy. These alternative actions can help agents break out of repeated patterns and continue progressing toward the food.

---

## Effect of Random Obstacles

Randomized obstacles increase environmental variability because each reset can produce a different valid obstacle layout. This means that the policy cannot simply memorize one fixed obstacle configuration.

In the 2-agent 5x5 random-obstacle task, deterministic PPO achieved 82.40% success, while stochastic PPO achieved 99.40% success. This indicates that the learned policy distribution was robust, but deterministic action selection did not fully exploit that robustness.

The 10x10 random-obstacle evaluation further increased the spatial search space. Despite being trained on 5x5 layouts, stochastic PPO generalized strongly to 10x10, reaching 99.98% success. The random baseline dropped substantially, showing that uninformed exploration became much less effective in the larger grid.

---

## Effect of Increasing the Number of Agents

The 3-agent experiment increased the centralized joint action space from 16 to 64 possible joint actions. This created a more difficult coordination problem because each selected action controlled three agents simultaneously.

The deterministic PPO policy dropped to 65.00% success in this setting, while stochastic PPO still achieved 99.82% mean success.

This result suggests that the learned policy distribution remained useful, but selecting only the single most probable joint action was insufficiently robust in the larger action space. Sampling from the policy distribution allowed the controller to use alternative learned joint actions and avoid repeated failures.

---

## Relation to Swarm-Inspired Behavior

The project should be described as swarm-inspired rather than as a fully decentralized swarm-intelligence system.

The environment contains several swarm-relevant elements:

- multiple agents acting in a shared space;
- shared foraging objective;
- collision and position-swap constraints;
- obstacle avoidance;
- robustness under randomized environments;
- scalability from two to three agents.

However, the PPO controller is centralized and selects joint actions for all agents. Therefore, the project does not implement decentralized local-agent policies. The contribution is instead a controlled reinforcement-learning study of swarm-inspired foraging behavior under increasing environmental and multi-agent complexity.

---

## Limitations

The main limitation is that the learned PPO controller is centralized. This makes the learning problem manageable for a small number of agents, but it does not scale naturally to large swarms because the joint action space grows exponentially with the number of agents.

A second limitation is that the reward function is sparse and task-focused. The policy is rewarded primarily for reaching the food, rather than for explicit coordination, coverage, communication, or energy efficiency.

A third limitation is that the greedy baseline is a hand-coded obstacle-aware heuristic. It is useful as a strong comparison, but it is not a learned policy and does not represent the same type of controller as PPO.

---

## Main Conclusion

The learned PPO policies successfully captured useful foraging behavior in multi-agent obstacle environments. The most important behavioral finding is that stochastic PPO deployment consistently outperformed deterministic PPO deployment.

This does not mean that random behavior is better. Instead, it shows that the trained PPO policy distribution contained multiple useful actions, and that sampling from this distribution improved robustness in environments where deterministic argmax selection could become trapped in repeated or locally suboptimal action patterns.

The effect became especially clear in the 3-agent 10x10 random-obstacle environment, where the joint action space increased to 64 possible actions. Deterministic PPO became fragile, while stochastic PPO maintained near-perfect success.

This supports the final interpretation that the project demonstrates scalable, swarm-inspired foraging behavior using reinforcement learning, while also revealing an important deployment insight: in complex multi-agent environments, stochastic policy sampling can be more robust than deterministic action selection.

---

## Report-Ready Takeaway

The final results show that PPO did not merely learn a single fixed path to the food. Instead, the learned policy distribution contained a set of useful alternative actions. In simple environments, deterministic action selection was often sufficient. In more complex environments with random obstacles and a larger joint action space, deterministic selection became vulnerable to repeated action loops. Stochastic deployment used the same learned policy but sampled from its action distribution, allowing the agents to escape local failures and maintain high success rates.
