from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def plot_beta_distributions(
    success_counts: List[int],
    failure_counts: List[int],
    num_bandits: int,
    episode: int,
    total_episodes: int,
) -> None:
    """
    Plots the beta distributions for each bandit machine.

    Args:
        success_counts: List of the counts of successful outcomes for
        each bandit.
        failure_counts: List of the counts of failed outcomes for each
        bandit.
        num_bandits: Total number of bandit machines.
        episode: Current episode number.
        total_episodes: Total number of episodes.
    """
    line_styles = ["-", "--"]
    x_values = np.linspace(0, 1, 1002)[1:-1]

    plt.clf()
    plt.xlim(0, 1)
    plt.ylim(0, 30)

    for i in range(num_bandits):
        dist = beta(success_counts[i], failure_counts[i])
        plt.plot(
            x_values,
            dist.pdf(x_values),
            ls=line_styles[i % len(line_styles)],
            c="black",
            label=f"Alpha:{success_counts[i]}, Beta:{failure_counts[i]}",
        )

    plt.legend(loc=0)
    plt.title(f"Episode {episode+1} of {total_episodes}")
    plt.draw()
    plt.pause(0.01)


class ThompsonAgent(object):
    """
    The ThompsonAgent class, which represents an agent interacting with
    a bandit machine.
    """

    def __init__(self, success_probabilities: List[float]):
        self.success_probabilities = success_probabilities

    def pull_lever(self, bandit_machine: int) -> int:
        """
        Simulates pulling the lever of a bandit machine.
        """
        if np.random.random() < self.success_probabilities[bandit_machine]:
            probability_sucess = 1
        else:
            probability_sucess = 0

        return probability_sucess


def run_simulation(
    success_probabilities: List[float], num_trials: int, num_episodes: int
) -> None:
    """
    Runs a simulation of a Thompson Sampling multi-armed bandit problem.

    Args:
        success_probabilities: List of probabilities of getting a successful
        outcome from each bandit machine.
        num_trials: Number of trials in each episode.
        num_episodes: Number of episodes.
    """
    # Initiate agent
    bandit_agent = ThompsonAgent(success_probabilities)

    # Initialize reward and count arrays
    estimated_probabilities = np.zeros(len(success_probabilities))
    total_rewards = []
    avg_rewards = []

    for episode in range(num_episodes):
        success_counts = np.ones(len(success_probabilities))
        failure_counts = np.full(len(success_probabilities), 1.0e-5)
        reward_counts = np.zeros(len(success_probabilities))
        lever_counts = np.full(len(success_probabilities), 1.0e-5)
        accumulated_reward = 0

        for trial in range(num_trials):
            # Agent - choice
            estimated_probabilities = np.random.beta(success_counts, failure_counts)
            chosen_bandit = np.argmax(estimated_probabilities)

            # agent - reward
            reward = bandit_agent.pull_lever(chosen_bandit)

            if reward == 1:
                success_counts[chosen_bandit] += 1
            else:
                failure_counts[chosen_bandit] += 1

            # Plot beta distributions
            if episode == 0:
                plot_beta_distributions(
                    success_counts,
                    failure_counts,
                    len(success_probabilities),
                    episode,
                    num_episodes,
                )

            # Update reward data
            reward_counts[chosen_bandit] += reward
            lever_counts[chosen_bandit] += 1
            accumulated_reward += reward

        # Update estimated probabilities at the end of each episode
        estimated_probabilities = reward_counts / lever_counts
        total_rewards.append(accumulated_reward)
        avg_rewards.append(np.mean(total_rewards))

    # Print and plot results
    for i, prob in enumerate(estimated_probabilities):
        print(f"Estimated Prob Bandit {i+1}: {100*np.round(prob, 2)}%")

    print(f"Average accumulated reward: {np.mean(avg_rewards)}\n")

    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average accumulated reward")
    plt.show()


# Probabilities of getting a positive result from the bandit machines
success_probabilities = [0.40, 0.5]

# Parameters for the experiment
num_trials = 1000
num_episodes = 200

run_simulation(success_probabilities, num_trials, num_episodes)
