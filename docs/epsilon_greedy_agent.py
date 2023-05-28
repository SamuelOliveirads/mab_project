import matplotlib.pyplot as plt
import numpy as np


class EpsGreedyAgent(object):
    # The agent class with a list of bandit machines' probabilities
    def __init__(self, prob_list):
        self.prob_list = prob_list

    # Simulates pulling the bandit machine's lever
    def pull(self, bandit_machine):
        if np.random.random() < self.prob_list[bandit_machine]:
            reward = 1
        else:
            reward = 0
        return reward


def run_simulation(prob_list, trials, episodes, exploration_phase, decay):
    # valores de decaimento do eps
    eps_array = [(exploration_phase * (1 - decay)) ** i for i in range(trials)]

    # Initiate agent
    bandit = EpsGreedyAgent(prob_list)

    # Initialize reward arrays
    prob_reward_array = np.zeros(len(prob_list))
    accumulated_reward_array = list()
    avg_accumulated_reward_array = list()

    for episode in range(episodes):
        # Initialize per-episode reward and bandit arrays
        reward_array = np.zeros(len(prob_list))
        bandit_array = np.full(len(prob_list), 1.0e-5)
        accumulated_reward = 0

        for trial in range(trials):
            # Agent - choice
            eps = eps_array[trial]
            if eps >= 0.5:  # exploration
                bandit_machine = np.random.randint(low=0, high=2, size=1)[0]
            else:  # explotation
                prob_reward = reward_array / bandit_array
                max_prob_reward = prob_reward.max()
                max_reward_indices = np.where(prob_reward == max_prob_reward)
                bandit_machine = np.random.choice(max_reward_indices[0])

            # agent - reward
            reward = bandit.pull(bandit_machine)

            # update reward data
            reward_array[bandit_machine] += reward
            bandit_array[bandit_machine] += 1
            accumulated_reward += reward

        prob_reward_array += reward_array / bandit_array
        accumulated_reward_array.append(accumulated_reward)
        avg_accumulated_reward_array.append(np.mean(accumulated_reward_array))

    # Print and plot results
    for i, prob in enumerate(prob_reward_array):
        print(f"Prob Bandit {i+1}: {100*np.round(prob / episodes, 2)}")

    print(f"Avg accumulated reward: {np.mean(avg_accumulated_reward_array)}\n")

    plt.plot(avg_accumulated_reward_array)
    plt.xlabel("Episodes")
    plt.ylabel("Average accumulated reward")
    plt.show()


# Probabilities of getting a positive result from the bandit machines
prob_list = [0.1, 0.3]

# Parameters for the experiment
trials = 1000
episodes = 200
exploration_phase = 0  # initial epsilon
decay = 0.5  # decay rate for epsilon

run_simulation(prob_list, trials, episodes, exploration_phase, decay)
