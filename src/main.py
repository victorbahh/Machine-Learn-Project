from QLearning import QLearning
from environment import ZipEnvironment

import numpy as np
import matplotlib.pyplot as plt

import yaml

if __name__ == "__main__":

    with open("src/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    numEpisodes = params["learning"]["episodes"]

    env = ZipEnvironment()
    env.initializeGrid()

    qlearning = QLearning(env)
    
    # -------------------------
    # Interactive plotting
    # -------------------------
    plt.ion()

    fig, ax = plt.subplots(figsize=(10,6))

    raw_line, = ax.plot([], [], label="Reward")
    avg_line, = ax.plot([], [], label="Moving Average")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Learning Curve")

    ax.legend()
    ax.grid(True)

    window = 100

    for episode in range(numEpisodes):

        qlearning.runEpisode(env)

        # Updates the plot every 10 episodes
        if episode % 10 == 0:

            rewards = np.array(
                qlearning.rewardsPerEpisode
            )

            episodes = np.arange(
                1,
                len(rewards)+1
            )

            raw_line.set_data(
                episodes,
                rewards
            )

            if len(rewards) >= window:

                moving_avg = np.convolve(
                    rewards,
                    np.ones(window)/window,
                    mode="valid"
                )

                avg_line.set_data(
                    np.arange(
                        window,
                        len(rewards)+1
                    ),
                    moving_avg
                )

            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    plt.show()