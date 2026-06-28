import numpy as np
import matplotlib.pyplot as plt

from environment import ZipEnvironment
from DQN import DQNAgent
from QLearning import QLearning

def main():
    env = ZipEnvironment()
    
    # Choose the learning method
    method = int(input(
        "Choose the method: Q-Learning (1) or DQN (2): "
    ).strip())
    
    if method == 1:
        agent = QLearning(env)
    elif method == 2:
        agent = DQNAgent(env)
    
    env.initializeGrid()

    rewards = []
    bestReward = -float("inf")

    print(f"Starting {type(agent).__name__} training...")

    # -------------------------
    # Interactive plotting
    # -------------------------
    plt.ion()

    fig, ax = plt.subplots(figsize=(10,6))

    raw_line, = ax.plot([], [], label="Reward", alpha=0.5)
    avg_line, = ax.plot([], [], label="Moving Average")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"Learning Curve ({type(agent).__name__})")

    ax.legend()
    ax.grid(True)

    window = 100

    for episode in range(agent.episodes):

        totalReward = agent.runEpisode(env)
        rewards.append(totalReward)

        if totalReward > bestReward:
            bestReward = totalReward

        if episode % 100 == 0:
            avgReward = np.mean(
                rewards[-100:]
            ) if len(rewards) >= 100 else np.mean(rewards)

            print(
                f"Episode {episode:5d} | "
                f"Reward: {totalReward:8.2f} | "
                f"Avg(100): {avgReward:8.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
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

    print("\nTraining finished.")
    print(f"Best reward: {bestReward}")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{type(agent).__name__} Training Reward")

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()