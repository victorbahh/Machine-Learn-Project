import numpy as np
import yaml

class QLearning():
    def __init__(self, env):
        # Read params from YAML file
        with open("src/params.yaml", "r") as file:
            params = yaml.safe_load(file)

        self.env = env

        # Learning parameters
        self.gamma = params["learning"]["gamma"]
        self.alpha = params["learning"]["alpha"]
        self.epsilon = params["learning"]["epsilon"]
        self.episodes = params["learning"]["episodes"]

        self.numActions = len(env.ACTIONS)

        self.episode = 0
        self.rewardsPerEpisode = []

        # The state is represented as a single integer index, but it corresponds to a combination of:
        # - Agent's position (row, col)
        # - Current target number (1, 2, ..., numTargets)
        self.numTargets = len(env.targets)
        self.numStates = (
            env.ROWS *
            env.COLS *
            self.numTargets
        )
        
        # Initialize Q-table with zeros
        self.Q = np.zeros(
            (self.numStates, len(env.ACTIONS))
        )

    # Convert state (agent position + current target) to a single index for the Q-table
    def stateToIndex(self, state):
        r, c = state

        t = self.env.currentTarget

        if t > self.numTargets:
            t = self.numTargets

        cells = self.env.ROWS * self.env.COLS

        return (
            (t-1)*cells
            + r*self.env.COLS
            + c
        )

    # Reset the environment
    def reset(self):
        return self.env.reset()

    # Epsilon-soft policy for action selection
    def tabularEpsilonSoftPolicy(self, state):

        valid = self.env.validActions(self.env.agentPos)

        if len(valid) == 0:
            return None

        qvals = self.Q[state, valid]

        # ---- Random tie-breaking among best actions ----
        maxQ = qvals.max()

        bestIndices = np.flatnonzero(
            qvals == maxQ
        )

        bestIdx = np.random.choice(bestIndices)

        bestAction = valid[bestIdx]
        # -----------------------------------------------

        n = len(valid)

        p1 = 1 - self.epsilon + self.epsilon/n
        p2 = self.epsilon/n

        prob = []

        for a in valid:
            if a == bestAction:
                prob.append(p1)
            else:
                prob.append(p2)

        prob = np.array(prob)
        prob = prob / prob.sum()

        return np.random.choice(valid, p=prob)

    # Q-Learning step
    def runQLearning(self, S):
        A = self.tabularEpsilonSoftPolicy(S)
        
        if A is None:
            return None, -500, True

        # Execute action A
        SlPos, R, done = self.env.step(self.env.ACTIONS[A])

        # Best possible next action value (greedy target)
        
        # If the episode ended, there is no next state, so we skip the Sl update
        # and use the reward R to update Q for the terminal state
        if done:
            self.Q[S,A] += self.alpha*(
                R - self.Q[S,A]
            )
            return None, R, done

        # Convert next state to index
        Sl = self.stateToIndex(SlPos)

        # Find the best Q-value for the next state Sl among valid actions
        validNext = self.env.validActions(SlPos)

        if len(validNext) == 0:
            bestNextQ = 0
        else:
            bestNextQ = np.max(
                self.Q[Sl, validNext]
            )

        # Q-Learning update
        self.Q[S,A] = self.Q[S,A] + self.alpha * (
            R + self.gamma * bestNextQ - self.Q[S,A]
        )

        return Sl, R, done

    def runEpisode(self, env):

        # Decay epsilon
        # self.epsilon = max(0.01, self.epsilon * 0.999)

        # New episode
        self.episode += 1

        # Reset environment
        S = self.stateToIndex(self.reset())

        rewards = []

        while True:
            # Q-Learning step
            Sl, R, done = self.runQLearning(S)

            # Advance state
            S = Sl

            rewards.append(R)

            # Render
            env.renderGame(self.episode)

            # End episode?
            if done == True:

                self.rewardsPerEpisode.append(
                    np.sum(np.array(rewards))
                )

                break

        return np.sum(np.array(rewards))