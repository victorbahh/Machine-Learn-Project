import random
import pygame
import yaml
import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from DQNNetwork import DQNNetwork

class DQNAgent:

    def __init__(self, env):
        with open("src/params.yaml", "r") as file:
            params = yaml.safe_load(file)

        self.env = env

        self.gamma = params["learning"]["gamma"]
        self.epsilon = params["learning"]["epsilon"]
        self.episodes = params["learning"]["episodes"]

        self.lr = 0.001
        self.batch_size = 128

        self.n_step_buffer = deque(maxlen=5)
        
        self.memory = []
        self.success_memory = deque(maxlen=5000)

        self.num_actions = len(env.ACTIONS)

        self.input_dim = (
            2 +                      # posição atual
            1 +                      # número do target atual
            2 +                      # posição do target atual
            env.ROWS * env.COLS      # visitados
        )
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DQNNetwork(
            self.input_dim,
            self.num_actions
        ).to(self.device)

        # Target network para estabilizar o treinamento
        self.target_model = DQNNetwork(
            self.input_dim,
            self.num_actions
        ).to(self.device)

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        
        self.loss_fn = nn.SmoothL1Loss() # Use Smooth L1 Loss for stability

        self.rewardsPerEpisode = []

        self.episode = 0
        
        # Frequência de atualização da target network
        self.target_update_frequency = 100
        
        self.current_episode = []
        
    def getStateVector(self):
        state = []

        # Normaliza a posição do agente para [0,1]
        state.append(
            self.env.agentPos[0] / self.env.ROWS
        )

        state.append(
            self.env.agentPos[1] / self.env.COLS
        )

        # Normaliza o índice do target atual para [0,1]
        state.append(
            self.env.currentTarget /
            len(self.env.targets)
        )

        # Normaliza a posição do target atual para [0,1]
        target_pos = self.env.targets.get(self.env.currentTarget, (0, 0))
        state.append(
            target_pos[0] / self.env.ROWS
        )
        
        state.append(
            target_pos[1] / self.env.COLS
        )

        # Adiciona um vetor binário indicando quais células foram visitadas
        for r in range(self.env.ROWS):
            for c in range(self.env.COLS):
                if (r,c) in self.env.visited:
                    state.append(1.0)
                else:
                    state.append(0.0)

        return np.array(
            state,
            dtype=np.float32
        )
        
    def chooseAction(self, state):
        valid = self.env.validActions(
            self.env.agentPos
        )

        if len(valid) == 0:
            return None

        if np.random.rand() < self.epsilon:
            return random.choice(valid)

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(
                state_tensor
            ).cpu().numpy()[0]

        masked_q = np.full(
            self.num_actions,
            -np.inf
        )

        for a in valid:
            masked_q[a] = q_values[a]

        return np.argmax(masked_q)
    
    def updateTargetNetwork(self):
        self.target_model.load_state_dict(
            self.model.state_dict()
        )
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        success_batch_size = min(
            len(self.success_memory),
            self.batch_size // 5      # 20%
        )

        normal_batch_size = (
            self.batch_size -
            success_batch_size
        )

        batch = (
            random.sample(
                self.memory,
                normal_batch_size
            )
            +
            random.sample(
                self.success_memory,
                success_batch_size
            )
        )

        random.shuffle(batch)

        states = []
        targets = []

        for (state, action, reward, next_state, done) in batch:

            state_tensor = torch.FloatTensor(
                state
            ).to(self.device)

            current_q = self.model(
                state_tensor
            ).detach().cpu().numpy()

            target_q = current_q.copy()

            if done:
                target_q[action] = reward

            else:
                next_tensor = torch.FloatTensor(
                    next_state
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    next_q = self.target_model(
                        next_tensor
                    )

                max_next_q = torch.max(
                    next_q
                ).item()

                target_q[action] = (
                    reward +
                    self.gamma * max_next_q
                )

            states.append(state)
            targets.append(target_q)

        states = torch.FloatTensor(
            np.array(states)
        ).to(self.device)

        targets = torch.FloatTensor(
            np.array(targets)
        ).to(self.device)

        predictions = self.model(states)

        loss = self.loss_fn(
            predictions,
            targets
        )
        
        if self.episode > 1000:
            self.lr = 0.0005
        elif self.episode > 2000:
            self.lr = 0.0001
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def runStep(self):
        state = self.getStateVector()
        action = self.chooseAction(state)

        if action is None:
            return 0, True

        next_pos, reward, done = (
            self.env.step(
                self.env.ACTIONS[action]
            )
        )
        
        next_state = self.getStateVector()

        # Store the transition in the n-step buffer
        self.n_step_buffer.append(
            (state, action, reward, next_state, done)
        )

        if done:
            # If the episode is done, we need to process all remaining transitions in the n-step buffer
            while len(self.n_step_buffer) > 0:

                n_step_return = sum(
                    (self.gamma ** i) * transition[2]
                    for i, transition in enumerate(self.n_step_buffer)
                )

                first_state, first_action, _, _, _ = self.n_step_buffer[0]

                last_next_state = self.n_step_buffer[-1][3]
                last_done = self.n_step_buffer[-1][4]

                transition = (
                    first_state,
                    first_action,
                    n_step_return,
                    last_next_state,
                    last_done
                )
                
                self.current_episode.append(transition)
                self.memory.append(transition)

                self.n_step_buffer.popleft()
        else:
            # If the n-step buffer is full, we can process the oldest transition
            if len(self.n_step_buffer) == self.n_step_buffer.maxlen:

                n_step_return = sum(
                    (self.gamma ** i) * transition[2]
                    for i, transition in enumerate(self.n_step_buffer)
                )

                first_state, first_action, _, _, _ = self.n_step_buffer[0]
                last_next_state = self.n_step_buffer[-1][3]
                last_done = self.n_step_buffer[-1][4]

                self.memory.append((
                    first_state,
                    first_action,
                    n_step_return,
                    last_next_state,
                    last_done
                ))

                self.n_step_buffer.popleft()

        for _ in range(3):
            self.train()
        return reward, done
    
    def runEpisode(self, env):
        self.episode += 1
        self.env.reset()

        rewards = []
        max_steps = 100

        for _ in range(max_steps):

            reward, done = self.runStep()
            rewards.append(reward)

            env.renderGame(self.episode)

            if done:
                # print(f"Episode {self.episode} finished after {step+1} steps with total reward {np.sum(rewards)}")
                # print(env.visited)
                break
                
        self.epsilon = self.epsilon * 0.999

        total_reward = np.sum(rewards)

        # If the episode is done and the total reward is high,
        # we can train multiple times to reinforce the learning
        if total_reward >= 1000:
            # print(f"Episode {self.episode} finished with total reward: {total_reward}")
            for transition in self.current_episode:
                self.success_memory.append(transition)
            
            # for _ in range(10):
            #     self.train()
                
        # Clear the current episode buffer after processing
        self.current_episode.clear()

        self.rewardsPerEpisode.append(
            total_reward
        )

        if (self.episode % self.target_update_frequency) == 0:
            self.updateTargetNetwork()

        return total_reward