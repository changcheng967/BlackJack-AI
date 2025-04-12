import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

# -----------------------------------------------------------------------------
# GPU Device Selection
# -----------------------------------------------------------------------------
# First, try CUDA (for NVIDIA GPUs). If not available, try torch-directml (for AMD/Intel on Windows).
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    try:
        import torch_directml as dml
        device = dml.device()
        print(f"Using DirectML device (AMD/Intel GPU): {device}")
    except ImportError:
        device = torch.device("cpu")
        print("No GPU acceleration available. Using CPU.")

# -----------------------------------------------------------------------------
# Deep Q-Network (DQN)
# -----------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 lr=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 memory_capacity=10000,
                 batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_capacity)

        # Policy and target networks setup: They both use the selected device.
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def update_target_network(self):
        """Copy the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
        """Train on a batch of experiences from the replay memory."""
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in minibatch]).to(device)
        actions = torch.LongTensor([exp[1] for exp in minibatch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([exp[2] for exp in minibatch]).to(device)
        next_states = torch.FloatTensor([exp[3] for exp in minibatch]).to(device)
        dones = torch.FloatTensor([1 if exp[4] else 0 for exp in minibatch]).to(device)

        # Compute Q-values for current states and the selected actions
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# -----------------------------------------------------------------------------
# State Preprocessing
# -----------------------------------------------------------------------------
def preprocess_state(state):
    """
    The Gym Blackjack environment returns a state tuple:
      (player_sum, dealer_card, usable_ace)
    We convert this into a numpy array.
    """
    player_sum, dealer_card, usable_ace = state
    ace_flag = 1.0 if usable_ace else 0.0
    return np.array([float(player_sum), float(dealer_card), ace_flag], dtype=np.float32)

# -----------------------------------------------------------------------------
# Create Blackjack Environment
# -----------------------------------------------------------------------------
def create_blackjack_env():
    try:
        # Recent versions of Gym may use 'Blackjack-v1'
        env = gym.make('Blackjack-v1')
    except Exception:
        env = gym.make('Blackjack-v0')
    return env

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_agent(num_episodes=5000, target_update_freq=10):
    env = create_blackjack_env()
    state_size = 3  # [player_sum, dealer_card, usable_ace]
    action_size = env.action_space.n  # Typically: hit or stick
    agent = DQNAgent(state_size, action_size)

    rewards_history = []
    losses_history = []

    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # For terminal states, use a dummy next state
            next_state_processed = preprocess_state(next_state) if not done else np.zeros(state_size, dtype=np.float32)
            agent.remember(state, action, reward, next_state_processed, done)
            state = next_state_processed
            total_reward += reward

            loss = agent.replay()
            if loss is not None:
                losses_history.append(loss)

        rewards_history.append(total_reward)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_loss = np.mean(losses_history[-100:]) if losses_history else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | Avg Loss: {avg_loss:.5f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("Training complete.")
    return agent

# -----------------------------------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate_agent(agent, num_games=1000):
    env = create_blackjack_env()
    wins = 0
    losses = 0
    draws = 0

    # Use a lower exploration rate during evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_games):
        state = env.reset()
        state = preprocess_state(state)
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = preprocess_state(next_state) if not done else np.zeros(3, dtype=np.float32)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"Evaluation over {num_games} games:")
    print(f"Wins: {wins}\tLosses: {losses}\tDraws: {draws}")

    agent.epsilon = original_epsilon  # Restore original epsilon

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting training for blackjack DQN agent...")
    agent = train_agent(num_episodes=5000)
    print("\nEvaluating the trained agent...")
    evaluate_agent(agent, num_games=1000)
