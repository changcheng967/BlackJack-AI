# Monkey patch to ensure np.bool8 exists.
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# -----------------------------------------------------------------------------
# GPU Device Selection
# -----------------------------------------------------------------------------
# Attempt to use CUDA (for NVIDIA); if not available, try DirectML (AMD/Intel on Windows);
# otherwise, fall back to CPU.
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
# Deep Q-Network (DQN) Definition
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
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay memory storage.
        self.memory = deque(maxlen=memory_capacity)

        # Policy and target networks.
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def update_target_network(self):
        """Copy the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Select an action using an epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
        """Sample a batch of experiences from memory and learn from them."""
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in minibatch]).to(device)
        actions = torch.LongTensor([exp[1] for exp in minibatch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([exp[2] for exp in minibatch]).to(device)
        next_states = torch.FloatTensor([exp[3] for exp in minibatch]).to(device)
        dones = torch.FloatTensor([1 if exp[4] else 0 for exp in minibatch]).to(device)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon after each replay.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# -----------------------------------------------------------------------------
# Gym Environment Helpers (used during training/evaluation)
# -----------------------------------------------------------------------------
def preprocess_state(state):
    """
    Expects the Gym state as a tuple [player_sum, dealer_card, usable_ace].
    Returns a numpy array of floats.
    """
    player_sum, dealer_card, usable_ace = state
    ace_flag = 1.0 if usable_ace else 0.0
    return np.array([float(player_sum), float(dealer_card), ace_flag], dtype=np.float32)

def create_blackjack_env():
    try:
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
    action_size = env.action_space.n  # 0: stick, 1: hit
    agent = DQNAgent(state_size, action_size)

    rewards_history = []
    losses_history = []

    for episode in range(num_episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            elif len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                raise ValueError("Unexpected number of items returned by env.step()")

            next_state_processed = preprocess_state(next_state) if not done else np.zeros(3, dtype=np.float32)
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
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.3f} | Avg Loss: {avg_loss:.5f} | Epsilon: {agent.epsilon:.3f}")

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

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # disable exploration for evaluation

    for _ in range(num_games):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state = preprocess_state(state)
        done = False
        while not done:
            action = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            elif len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                raise ValueError("Unexpected number of items returned by env.step()")
            state = preprocess_state(next_state) if not done else np.zeros(3, dtype=np.float32)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"Evaluation over {num_games} games:")
    print(f"Wins: {wins}\tLosses: {losses}\tDraws: {draws}")

    agent.epsilon = original_epsilon

# -----------------------------------------------------------------------------
# --- Additional Helpers for Human-vs-AI Game Mode ---
# -----------------------------------------------------------------------------
def draw_card():
    """Draw a card from an infinite deck."""
    cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    return random.choice(cards)

def hand_value(hand):
    """
    Calculate the blackjack value of a hand.
    Aces count as 1 by default, but if an Ace is present and adding 10 doesn't bust, count one as 11.
    Returns (total, usable_ace) where usable_ace is True if an ace is counted as 11.
    """
    total = 0
    num_aces = 0
    for card in hand:
        if card in ['J', 'Q', 'K']:
            total += 10
        elif card == 'A':
            total += 1
            num_aces += 1
        else:
            total += int(card)
    usable = False
    if num_aces > 0 and total + 10 <= 21:
        total += 10
        usable = True
    return total, usable

def play_human_vs_ai(agent, rounds=5):
    """
    Play an interactive blackjack game where both a human and the trained AI
    play against the dealer. The results are tallied and displayed.
    """
    human_wins = 0
    ai_wins = 0
    ties = 0

    for r in range(rounds):
        print("\n=== Round", r+1, "===")
        # Deal initial hands.
        human_hand = [draw_card(), draw_card()]
        ai_hand = [draw_card(), draw_card()]
        dealer_hand = [draw_card(), draw_card()]

        # Show dealer's up card.
        print("Dealer shows:", dealer_hand[0])
        print("Your hand:", human_hand, "Total:", hand_value(human_hand)[0])
        
        # Human player's turn.
        while True:
            total, _ = hand_value(human_hand)
            if total > 21:
                print("You bust with a total of:", total)
                break
            decision = input("Do you want to (h)it or (s)tand? ").lower().strip()
            if decision == "h":
                card = draw_card()
                human_hand.append(card)
                total, _ = hand_value(human_hand)
                print("You drew:", card, "| Your hand:", human_hand, "| Total:", total)
                if total > 21:
                    print("You bust!")
                    break
            elif decision == "s":
                print("You stand with a total of:", total)
                break
            else:
                print("Invalid input. Please enter 'h' to hit or 's' to stand.")
                
        human_total, _ = hand_value(human_hand)

        # AI player's turn.
        print("\nAI player's turn:")
        while True:
            ai_total, ai_usable = hand_value(ai_hand)
            # Prepare observation: [player total, dealer's up card value, usable ace flag].
            dealer_card = dealer_hand[0]
            if dealer_card == 'A':
                dealer_value = 11
            elif dealer_card in ['J', 'Q', 'K']:
                dealer_value = 10
            else:
                dealer_value = int(dealer_card)
            observation = np.array([float(ai_total), float(dealer_value), 1.0 if ai_usable else 0.0], dtype=np.float32)
            action = agent.select_action(observation)
            # (Assuming action: 1 = hit, 0 = stand)
            if action == 1:
                card = draw_card()
                ai_hand.append(card)
                ai_total, ai_usable = hand_value(ai_hand)
                print("AI hits, draws:", card, "| AI hand:", ai_hand, "| Total:", ai_total)
                if ai_total > 21:
                    print("AI busts!")
                    break
            else:
                print("AI stands with a total of:", ai_total)
                break
        ai_total, _ = hand_value(ai_hand)

        # Dealer's turn.
        print("\nDealer's turn:")
        print("Dealer's hand:", dealer_hand, "| Total:", hand_value(dealer_hand)[0])
        while True:
            dealer_total, _ = hand_value(dealer_hand)
            if dealer_total < 17:
                card = draw_card()
                dealer_hand.append(card)
                print("Dealer draws:", card, "| Dealer's hand:", dealer_hand, "| Total:", hand_value(dealer_hand)[0])
                if hand_value(dealer_hand)[0] > 21:
                    print("Dealer busts!")
                    break
            else:
                break
        dealer_total, _ = hand_value(dealer_hand)
        print("Dealer stands with a total of:", dealer_total)
        
        # Determine outcomes for human and AI.
        # For human:
        if human_total > 21:
            human_result = "lose"
        elif dealer_total > 21 or human_total > dealer_total:
            human_result = "win"
        elif human_total == dealer_total:
            human_result = "tie"
        else:
            human_result = "lose"
        # For AI:
        if ai_total > 21:
            ai_result = "lose"
        elif dealer_total > 21 or ai_total > dealer_total:
            ai_result = "win"
        elif ai_total == dealer_total:
            ai_result = "tie"
        else:
            ai_result = "lose"
        
        print("\nRound", r+1, "results:")
        print("Your total:", human_total, "->", human_result)
        print("AI total:", ai_total, "->", ai_result)
        print("Dealer total:", dealer_total)
        
        if human_result == "win":
            human_wins += 1
        if ai_result == "win":
            ai_wins += 1
        if human_result == "tie" or ai_result == "tie":
            ties += 1

    print("\nFinal results after", rounds, "rounds:")
    print("You won:", human_wins, "rounds")
    print("AI won:", ai_wins, "rounds")
    print("Ties:", ties)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting training for blackjack DQN agent...")
    agent = train_agent(num_episodes=5000)
    print("\nEvaluating the trained agent...")
    evaluate_agent(agent, num_games=1000)
    
    play_choice = input("\nDo you want to challenge the AI in a human vs AI game? (y/n): ").strip().lower()
    if play_choice == 'y':
        rounds_input = input("How many rounds do you want to play? (default 5): ").strip()
        rounds = int(rounds_input) if rounds_input.isdigit() else 5
        play_human_vs_ai(agent, rounds)
