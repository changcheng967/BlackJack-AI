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

# ------------------------------
# GPU Device Selection
# ------------------------------
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

# ------------------------------
# Deep Q-Network (DQN) Definition
# ------------------------------
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

# ------------------------------
# DQN Agent
# ------------------------------
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

        # Experience replay memory
        self.memory = deque(maxlen=memory_capacity)

        # Policy & target networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# ------------------------------
# Gym Environment Helpers
# ------------------------------
def preprocess_state(state):
    # Expects state: (player_sum, dealer_card, usable_ace)
    player_sum, dealer_card, usable_ace = state
    ace_flag = 1.0 if usable_ace else 0.0
    return np.array([float(player_sum), float(dealer_card), ace_flag], dtype=np.float32)

def create_blackjack_env():
    try:
        env = gym.make('Blackjack-v1')
    except Exception:
        env = gym.make('Blackjack-v0')
    return env

# ------------------------------
# Training & Evaluation
# ------------------------------
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
        if (episode+1) % target_update_freq == 0:
            agent.update_target_network()
        if (episode+1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_loss = np.mean(losses_history[-100:]) if losses_history else 0
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.3f} | Avg Loss: {avg_loss:.5f} | Epsilon: {agent.epsilon:.3f}")
    print("Training complete.")
    return agent

def evaluate_agent(agent, num_games=1000):
    env = create_blackjack_env()
    wins, losses, draws = 0, 0, 0
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

# ------------------------------
# Console-Based Human vs. AI (for reference)
# ------------------------------
def draw_card():
    cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    return random.choice(cards)

def hand_value(hand):
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
    if num_aces > 0 and total + 10 <= 21:
        total += 10
        usable = True
    else:
        usable = False
    return total, usable

# ------------------------------
# GUI for Human vs. AI Game Mode (Tkinter)
# ------------------------------
import tkinter as tk
from tkinter import messagebox

class BlackjackGUI:
    def __init__(self, agent):
        self.agent = agent
        self.root = tk.Tk()
        self.root.title("Blackjack: Human vs AI")
        # Labels for dealer, human, AI, and messages
        self.dealer_label = tk.Label(self.root, text="Dealer: ", font=("Helvetica", 14))
        self.dealer_label.pack(pady=5)
        self.human_label = tk.Label(self.root, text="Your Hand: ", font=("Helvetica", 14))
        self.human_label.pack(pady=5)
        self.ai_label = tk.Label(self.root, text="AI Hand: ", font=("Helvetica", 14))
        self.ai_label.pack(pady=5)
        self.message_label = tk.Label(self.root, text="Welcome to Blackjack!", font=("Helvetica", 14))
        self.message_label.pack(pady=10)
        # Buttons for actions
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)
        self.hit_button = tk.Button(self.button_frame, text="Hit", command=self.hit, width=10, font=("Helvetica", 12))
        self.hit_button.grid(row=0, column=0, padx=5)
        self.stand_button = tk.Button(self.button_frame, text="Stand", command=self.stand, width=10, font=("Helvetica", 12))
        self.stand_button.grid(row=0, column=1, padx=5)
        self.new_round_button = tk.Button(self.root, text="New Round", command=self.new_round, width=20, font=("Helvetica", 12))
        self.new_round_button.pack(pady=10)
        self.game_over = False
        self.new_round()
    
    def new_round(self):
        self.human_hand = [draw_card(), draw_card()]
        self.ai_hand = [draw_card(), draw_card()]
        self.dealer_hand = [draw_card(), draw_card()]
        self.game_over = False
        self.message_label.config(text="Your turn: Hit or Stand?")
        self.hit_button.config(state=tk.NORMAL)
        self.stand_button.config(state=tk.NORMAL)
        self.new_round_button.config(state=tk.DISABLED)
        self.update_display()
        
    def update_display(self):
        if not self.game_over:
            dealer_text = f"Dealer: {self.dealer_hand[0]}, ?"
        else:
            dealer_total, _ = hand_value(self.dealer_hand)
            dealer_text = f"Dealer: {', '.join(self.dealer_hand)} | Total: {dealer_total}"
        self.dealer_label.config(text=dealer_text)
        human_total, _ = hand_value(self.human_hand)
        human_text = f"Your Hand: {', '.join(self.human_hand)} | Total: {human_total}"
        self.human_label.config(text=human_text)
        ai_total, _ = hand_value(self.ai_hand)
        ai_text = f"AI Hand: {', '.join(self.ai_hand)} | Total: {ai_total}"
        self.ai_label.config(text=ai_text)
    
    def hit(self):
        if self.game_over:
            return
        self.human_hand.append(draw_card())
        human_total, _ = hand_value(self.human_hand)
        self.update_display()
        if human_total > 21:
            self.message_label.config(text="You busted!")
            self.end_round()
    
    def stand(self):
        if self.game_over:
            return
        self.hit_button.config(state=tk.DISABLED)
        self.stand_button.config(state=tk.DISABLED)
        self.message_label.config(text="You stand. Dealer's turn...")
        self.root.after(1000, self.dealer_turn)
    
    def dealer_turn(self):
        dealer_total, _ = hand_value(self.dealer_hand)
        while dealer_total < 17:
            self.dealer_hand.append(draw_card())
            dealer_total, _ = hand_value(self.dealer_hand)
        self.update_display()
        self.root.after(1000, self.ai_turn)
    
    def ai_turn(self):
        while True:
            ai_total, ai_usable = hand_value(self.ai_hand)
            dealer_card = self.dealer_hand[0]
            if dealer_card == 'A':
                dealer_value = 11
            elif dealer_card in ['J', 'Q', 'K']:
                dealer_value = 10
            else:
                dealer_value = int(dealer_card)
            observation = np.array([float(ai_total), float(dealer_value), 1.0 if ai_usable else 0.0], dtype=np.float32)
            action = self.agent.select_action(observation)
            if action == 1:
                self.ai_hand.append(draw_card())
                self.update_display()
                if hand_value(self.ai_hand)[0] > 21:
                    break
            else:
                break
        self.end_round()
    
    def end_round(self):
        self.game_over = True
        self.hit_button.config(state=tk.DISABLED)
        self.stand_button.config(state=tk.DISABLED)
        self.new_round_button.config(state=tk.NORMAL)
        self.update_display()
        dealer_total, _ = hand_value(self.dealer_hand)
        human_total, _ = hand_value(self.human_hand)
        ai_total, _ = hand_value(self.ai_hand)
        if human_total > 21:
            human_result = "Lose (Busted)"
        elif dealer_total > 21 or human_total > dealer_total:
            human_result = "Win"
        elif human_total == dealer_total:
            human_result = "Tie"
        else:
            human_result = "Lose"
        if ai_total > 21:
            ai_result = "Lose (Busted)"
        elif dealer_total > 21 or ai_total > dealer_total:
            ai_result = "Win"
        elif ai_total == dealer_total:
            ai_result = "Tie"
        else:
            ai_result = "Lose"
        result_message = f"Round Over!\nDealer Total: {dealer_total}\nYour Result: {human_result}\nAI Result: {ai_result}"
        self.message_label.config(text=result_message)
    
    def run(self):
        self.root.mainloop()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    print("Starting training for blackjack DQN agent...")
    agent = train_agent(num_episodes=5000)
    print("\nEvaluating the trained agent...")
    evaluate_agent(agent, num_games=1000)
    
    gui_choice = input("\nDo you want to play the GUI version (human vs AI)? (y/n): ").strip().lower()
    if gui_choice == "y":
        app = BlackjackGUI(agent)
        app.run()
