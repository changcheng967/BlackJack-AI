# Blackjack Deep Q-Network (DQN) AI

This repository contains a Python project that builds a blackjack-playing AI using a Deep Q-Network (DQN) implemented in PyTorch. The code dynamically selects the GPU backend based on your hardware:

- **NVIDIA GPU Users:** The project uses CUDA for acceleration. Install the dependencies from [`requirements.txt`](requirements.txt).
- **Intel/AMD GPU Users:** The project uses Microsoft's DirectML via [`torch-directml`](https://github.com/microsoft/torch-directml) (for Windows systems with supported drivers). Install the dependencies from [`requirements2.txt`](requirements2.txt).

The agent learns by playing games in the Gym Blackjack environment and is evaluated over several games when training is complete.

---

## Features

- **Dynamic GPU Support:**
  - CUDA for NVIDIA GPUs
  - DirectML for Intel/AMD GPUs on Windows
- **Deep Q-Network Agent:** Implements experience replay and target network updates.
- **OpenAI Gym Integration:** Uses Gym's Blackjack environment.
- **Training & Evaluation:** Complete loops for training the agent and then evaluating its performance.

---

## Project Structure

```
.
├── blackjack_dqn.py        # Main Python script for training and evaluation
├── requirements.txt        # Dependencies for NVIDIA (CUDA) enabled systems
├── requirements2.txt       # Dependencies for Intel/AMD (DirectML) systems on Windows
└── README.md               # This file
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/blackjack-dqn.git
cd blackjack-dqn
```

### 2. Set Up the Python Environment

We recommend using a virtual environment. For example, using `venv`:

```bash
python -m venv venv
source venv/bin/activate    # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

#### For NVIDIA GPUs (CUDA)

Make sure you've installed a CUDA‑enabled version of PyTorch as per the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). For example, you might use:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

Then install the other requirements:

```bash
pip install -r requirements.txt
```

#### For Intel/AMD GPUs (DirectML)

On Windows, ensure your GPU drivers support DirectML and install [torch-directml](https://github.com/microsoft/torch-directml). Then install:

```bash
pip install -r requirements2.txt
```

---

## Running the Project

### Training the DQN Agent

To start training, simply run the main Python script. The training loop is set to run for 5000 episodes by default, showing periodic updates on the agent's performance:

```bash
python blackjack_dqn.py
```

During training, the agent uses an epsilon-greedy strategy to balance exploration with exploitation.

### Evaluating the Agent

After training, the script automatically evaluates the agent over 1000 games, printing the numbers of wins, losses, and draws. You can also modify the evaluation parameters in the script if needed.

---

## Customization

You can adjust various hyperparameters directly in `blackjack_dqn.py`:
- **Learning Rate (`lr`)**
- **Discount Factor (`gamma`)**
- **Epsilon Settings (`epsilon`, `epsilon_min`, `epsilon_decay`)**
- **Replay Memory Size and Batch Size**
- **Neural Network Architecture** (e.g., adding extra layers or units)

Feel free to experiment with more advanced variants (such as Double DQN or Dueling DQN) to further enhance performance.

---

## Troubleshooting

- **CUDA Issues:**  
  If you encounter issues with CUDA, double-check that you have installed the correct version of PyTorch for your CUDA version and that your NVIDIA drivers are current.
  
- **DirectML Issues (Intel/AMD):**  
  Ensure you are running on a Windows machine with DirectML-compatible GPU drivers. See the [torch-directml GitHub page](https://github.com/microsoft/torch-directml) for additional setup instructions.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.