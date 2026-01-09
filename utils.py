import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from individual_networks import pad_state, UNIFIED_STATE_DIM

# ================= CONFIGURATION =================
ENV_COLORS = {
    "CartPole-v1": "#1f77b4",             # Blue
    "Acrobot-v1": "#ff7f0e",              # Orange
    "MountainCarContinuous-v0": "#2ca02c" # Green
}

THRESHOLDS = {
    "CartPole-v1": 475,
    "Acrobot-v1": -90,
    "MountainCarContinuous-v0": 90
}

# ================= METRICS & EVALUATION =================

def evaluate_policy_performance(policy_net, env_name, episodes=10):
    """
    Runs the trained model in deterministic mode to measure final performance.
    """
    env = gym.make(env_name)
    raw_state_dim = env.observation_space.shape[0]
    scores = []

    policy_net.eval()  # Switch to evaluation mode

    for _ in range(episodes):
        state_raw, _ = env.reset()
        state = torch.FloatTensor(pad_state(state_raw, raw_state_dim)).unsqueeze(0)
        done = False
        score = 0

        while not done:
            with torch.no_grad():
                logits = policy_net(state)
                
                # --- Reproduce Transfer Learning Masking/Mapping ---
                if env_name == "CartPole-v1":
                    mask = torch.tensor([0.0, 0.0, -float('inf')])
                    logits = logits + mask

                # Greedy selection (Argmax) for evaluation
                action_probs = F.softmax(logits, dim=-1)
                action_index = torch.argmax(action_probs)
                
                env_action = action_index.item()
                
                # Handle MountainCar Continuous mapping
                if env_name == "MountainCarContinuous-v0":
                    mapping = {0: [-1.0], 1: [0.0], 2: [1.0]}
                    env_action = mapping[action_index.item()]

            next_state_raw, reward, terminated, truncated, _ = env.step(env_action)
            state = torch.FloatTensor(pad_state(next_state_raw, raw_state_dim)).unsqueeze(0)
            score += reward
            done = terminated or truncated

        scores.append(score)
    
    env.close()
    return {
        'mean_reward': np.mean(scores),
        'std_reward': np.std(scores)
    }

def compute_convergence_metrics(rewards, threshold):
    """
    Calculates moving average and identifies convergence episode.
    """
    window = 50
    if len(rewards) < window:
        return {
            'moving_avg': rewards,
            'convergence_episode': None,
            'final_performance': np.mean(rewards),
            'reward_variance': np.var(rewards)
        }

    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # Check for threshold crossing
    converged_idx = np.where(moving_avg >= threshold)[0]
    convergence_episode = (converged_idx[0] + window) if len(converged_idx) > 0 else None

    return {
        'moving_avg': moving_avg,
        'convergence_episode': convergence_episode,
        'final_performance': np.mean(rewards[-50:]),
        'reward_variance': np.var(rewards[-50:])
    }

# ================= PLOTTING =================

def plot_comparison_generic(results):
    """
    Generates the 4-panel comparison visualization.
    Args:
        results: Dict mapping env_name -> result_dict from training
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Transfer Learning Actor-Critic: Multi-Environment Analysis", fontsize=16)
    
    window = 50

    # --- Plot 1: Training Rewards ---
    ax1 = axes[0, 0]
    for name, res in results.items():
        color = ENV_COLORS.get(name, 'black')
        rewards = res['rewards']
        
        # Raw data (faint)
        ax1.plot(rewards, alpha=0.15, color=color)
        
        # Moving Average (solid)
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(ma, label=name, color=color, linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Training Rewards (MA-{window})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Episode Lengths ---
    ax2 = axes[0, 1]
    for name, res in results.items():
        color = ENV_COLORS.get(name, 'black')
        lengths = res['lengths']
        if len(lengths) >= window:
            ma_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax2.plot(ma_len, label=name, color=color, linewidth=2)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Duration (Moving Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Convergence (Normalized) ---
    ax3 = axes[1, 0]
    for name, res in results.items():
        color = ENV_COLORS.get(name, 'black')
        rewards = res['rewards']
        threshold = THRESHOLDS.get(name, 0)
        
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # Normalize to threshold for visual comparison
        # If threshold is 0 or negative, we handle scaling carefully
        denom = abs(threshold) if threshold != 0 else 1.0
        norm_ma = ma / denom
        
        ax3.plot(norm_ma, label=f'{name}', color=color, linewidth=2)

    ax3.axhline(y=1.0 if list(THRESHOLDS.values())[0] > 0 else -1.0, 
                color='gray', linestyle='--', label='Solved Threshold (Norm)')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Score / |Threshold|')
    ax3.set_title('Convergence Speed (Normalized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Statistical Table ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    col_labels = ["Env", "Conv. Ep", "Time (s)", "Test Score"]
    table_data = []

    for name, res in results.items():
        metrics = compute_convergence_metrics(res['rewards'], THRESHOLDS.get(name, 0))
        eval_stats = evaluate_policy_performance(res['policy_net'], name)
        
        conv_ep = metrics['convergence_episode']
        conv_str = str(conv_ep) if conv_ep else "> Max"
        
        table_data.append([
            name,
            conv_str,
            f"{res['training_time']:.1f}",
            f"{eval_stats['mean_reward']:.1f} ± {eval_stats['std_reward']:.1f}"
        ])

    table = ax4.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    ax4.set_title('Performance Summary')

    plt.tight_layout()
    plt.show()

def save_models(name, actor, critic):
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(actor.state_dict(), f"models/{name}_actor.pth")
    torch.save(critic.state_dict(), f"models/{name}_critic.pth")
    print(f"Models saved for {name}")

def load_models(name, actor, critic):
    actor_path = f"models/{name}_actor.pth"
    critic_path = f"models/{name}_critic.pth"
    
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))
        print(f"Loaded existing models for {name}. Skipping training.")
        return True
    return False