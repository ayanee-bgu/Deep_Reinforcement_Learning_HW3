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
        eval_stats = evaluate_policy_performance(res['actor'], name)
        
        conv_ep = metrics['convergence_episode']
        conv_str = str(conv_ep) if conv_ep else "> Max"
        
        table_data.append([
            name,
            conv_str,
            f"{res['time']:.1f}",
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

import pandas as pd
import numpy as np

def print_statistics(results_dict, env_configs):
    stats_data = []

    for env_name, data in results_dict.items():
        rewards = data['rewards']
        lengths = data['lengths']
        total_time = data['time']
        
        # Get config for this env
        config = env_configs[env_name]
        solved_score = config['solved_score']
        
        # 1. Calculate Moving Average
        window = 50
        moving_avgs = pd.Series(rewards).rolling(window=window).mean()
        
        # 2. Find Convergence Point
        # Find the first index where the moving average crosses the solved score
        convergence_indices = np.where(moving_avgs >= solved_score)[0]
        
        if len(convergence_indices) > 0:
            conv_ep = convergence_indices[0]
            # Estimate time to convergence (assuming linear time distribution)
            # (Time per episode * convergence episode)
            avg_time_per_ep = total_time / len(rewards)
            conv_time = avg_time_per_ep * conv_ep
            status = "Solved"
        else:
            conv_ep = len(rewards) # Did not converge
            conv_time = total_time
            status = "Not Solved"

        # 3. Calculate Total Steps (Sample Efficiency)
        # Sum of lengths up to the convergence point
        steps_to_solve = sum(lengths[:conv_ep])

        stats_data.append({
            "Environment": env_name,
            "Status": status,
            "Conv. Episode": conv_ep,
            "Conv. Time (s)": round(conv_time, 2),
            "Steps to Solve": steps_to_solve,
            "Final Avg Score": round(moving_avgs.iloc[-1], 2) if len(moving_avgs) > 0 else 0,
            "Std Dev (Last 50)": round(np.std(rewards[-50:]), 2)
        })

    # Create DataFrame for clean display
    df = pd.DataFrame(stats_data)
    
    print("\n" + "="*60)
    print(f"{'FINAL PERFORMANCE STATISTICS':^60}")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

def plot_training_results(env_name, rewards, config, window=50, save_dir="training_plots"):
    """
    Generates and saves a plot of reward history for a single environment.
    - Raw rewards are faint background lines.
    - Moving average is a thick, solid line representing the trend.
    - Solved threshold is a red dashed line.
    """
    # Ensure plot directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert list to pandas Series for easy rolling average
    reward_series = pd.Series(rewards)
    # Calculate moving average (min_periods=1 ensures line starts immediately)
    moving_avg = reward_series.rolling(window=window, min_periods=1).mean()
    
    solved_score = config['solved_score']
    episodes = np.arange(1, len(rewards) + 1)

    # --- Plotting Setup ---
    plt.figure(figsize=(10, 6)) # Standard landscape aspect ratio
    
    # 1. Plot Raw Data (Faint, noisy background signal)
    plt.plot(episodes, rewards, 
             color='tab:blue', alpha=0.2, linewidth=1.0, zorder=1, 
             label='Raw Episode Reward')
    
    # 2. Plot Moving Average (Strong, clear trend signal)
    plt.plot(episodes, moving_avg, 
             color='tab:blue', linewidth=2.5, zorder=2, 
             label=f'{window}-Episode Moving Avg')
    
    # 3. Plot Goal Threshold (Red dashed line)
    plt.axhline(y=solved_score, color='tab:red', linestyle='--', linewidth=2, zorder=3,
                label=f'Solved Threshold ({solved_score})')

    # --- Aesthetics & Labels ---
    plt.title(f"Training Convergence: {env_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True, shadow=True)
    
    # Ensure tight layout so labels aren't cut off
    plt.tight_layout()
    
    # Construct filename and save
    safe_name = env_name.replace("-", "_")
    filepath = os.path.join(save_dir, f"{safe_name}_convergence.png")
    plt.savefig(filepath, dpi=150) # dpi=150 is good for documents
    print(f"Plot saved to: {filepath}")
    
    # Close plot to free memory (crucial if running many in a loop)
    plt.close()