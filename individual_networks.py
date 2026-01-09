import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

# from utils import load_models
UNIFIED_STATE_DIM = 6 
UNIFIED_ACTION_DIM = 3 
HIDDEN_SIZE = 128

# --- Hyperparameters ---
# --- Environment Specific Configurations ---
# This is the key to making them all work in one script.
CART_POLE = "CartPole-v1"
MOUNTAIN_CART = "MountainCarContinuous-v0"
ACROBOT = "Acrobot-v1"
ENV_CONFIGS = {
    CART_POLE: {
        "lr_actor": 7e-4,       # Learn fast
        "lr_critic": 1e-3,      # Critic needs to be aggressive
        "entropy_start": 0.5,
        "entropy_decay": 0.995,  # Decay FAST. Stop exploring, start balancing.
        "entropy_end": 0.001,
        "max_episodes": 2000,
        "solved_score": 475,
        "precision_trigger": 420, # When to switch to "Pro Mode"
        "reward_scale": 0.01    # Scale 1.0 -> 0.01
    },
    ACROBOT: {
        "lr_actor": 7e-4,       # Learn slow and steady
        "lr_critic": 1e-3,
        "entropy_start": 1.0,
        "entropy_decay": 0.995,
        "entropy_end": 0.0001,
        "max_episodes": 2000,
        "solved_score": -90,
        "precision_trigger": -110, # When to switch to "Pro Mode"
        "reward_scale": 1
    },
    MOUNTAIN_CART: {
        "lr_actor": 1e-3,       # Medium speed
        "lr_critic": 1e-3,
        "entropy_start": 1.0,   # HIGH exploration needed initially
        "entropy_decay": 0.995, # Very slow decay to find the swing
        "entropy_end": 0.001,
        "max_episodes": 2000,
        "solved_score": 90,
        "precision_trigger": 50, # When to switch to "Pro Mode"
        "reward_scale": 1.0     # Complex shaping used instead
    }
}

# --- Unified Architecture Constants ---
# Max input dimension (Acrobot has 6)
UNIFIED_STATE_DIM = 6 
# Max action dimension (Acrobot has 3)
UNIFIED_ACTION_DIM = 3 

class PolicyNetwork(nn.Module):
    def __init__(self, gain):
        super(PolicyNetwork, self).__init__()
        # Fixed input and output sizes for Transfer Learning requirement
        self.fc1 = nn.Linear(UNIFIED_STATE_DIM, HIDDEN_SIZE)
        self.ln1 = nn.LayerNorm(HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, UNIFIED_ACTION_DIM)

        nn.init.orthogonal_(self.fc1.weight, gain=gain)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x) 
        x = torch.tanh(x)
        return self.fc2(x)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # Fixed input size
        self.fc1 = nn.Linear(UNIFIED_STATE_DIM, 128)
        self.ln1 = nn.LayerNorm(HIDDEN_SIZE) # LayerNorm for Critic too
        self.fc2 = nn.Linear(128, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.tanh(x)
        return self.fc2(x)

def pad_state(state, original_dim):
    """
    Pads the input state with zeros to reach UNIFIED_STATE_DIM (6).
    """
    padded = np.zeros(UNIFIED_STATE_DIM)
    padded[:original_dim] = state
    return padded

def get_shaped_reward(env_name, raw_reward, state, next_state, config):
    if env_name == MOUNTAIN_CART:
        # Encourages swinging and climbing
        pos, vel = next_state[0], next_state[1]
        shaped = raw_reward + (abs(vel) * 3.0)
        if pos > -0.5: shaped += (pos + 0.2) * 5.0
        if raw_reward > 50: shaped += 100.0
        return shaped
    elif env_name == ACROBOT:
        # Acrobot State: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), vel1, vel2]
        # We want to reward being "high up".
        # The height is related to -cos(theta1) - cos(theta1 + theta2).
        # But a simple proxy is just penalizing "hanging down".
        
        # Reward based on height (inverted cosine: -1 is down, 1 is up)
        # state[0] is cos(theta1). We want it to be -1 (upwards) in standard Acrobot coords? 
        # Actually, in gym: 1.0 is down, -1.0 is up.
        
        # Simple Shaping: Reward getting the tip closer to the line
        # This encourages swinging higher.
        height_reward = -state[0] - state[2] # Reward negative cosine (height)
        
        # Scale it so it doesn't overpower the main goal
        return raw_reward + (2.0 * height_reward)
    else:
        return raw_reward * config['reward_scale']

def train_actor_critic(env_name=CART_POLE, reuse_saved=False):
    env = gym.make(env_name)
    
    # Identify environment specific dimensions for padding/masking logic
    raw_state_dim = env.observation_space.shape[0]
    
    # Load specific config
    config = ENV_CONFIGS[env_name]

    # Initialize networks with the UNIFIED architecture
    actor = PolicyNetwork(1.73 if env_name == CART_POLE else np.sqrt(2))
    critic = ValueNetwork()
    
    # Optimizers with environment-specific Learning Rates
    actor_opt = optim.Adam(actor.parameters(), lr=config['lr_actor'])
    critic_opt = optim.Adam(critic.parameters(), lr=config['lr_critic'])
    
    if env_name == CART_POLE:
        # --- ADDED: LR SCHEDULERS ---
        # Decays LR by 0.9 every 100 episodes. Prevents late-game collapse.
        actor_scheduler = optim.lr_scheduler.StepLR(actor_opt, step_size=100, gamma=0.9)
        critic_scheduler = optim.lr_scheduler.StepLR(critic_opt, step_size=100, gamma=0.9)
    episode_rewards = []
    episode_lengths = []
    current_entropy = config['entropy_start']
    start_time = time.time()

    # --- DEBUG UTILS ---
    debug_probs = []
    debug_grads = []
    
    print(f"\nTraining on: {env_name}")
    print(f"Config: LR_Actor={config['lr_actor']}, Ent_Decay={config['entropy_decay']}")
    print(f"{'Episode':>8} | {'Score':>6} | {'Avg':>6} | {'Entropy':>8}")
    print("-" * 65)

    for episode in range(config['max_episodes']):
        state_raw, _ = env.reset()
        
        # --- TRANSFER LEARNING: PAD INPUT ---
        state_padded = pad_state(state_raw, raw_state_dim)
        state = torch.FloatTensor(state_padded).unsqueeze(0) 

        score = 0
        steps = 0
        done = False

        while not done:
            # --- Actor Step ---
            action_logits = actor(state)

            # 2. ACTION MASKING (For CartPole)
            # CartPole uses actions 0,1. Ignore action 2.
            if env_name == CART_POLE:
                action_logits = action_logits[:, :2]
            
            # For Acrobot and MountainCar, we use all 3 logits (no masking needed).
            # Note: For MountainCarContinuous, we treat the 3 outputs as discrete choices.

            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action_index = dist.sample()

            # Debugging: Track probabilities of action 0 vs 1
            if episode % 20 == 0 and env_name == CART_POLE:
                debug_probs.append(action_probs.detach().numpy()[0])

            # --- CONVERT NETWORK OUTPUT TO ENV ACTION ---
            env_action = action_index.item()
            
            if env_name == MOUNTAIN_CART:
                # Map discrete network outputs to continuous actions
                # 0 -> Drive Left (-1.0)
                # 1 -> Neutral (0.0)
                # 2 -> Drive Right (1.0)
                mapping = {0: [-1.0], 1: [0.0], 2: [1.0]}
                env_action = mapping[action_index.item()]

            # --- Environment Step ---
            next_state_raw, raw_reward, terminated, truncated, _ = env.step(env_action)
            
            # Pad next state
            next_state_padded = pad_state(next_state_raw, raw_state_dim)
            next_state = torch.FloatTensor(next_state_padded).unsqueeze(0)
            
            score += raw_reward
            steps += 1
            done = terminated or truncated
            
            # --- Critic Step ---
            # Acrobot and MountainCar have -1 rewards per step, no scaling needed usually.
            # CartPole rewards are +1. Scaling helps convergence.
            # Use SHAPED reward for training (dense signal), RAW reward for score (logging)
            shaped_reward = get_shaped_reward(env_name, raw_reward, state_raw, next_state_raw, config)
            done_mask = 0.0 if terminated else 1.0
            value = critic(state)
            
            with torch.no_grad():
                next_value = critic(next_state)
                # Apply bootstrap logic
                target = shaped_reward + (0.99 * next_value * done_mask)
            
            delta = target - value

            # --- Critic Update ---
            if env_name == CART_POLE:
                critic_loss = F.smooth_l1_loss(value, target)
            else:
                critic_loss = F.mse_loss(value, target)
            critic_opt.zero_grad()
            critic_loss.backward()
            # Clip gradients to prevent explosion during the "swinging" phase
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # --- Actor Update ---
            log_prob = dist.log_prob(action_index)
            entropy = dist.entropy()
            
            actor_loss = - (delta.detach() * log_prob) - (current_entropy * entropy)
            
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0) # Stability
            
            # Debugging: Track Gradient Norms
            if episode % 20 == 0:
                total_norm = 0.0
                for p in actor.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item()
                debug_grads.append(total_norm)
            
            actor_opt.step()

            state = next_state
            state_raw = next_state_raw

        episode_rewards.append(score)
        episode_lengths.append(steps)
        avg_score = np.mean(episode_rewards[-50:])
        current_entropy = max(config['entropy_end'], current_entropy * config['entropy_decay'])
        # if episode % 50 == 0:
        #     print(f"{episode:>8} | {score:>6.0f} | {avg_score:>6.1f} | {current_entropy:>8.4f}")
            
        #     if avg_score > config['solved_score']:
        #         print(f"--> Solved {env_name}!")
        #         # save_models(env_name, actor, critic)
        #         break
        
        if env_name == CART_POLE:
            actor_scheduler.step()
            critic_scheduler.step()
        
        # --- PRINT PROGRESS ---
        if episode % 20 == 0:
            # Calculate Debug Stats
            avg_prob_str = "N/A"
            avg_grad_str = "0.0"
            if len(debug_probs) > 0:
                p_mean = np.mean(debug_probs, axis=0)
                avg_prob_str = f"[{p_mean[0]:.2f}, {p_mean[1]:.2f}]"
                debug_probs = [] # Reset
            if len(debug_grads) > 0:
                avg_grad_str = f"{np.mean(debug_grads):.4f}"
                debug_grads = []

            print(f"{episode:>8} | {score:>6.0f} | {avg_score:>6.1f} | {current_entropy:>8.4f} | {avg_prob_str:>18} | {avg_grad_str:>8}")
            
            if avg_score > config['solved_score']:
                print(f"--> Solved {env_name}!")
                break

    env.close()
    
    # --- RETURN STATISTICS FOR EVALUATION SCRIPT ---
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'training_time': time.time() - start_time,
        'policy_net': actor,
        'value_net': critic
    }

def train_agent_batch(env_name="CartPole-v1"):
    env = gym.make(env_name)
    raw_dim = env.observation_space.shape[0]
    config = ENV_CONFIGS[env_name]

    # Initialize Networks
    actor = PolicyNetwork(1.73 if env_name == CART_POLE else np.sqrt(2))
    critic = ValueNetwork()
    
    # Combined Optimizer for stability (often works better for shared dynamics)
    # Using a slightly lower LR for the Actor to prevent "panic forgetting"
    actor_opt = optim.Adam(actor.parameters(), lr=config['lr_actor'])
    critic_opt = optim.Adam(critic.parameters(), lr=config['lr_critic'])
    
    scheduler = optim.lr_scheduler.StepLR(actor_opt, step_size=200, gamma=0.9)

    episode_rewards = []
    episode_lengths = []
    current_entropy = config['entropy_start']
    start_time = time.time()
    precision_mode = False

    print(f"\n========================================================")
    print(f"Unified Batch Training: {env_name}")
    print(f"Goal: {config['solved_score']} | Precision Trigger: {config['precision_trigger']}")
    print(f"========================================================")
    print(f"{'Episode':>8} | {'Score':>6} | {'Avg':>6} | {'Entropy':>8} | {'Mode':>10}")

    for episode in range(1, config['max_episodes'] + 1):
        state_raw, _ = env.reset()
        state = torch.FloatTensor(pad_state(state_raw, raw_dim)).unsqueeze(0)
        
        # --- BATCH STORAGE ---
        log_probs = []
        values = []
        rewards = []
        raw_rewards = []
        entropies = []
        done = False
        steps = 0

        while not done:
            # 1. Action Selection
            logits = actor(state)
            if env_name == "CartPole-v1": logits = logits[:, :2]
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # 2. Record Data
            log_probs.append(dist.log_prob(action))
            values.append(critic(state))
            entropies.append(dist.entropy())

            # 3. Step
            env_act = action.item()
            if env_name == MOUNTAIN_CART:
                # Discrete 0,1,2 -> Continuous -1, 0, 1
                mapping = {0: [-1.0], 1: [0.0], 2: [1.0]}
                env_act = mapping[env_act]
            
            next_state_raw, raw_reward, term, trunc, _ = env.step(env_act)
            shaped_reward = get_shaped_reward(env_name, raw_reward, state_raw, next_state_raw, config)
            rewards.append(shaped_reward)  # Use this to calculate Loss
            raw_rewards.append(raw_reward)

            state = torch.FloatTensor(pad_state(next_state_raw, raw_dim)).unsqueeze(0)            
            state_raw = next_state_raw
            
            steps += 1
            done = term or trunc

        # --- THE MAGIC: BATCH UPDATE AT END OF EPISODE ---
        
        # 1. Calculate Discounted Returns (The "True" Score)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # 2. CRITICAL: Normalize Returns (Stabilizes Gradient Variance)
        # This prevents the "Score 500 -> Score 20" crash
        # 2. NORMALIZE RETURNS (The Secret Sauce)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean() # Fallback for single step episodes
        
        values_tens = torch.cat(values).squeeze()
        log_probs_tens = torch.cat(log_probs)
        entropies_tens = torch.stack(entropies).mean()
        
        if values_tens.dim() == 0: values_tens = values_tens.unsqueeze(0)
        advantage = returns - values_tens.detach()

        # 5. Loss Calculation
        actor_loss = -(log_probs_tens * advantage).mean() - (current_entropy * entropies_tens)
        critic_loss = F.smooth_l1_loss(values_tens, returns)

        # 6. Optimization
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_opt.step()
        
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_opt.step()

        # --- Logging ---
        ep_score = sum(raw_rewards) # Approximate
        
        episode_rewards.append(ep_score)
        episode_lengths.append(steps)
        avg_score = np.mean(episode_rewards[-50:])
        
        # --- PRECISION MODE TRIGGER ---
        mode_str = "Explore"
        min_entropy = config['entropy_end']
        
        if avg_score > config['precision_trigger'] and not precision_mode:
            precision_mode = True
            print(f"--> PRECISION MODE ACTIVATED at Ep {episode}")
        
        if precision_mode:
            mode_str = "PRECISION"
            # Drop entropy floor drastically to allow perfect convergence
            min_entropy = 0.001 
            # Decay faster
            current_entropy = max(min_entropy, current_entropy * 0.95)
        else:
            # Standard decay
            current_entropy = max(config['entropy_end'], current_entropy * config['entropy_decay'])
        scheduler.step()

        if episode % 20 == 0:
            print(f"{episode:>8} | {ep_score:>6.0f} | {avg_score:>6.1f} | {current_entropy:>8.4f} | {mode_str:>10}")            
            if avg_score >= config['solved_score']:
                print(f"--> SOLVED {env_name} at Episode {episode}!")
                break

    env.close()
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'time': time.time() - start_time,
        'actor': actor,
        'critic': critic
    }

if __name__ == "__main__":
    # Test run
    # train_agent_batch(CART_POLE)
    # train_agent_batch(ACROBOT)
    train_agent_batch(MOUNTAIN_CART)