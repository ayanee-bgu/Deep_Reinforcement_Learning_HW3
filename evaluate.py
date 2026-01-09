import matplotlib.pyplot as plt

# --- IMPORTS ---
# 1. Import training logic from your main script (assumed filename 'train.py')
from individual_networks import train_actor_critic

# 2. Import visualization and metrics tools from our new utils file
from utils import plot_comparison_generic

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    
    # Select which environments you want to evaluate
    # You can comment out specific envs if you want to test quicker
    environments = [
        "CartPole-v1", 
        "Acrobot-v1", 
        "MountainCarContinuous-v0"
    ]
    
    all_results = {}

    print("Starting Transfer Learning Evaluation Pipeline...")
    print("=" * 60)
    
    for env_name in environments:
        print(f"Training on {env_name}...")
        
        # Run training
        # Expects return dict: {'rewards', 'lengths', 'training_time', 'policy_net', 'value_net'}
        train_data = train_actor_critic(env_name)
        
        # Store for plotting
        all_results[env_name] = train_data
        print(f"-> {env_name} finished in {train_data['training_time']:.2f}s")
        print("-" * 60)

    print("\nGenerating Comparison Graphs and Statistics...")
    
    # Generate the 4-panel plot with table
    plot_comparison_generic(all_results)