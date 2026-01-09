import matplotlib.pyplot as plt

# --- IMPORTS ---
# 1. Import training logic from your main script (assumed filename 'train.py')
from individual_networks import train_agent_batch, CART_POLE, ACROBOT, MOUNTAIN_CART, ENV_CONFIGS

# 2. Import visualization and metrics tools from our new utils file
from utils import plot_comparison_generic, print_statistics, plot_training_results

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    
    # Select which environments you want to evaluate
    # You can comment out specific envs if you want to test quicker
    environments = [
        CART_POLE, 
        ACROBOT,
        MOUNTAIN_CART
    ]
    
    all_results = {}

    print("Starting Transfer Learning Evaluation Pipeline...")
    print("=" * 60)
    
    for env_name in environments:
        print(f"Training on {env_name}...")
        
        # Run training
        # Expects return dict: {'rewards', 'lengths', 'training_time', 'policy_net', 'value_net'}
        train_data = train_agent_batch(env_name)
        
        # Store for plotting
        all_results[env_name] = train_data
        print(f"-> {env_name} finished in {train_data['time']:.2f}s")
        print("-" * 60)

    print("\nGenerating Comparison Graphs and Statistics...")
    
    # Generate the 4-panel plot with table
    print_statistics(all_results, ENV_CONFIGS)

    # --- GENERATE PLOTS ---
    print("\n"+"="*40)
    print("Generating Convergence Plots...")
    print("="*40)
    
    # Iterate through whatever results you gathered
    for env_name, data in all_results.items():
        rewards = data['rewards']
        # Retrieve the specific config used for this environment
        config = ENV_CONFIGS[env_name]
        
        # Call the plotting function
        # window=50 matches the window used in the training loop print statements
        plot_training_results(env_name, rewards, config, window=50)
        
    print("\nAll plots generated in the 'training_plots' folder.")