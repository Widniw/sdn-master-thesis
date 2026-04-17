import time
from flow_based_network_env import FlowBasedNetworkEnv

def test_naive_baseline():
    print("Initializing Environment...")
    # Initialize your environment
    env = FlowBasedNetworkEnv()
    
    print("\nResetting Environment (Generating 150 random flows)...")
    # We use a fixed seed (e.g., 42) so the random traffic is exactly 
    # the same every time you run this script to test changes.
    obs, _ = env.reset() 
    
    done = False
    step_count = 0
    final_reward = 0.0
    
    start_time = time.time()
    
    print("Routing all 150 flows down Action 0 (Shortest Path)...")
    
    while not done:
        # Always pick Action 0
        action = 0 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # In your sparse reward setup, reward is 0.0 until the very last step.
        # We capture the non-zero reward when it finally drops.
        if reward != 0.0:
            final_reward = reward

    elapsed_time = time.time() - start_time

    # ==========================================
    # PRINT DIAGNOSTIC RESULTS
    # ==========================================
    print("\n" + "="*50)
    print("ENVIRONMENT DIAGNOSTIC RESULTS (Naive ECMP)")
    print("="*50)
    print(f"Total Steps Executed: {step_count}")
    print(f"Physics Engine Time:  {elapsed_time:.4f} seconds")
    print("-" * 50)
    
    if 'avg_delay' in info and 'packet_loss' in info:
        print(f"Average Delay:        {info['avg_delay']:.4f}")
        print(f"Total Packet Loss:    {info['packet_loss']:.2f} Mbps")
        print(f"Total Network Load:   {env.total_incoming_network:.2f} Mbps")
    else:
        print("WARNING: 'avg_delay' and 'packet_loss' not found in info dict!")
        
    print("-" * 50)
    print(f"FINAL EPISODE REWARD: {final_reward:.4f}")
    print("="*50)

if __name__ == "__main__":
    test_naive_baseline()