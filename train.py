from stable_baselines3 import PPO
from network_env import NetworkEnv

def main():
    print("Initializing Flow-Based Network Environment...")
    env = NetworkEnv()

    print("Building the PPO Agent...")
    # PPO does not require action noise, making setup much simpler
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003,      
        batch_size=64,             
        verbose=1,                  
        device="auto" # Will use CPU since you are on Linux without CUDA               
    )

    total_timesteps = 200000 
    print(f"Starting training for {total_timesteps} iterations...")
    
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    print("Training complete! Saving model...")
    model.save("ppo_sdn_flow_routing")

if __name__ == "__main__":
    main()