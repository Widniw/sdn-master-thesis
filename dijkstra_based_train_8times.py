import os
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from network_env import NetworkEnv

def train_single_model(seed_value, n_envs=4, total_timesteps=200000):
    print(f"\n{'='*50}")
    print(f"STARTING TRAINING RUN FOR SEED: {seed_value}")
    print(f"{'='*50}\n")

    # 1. Create a dedicated logging folder for this specific seed
    log_dir = f"./training_logs/seed_{seed_value}/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Spin up the vectorized environments and apply the seed!
    # The monitor_dir ensures we capture the CSV data for your thesis graphs
    env = make_vec_env(
        NetworkEnv, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        seed=seed_value,
        monitor_dir=log_dir
    )

    # 3. Setup the Ornstein-Uhlenbeck noise [cite: 452]
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.2 * np.ones(n_actions) 
    )

    # 4. Neural Network Architecture [cite: 446]
    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300], 
            qf=[400, 300]  
        )
    )

    # 5. Build the DDPG Agent
    # We pass the seed to the DDPG algorithm so the neural weights initialize differently!
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=0.001,        # Using the faster learning rate we discussed
        buffer_size=50000,          
        batch_size=100,             
        gamma=0.99,                 
        learning_starts=100,        
        verbose=1,                  # Set to 0 so your console doesn't get flooded with 8 models of text
        device="cpu",               
        train_freq=1,         
        gradient_steps=-1,
        seed=seed_value             # <--- THE CRITICAL PIECE
    )

    # 6. Train the model [cite: 492]
    print(f"Training for {total_timesteps} steps... (This will take a while, grabbing coffee is advised!)")
    model.learn(total_timesteps=total_timesteps)

    # 7. Save the model with a unique name
    model_filename = f"ddpg_sdn_routing_seed_{seed_value}"
    model.save(model_filename)
    print(f" Training complete! Model saved as '{model_filename}.zip'")

    # 8. Clean up the environment before the next loop
    env.close()

def main():
    # Define the 8 random seeds we want to test
    # You can literally use any integers here (e.g., 10, 20, 30...)
    seeds_to_test = [1001, 1002, 1003]
    
    print(f"Starting batch training for {len(seeds_to_test)} different models.")
    
    for seed in seeds_to_test:
        train_single_model(seed_value=seed)
        
    print("\n ALL 8 MODELS HAVE FINISHED TRAINING! ")

if __name__ == "__main__":
    main()