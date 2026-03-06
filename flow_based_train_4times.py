import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from flow_based_network_env import NetworkEnv

def train_single_ppo_model(seed_value, n_envs=4, total_timesteps=200000):
    print(f"\n{'='*50}")
    print(f"STARTING PPO TRAINING RUN FOR SEED: {seed_value}")
    print(f"{'='*50}\n")

    # 1. Create a dedicated logging folder for this specific seed
    log_dir = f"./ppo_training_logs/seed_{seed_value}/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Spin up the vectorized environments and apply the seed
    # monitor_dir ensures we capture the CSV data for your learning curves
    env = make_vec_env(
        NetworkEnv, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        seed=seed_value,
        monitor_dir=log_dir
    )

    # 3. Give PPO the exact same size "brain" as DDPG for a fair comparison
    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300], # Actor network 
            vf=[400, 300]  # Value network (PPO's version of the Critic)
        )
    )

    print("Building the PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0001,    # Lowered from 0.0003 to stop the violent KL divergence
        n_steps=1024,            
        batch_size=256,          # Increased from 64 to 256 to give stable, less noisy gradients
        ent_coef=0.0,            # TURNED OFF: With 150x20 choices, natural exploration is enough!
        policy_kwargs=policy_kwargs, 
        verbose=1,                  
        device="cpu",               
        seed=seed_value          
    )

    # 5. Train the model
    print(f"Starting training for {total_timesteps} iterations...")
    model.learn(total_timesteps=total_timesteps, log_interval=1)

    # 6. Save the model with a unique name
    model_filename = f"ppo_sdn_flow_routing_seed_{seed_value}"
    model.save(model_filename)
    print(f"Training complete! Model saved as '{model_filename}.zip'")

    # 7. Clean up the environment before the next loop
    env.close()

def main():
    # Define the 4 random seeds we want to test for PPO
    seeds_to_test = [2001, 2002, 2003, 2004]
    
    print(f"Starting batch training for {len(seeds_to_test)} different PPO models.")
    
    for seed in seeds_to_test:
        train_single_ppo_model(seed_value=seed)
        
    print("\n🎉 ALL 4 PPO MODELS HAVE FINISHED TRAINING! 🎉")

if __name__ == "__main__":
    main()