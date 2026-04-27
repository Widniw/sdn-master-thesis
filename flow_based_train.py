from stable_baselines3 import PPO
from flow_based_network_env import FlowBasedNetworkEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os 


def main():
    # 1. Define the number of parallel processes (CPU cores) you want to use
    n_envs = 16

    # 2. Wrap your custom environment in the SubprocVecEnv
    # This automatically spins up 4 independent background processes
    env = make_vec_env(FlowBasedNetworkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

# 1. Give PPO the exact same size "brain" as DDPG (The paper used [400, 300])
    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300], # The Actor MUST be big enough for 1250 inputs
            vf=[400, 300]  # The Critic
        )
    )

    print("Building the PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003,    
        n_steps=512,            
        batch_size=256,             
        ent_coef=0.0001,       
        gamma = 0.999,   
        policy_kwargs=policy_kwargs,
        verbose=1,                  
        device="cpu"               
    )

    total_timesteps = 1500000 
    print(f"Starting training for {total_timesteps} iterations...")

    checkpoint_dir = './models/my_approach_flow_based/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=12500, # Saves every 50k total steps across 4 envs
        save_path=checkpoint_dir,
        name_prefix='ppo_correct_discrete_3_paths'
    )
    
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=checkpoint_callback)

    print("Training complete! Saving model...")
    model.save("ppo_sdn_flow_routing")

if __name__ == "__main__":
    main()