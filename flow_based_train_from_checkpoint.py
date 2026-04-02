import os
from stable_baselines3 import PPO
from flow_based_network_env import FlowBasedNetworkEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    # 1. Spin up the parallel environments
    n_envs = 4
    env = make_vec_env(FlowBasedNetworkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # 2. Point to your 200k checkpoint
    checkpoint_path = './models/my_approach_flow_based/ppo_discrete_7_paths_2200000_steps.zip'

    custom_objects = {
        "learning_rate": 0.00001 
    }

    print(f"Loading model from {checkpoint_path}...")
    # PPO.load automatically restores your brain size, learning rate, and entropy!
    model = PPO.load(checkpoint_path, env=env, device="cpu", custom_objects=custom_objects)

    # 3. We want to reach 500k total. We are currently at 200k.
    # Therefore, we need to train for 300,000 MORE steps.
    additional_timesteps = 1500000 
    print(f"Resuming training for {additional_timesteps} MORE iterations...")

    checkpoint_dir = './models/my_approach_flow_based/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=12500, # Still saves every 50k steps across 4 envs
        save_path=checkpoint_dir,
        name_prefix='ppo_discrete_7_paths'
    )
    
    # 4. CRITICAL: reset_num_timesteps=False
    # This tells SB3 to continue its internal step counter from 200,001 instead of resetting to 0.
    model.learn(
        total_timesteps=additional_timesteps, 
        log_interval=1, 
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    print("Training complete! Saving final model...")
    model.save("ppo_sdn_flow_routing_500k")

if __name__ == "__main__":
    main()