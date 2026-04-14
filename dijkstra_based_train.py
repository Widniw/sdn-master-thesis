import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from network_env import NetworkEnv

def main():
    print("Initializing Multi-Process Network Environment...")
    
    # 1. Define the number of parallel processes (CPU cores) you want to use
    n_envs = 4

    # 2. Wrap your custom environment in the SubprocVecEnv
    env = make_vec_env(NetworkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # 3. Apply VecNormalize
    # This automatically scales observations and rewards to have a mean of 0 and std of 1.
    # It also clips extreme outliers to prevent gradient explosions.
    # env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10.0)

    # The article uses the Ornstein-Uhlenbeck process to produce exploration noise
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions) 
    )

    # The article specifies two fully-connected hidden layers with 400 and 300 units 
    policy_kwargs = dict(
            net_arch=dict(
                pi=[400, 300], # The Actor (chooses the 625 paths)
                qf=[400, 300]  # The Critic (judges the choices)
            )
        )

    print("Building the DDPG Agent...")
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,       # Dropped to a safe, stable value (10x faster than paper, but safe)
        buffer_size=50000,   
        tau=0.00001,       
        batch_size=100,             
        gamma=0.0,                 
        learning_starts=1000,       # Let it wander randomly for 1000 steps before doing any math
        verbose=1,                  
        device="cuda",               
        train_freq=1,         
        gradient_steps=-1     
    )

    total_timesteps = 1500000 
    print(f"Starting training for {total_timesteps} iterations. This may take a while depending on your CPU/GPU...")

    checkpoint_dir = './models/article_dijkstra/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=12500, # Saves every 50k total steps across 4 envs
        save_path=checkpoint_dir,
        name_prefix='ddpg_sdn_routing'
    )
    
    model.learn(total_timesteps=total_timesteps, log_interval=1000, callback=checkpoint_callback)

    # Save the fully trained model
    print("Training complete! Saving model to 'ddpg_sdn_routing.zip'...")
    model.save("ddpg_sdn_routing_article")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()