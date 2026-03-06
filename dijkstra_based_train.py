import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize # Added VecNormalize

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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # The article uses the Ornstein-Uhlenbeck process to produce exploration noise
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.2 * np.ones(n_actions) 
    )

    # The article specifies two fully-connected hidden layers with 400 and 300 units 
    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300], 
            qf=[400, 300]  
        )
    )

    print("Building the DDPG Agent...")
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=0.001,      # Both actor and critic learning rates are set to 10^-5
        buffer_size=50000,          # Replay buffer size
        batch_size=100,             # Mini batch size
        gamma=0.99,                 # Discount factor
        learning_starts=100,        # Collect 100 transitions before learning begins
        verbose=1,                  # Print training progress to console
        device="cpu",               # Automatically use GPU (CUDA) if available
        train_freq=1,         # Tell the AI to train after every single step 
        gradient_steps=-1     # The magic SB3 code for: "Do exactly as many brain updates as the number of environments running!"
    )

    total_timesteps = 200000 
    print(f"Starting training for {total_timesteps} iterations. This may take a while depending on your CPU/GPU...")
    
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    # Save the fully trained model
    print("Training complete! Saving model to 'ddpg_sdn_routing.zip'...")
    model.save("ddpg_sdn_routing")

    # CRITICAL: Save the VecNormalize statistics
    # If you do not save this, the model will be blind during evaluation!
    print("Saving VecNormalize statistics to 'vec_normalize.pkl'...")
    env.save("vec_normalize.pkl")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()