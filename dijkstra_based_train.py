import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from network_env import NetworkEnv

def main():
    print("Initializing Multi-Process Network Environment...")
    
    # 1. Define the number of parallel processes (CPU cores) you want to use
    n_envs = 4

    # 2. Wrap your custom environment in the SubprocVecEnv
    # This automatically spins up 4 independent background processes
    env = make_vec_env(NetworkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # The article uses the Ornstein-Uhlenbeck process to produce exploration noise
    # Note: SB3 automatically scales this noise across all 4 processes behind the scenes!
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.2 * np.ones(n_actions) # Sigma can be tuned, 0.2 is standard
    )

    # The article specifies two fully-connected hidden layers with 400 and 300 units 
    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300], # Actor (Policy) network architecture
            qf=[400, 300]  # Critic (Q-value) network architecture
        )
    )

    print("Building the DDPG Agent...")
    # Setting hyperparameters exactly as defined in the article's Table 3 and text
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=0.00001,      # Both actor and critic learning rates are set to 10^-5
        buffer_size=50000,          # Replay buffer size
        batch_size=100,             # Mini batch size
        gamma=0.99,                 # Discount factor
        learning_starts=100,        # Collect 100 transitions before learning begins
        verbose=1,                  # Print training progress to console
        device="auto",               # Automatically use GPU (CUDA) if available
        train_freq=1,         # Tell the AI to train after every single step 
        gradient_steps=-1     # The magic SB3 code for: "Do exactly as many brain updates as the number of environments running!"
    )

    # Train the model
    # The simulation graphs in the paper show training up to 200,000 iterations
    total_timesteps = 200000 
    print(f"Starting training for {total_timesteps} iterations. This may take a while depending on your CPU/GPU...")
    
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    # Save the fully trained model
    print("Training complete! Saving model to 'ddpg_sdn_routing.zip'...")
    model.save("ddpg_sdn_routing")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()