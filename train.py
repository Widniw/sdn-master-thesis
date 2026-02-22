import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from network_env import NetworkEnv

def main():
    print("Initializing Network Environment...")
    env = NetworkEnv()

    # The article uses the Ornstein-Uhlenbeck process to produce exploration noise
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.2 * np.ones(n_actions) # Sigma can be tuned, 0.2 is standard
    )

    # The article specifies two fully-connected hidden layers with 400 and 300 units 
    # for both the actor network and the critic network
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
        device="auto"               # Automatically use GPU (CUDA) if available
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