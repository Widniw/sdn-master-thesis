from stable_baselines3 import PPO
from flow_based_network_env import NetworkEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def main():
    # 1. Define the number of parallel processes (CPU cores) you want to use
    n_envs = 4

    # 2. Wrap your custom environment in the SubprocVecEnv
    # This automatically spins up 4 independent background processes
    env = make_vec_env(NetworkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

# 1. Give PPO the exact same size "brain" as DDPG (The paper used [400, 300])
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
        learning_rate=0.0003,    # 0.0003 is the industry standard for PPO
        n_steps=1024,            # How many steps to collect per CPU core before learning
        batch_size=64,             
        ent_coef=0.01,           # ENTROPY BONUS: Forces the AI to explore different flow paths
        policy_kwargs=policy_kwargs, # Give it the bigger brain!
        verbose=1,                  
        device="cpu"               
    )

    total_timesteps = 200000 
    print(f"Starting training for {total_timesteps} iterations...")
    
    model.learn(total_timesteps=total_timesteps, log_interval=1)

    print("Training complete! Saving model...")
    model.save("ppo_sdn_flow_routing")

if __name__ == "__main__":
    main()