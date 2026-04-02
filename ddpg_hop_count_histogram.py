import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from flow_based_network_env import FlowBasedNetworkEnv  



def main():
    ddpg_env = FlowBasedNetworkEnv()
    
    # Load your best trained DDPG model
    best_model_path = "./models/my_approach_flow_based/ppo_discrete_1900000_steps" 
    article_model = PPO.load(best_model_path, env = ddpg_env)

    seed = 26

    obs, _ = ddpg_env.reset(seed=seed)

    ddpg_action, _ = article_model.predict(obs, deterministic=True)
    
    # Capture the REWARD
    _, ddpg_reward, _, _, info = ddpg_env.step(ddpg_action)

    print(f"{ddpg_reward = }")
    flows_paths_lengths = []
    for flow, path in info['flows_paths'].items():
        length = len(path)
        flows_paths_lengths.append(length)
    
    print(f'{ddpg_action = }')
    print(info['flows_paths'])
    print(flows_paths_lengths)


if __name__ == "__main__":
    main()