from stable_baselines3 import PPO
from flow_based_network_env_one_arm_bandit import FlowBasedNetworkEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os 
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
import json


def objective(trial):
    # 1. Define the number of parallel processes (CPU cores) you want to use
    n_envs = 8

    # 2. Wrap your custom environment in the SubprocVecEnv
    # This automatically spins up 4 independent background processes
    env = make_vec_env(FlowBasedNetworkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.05, log = True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    net_arch_size = trial.suggest_categorical("net_arch", ["large", "xlarge", "xxlarge"])
    arch_mapping = {
        "large": [512, 512],
        "xlarge": [1024, 1024],
        "xxlarge": [2048, 1024],
    }
    net_arch = arch_mapping[net_arch_size] 
    policy_kwargs = dict(net_arch = dict(pi = net_arch, vf = net_arch))

    print("Building the PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,    
        n_steps=n_steps,            
        batch_size=batch_size,             
        ent_coef=ent_coef,       
        gamma = 0.99,   
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        clip_range=clip_range,
        verbose=1,                  
        device="cpu"               
    )

    model.learn(total_timesteps=60000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=75)

    return mean_reward

def save_best_params_callback(study, trial):
    # Sprawdzamy, czy obecny trial jest tym najlepszym
    if study.best_trial.number == trial.number:
        # Zapisujemy parametry do pliku JSON
        with open("best_params_3paths_150_flows_one_arm_bandit.json", "w") as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trial_number": trial.number
            }, f, indent=4)
        print(f"\n[ZAPISANO] Nowe najlepsze parametry (Trial {trial.number}) zapisane do best_params.json!\n")

if __name__ == "__main__":
    print("Starting searching...")

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials = 300, show_progress_bar=True, callbacks=[save_best_params_callback])

    print(f"Najlepsze parametry: {study.best_params}")
    print(f"Najlepszy uzyskany wynik: {study.best_value}")