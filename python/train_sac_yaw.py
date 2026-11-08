"""Train the SAC yaw controller on the identified yaw NN (domain randomisation) and keep the best
checkpoint on robust evaluation flights (rotation mission, perturbed model, sensor noise).
Same setup as section 3 of 5_rl_yaw.ipynb, for long runs / several seeds from a terminal:

    python train_sac_yaw.py [total_steps] [seed]

Writes runs/sac_yaw_s<seed>/best_model.zip and evaluations.npz, TensorBoard logs in runs/tb.
To use a new agent in the notebook, copy runs/sac_yaw_s<seed>/best_model.zip to models/sac_yaw.zip.
"""
import sys, os, torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from yaw_env import YawEnv, load_yaw_plant, HERE

torch.set_num_threads(1)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
name = f"sac_yaw_s{SEED}"
out = os.path.join(HERE, "runs", name)
os.makedirs(out, exist_ok=True)

plant = load_yaw_plant()
train_env = Monitor(YawEnv(plant=plant, randomize=True))
eval_env = Monitor(YawEnv(plant=plant, randomize=True, fig_prob=1.0))   # rotation mission, perturbed + noise

agent = SAC("MlpPolicy", train_env, seed=SEED, verbose=0,
            learning_rate=3e-4, buffer_size=300_000, batch_size=256, gamma=0.98, tau=0.005,
            learning_starts=5_000, train_freq=1, gradient_steps=1, ent_coef="auto",
            policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log=os.path.join(HERE, "runs", "tb"))
cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=10_000, deterministic=True,
                  best_model_save_path=out, log_path=out, verbose=1)
agent.learn(total_timesteps=STEPS, callback=cb, tb_log_name=name)
print("done:", os.path.join(out, "best_model.zip"))
