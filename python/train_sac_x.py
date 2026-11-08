"""Train the SAC x controller on the identified pitch NN (domain randomisation) and keep the best
checkpoint on robust evaluation flights (move mission, perturbed model, sensor noise).
Same setup as section 3 of 7_rl_x.ipynb, for long runs / several seeds from a terminal:

    python train_sac_x.py [total_steps] [seed] [gamma] [du_weight] [step | track]

Writes runs/sac_x_s<seed>/best_model.zip (runs/sac_x_g<gamma>_du<du_weight>_s<seed>/ if they are not the
defaults 0.99 and 0.2) and evaluations.npz, TensorBoard logs in runs/tb.
gamma = 0.99 (horizon ~5 s): with 0.98 (~2.5 s, enough for the yaw) the policy did not learn to brake early
enough for a chain of four integrators (runs/sac_x_g0.98_du0.05_s0, see 7_rl_x.ipynb section 3).
To use a new agent in the notebook, copy runs/sac_x_s<seed>/best_model.zip to models/sac_x.zip.
Every 10 k steps the agent is also saved as runs/<name>/ckpt_<steps>_steps.zip: the evaluation return is a
noisy selector, the checkpoint is chosen afterwards on the mission metrics (7_rl_x.ipynb, section 3).
track: task 3 (follow the vehicle), XEnv(tracking=True), runs/sac_xtrack_s<seed>/ -> models/sac_x_track.zip.
"""
import sys, os, torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from x_env import XEnv, load_x_plant, HERE

torch.set_num_threads(1)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
GAMMA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99
DU_W = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
TRACK = len(sys.argv) > 5 and sys.argv[5] == "track"
prefix = "sac_xtrack" if TRACK else "sac_x"
name = f"{prefix}_s{SEED}" if (GAMMA, DU_W) == (0.99, 0.2) else f"{prefix}_g{GAMMA}_du{DU_W}_s{SEED}"
out = os.path.join(HERE, "runs", name)
os.makedirs(out, exist_ok=True)

plant = load_x_plant()
train_env = Monitor(XEnv(plant=plant, randomize=True, du_weight=DU_W, tracking=TRACK))
eval_env = Monitor(XEnv(plant=plant, randomize=True, fig_prob=1.0, du_weight=DU_W, tracking=TRACK))   # perturbed + noise

agent = SAC("MlpPolicy", train_env, seed=SEED, verbose=0,
            learning_rate=3e-4, buffer_size=300_000, batch_size=256, gamma=GAMMA, tau=0.005,
            learning_starts=5_000, train_freq=1, gradient_steps=1, ent_coef="auto",
            policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log=os.path.join(HERE, "runs", "tb"))
cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=10_000, deterministic=True,
                  best_model_save_path=out, log_path=out, verbose=1)
ckpt = CheckpointCallback(save_freq=10_000, save_path=out, name_prefix="ckpt")
agent.learn(total_timesteps=STEPS, callback=[cb, ckpt], tb_log_name=name)
print("done:", os.path.join(out, "best_model.zip"))
