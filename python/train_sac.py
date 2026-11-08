"""Train the SAC altitude controller on the identified model (domain randomisation) and keep the best
checkpoint on robust evaluation flights (Figure 1 mission, perturbed model, sensor noise).
Same setup as section 3 of 3_rl_controller.ipynb, for long runs / several seeds from a terminal:

    python train_sac.py [total_steps] [seed] [plant: nn | tanh]

Writes runs/<name>/best_model.zip and evaluations.npz (learning curve), TensorBoard logs in runs/tb.
To use a new agent in the notebook, copy runs/<name>/best_model.zip to models/sac_nn.zip.
"""
import sys, os, torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from drone_env import DroneAltitudeEnv, TanhPlant, load_nn_plant, HERE

torch.set_num_threads(1)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
PLANT = sys.argv[3] if len(sys.argv) > 3 else "nn"
plant = load_nn_plant() if PLANT == "nn" else TanhPlant()
name = f"sac_{PLANT}_s{SEED}" if PLANT == "nn" else f"sac_s{SEED}"
out = os.path.join(HERE, "runs", name)
os.makedirs(out, exist_ok=True)

train_env = Monitor(DroneAltitudeEnv(plant=plant, randomize=True))
# evaluation: Figure 1 mission, but perturbed model + sensor noise -> picks a ROBUST checkpoint
eval_env = Monitor(DroneAltitudeEnv(plant=plant, randomize=True, fig1_prob=1.0))

agent = SAC("MlpPolicy", train_env, seed=SEED, verbose=0,
            learning_rate=3e-4, buffer_size=300_000, batch_size=256, gamma=0.98, tau=0.005,
            learning_starts=5_000, train_freq=1, gradient_steps=1, ent_coef="auto",
            policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log=os.path.join(HERE, "runs", "tb"))

cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=10_000, deterministic=True,
                  best_model_save_path=out, log_path=out, verbose=1)
agent.learn(total_timesteps=STEPS, callback=cb, tb_log_name=name)
print("done:", os.path.join(out, "best_model.zip"))
