"""
Full mission: altitude policy + yaw policy + mixer, flown on the identified models (Python side of
rl_interface_yaw/). The SAME computations are in rl_interface_yaw/rl_controller_full.m.

    0-1 s   on the platform (z_ref = 0)
    1-10 s  at 0.5 m; four quarter turns at t = 2, 4, 6, 8 s (90, 180, 270, 360 deg)
    10-13 s land on the ground (z_ref = -0.25 m)

At each step (Ts = 0.05 s), in this order:
    1. altitude policy: obs_z   = Observer(z_ref, z, vz, az_prev, u_T_prev)            -> u_T
    2. yaw policy:      obs_yaw = YawObserver(psi_ref, psi, wz, w_dot_z_prev, u_yaw_prev, u_T) -> u_yaw
    3. mixer: u_yaw clipped to +-(1 - |u_T|);  u1 = u3 = u_T + u_yaw,  u2 = u4 = u_T - u_yaw

    python full_mission.py      -> writes ../rl_interface_yaw/{policy.mat, policy_yaw.mat, traj_full.mat}
                                   and figures/full_mission.png
"""
import os
import numpy as np
from scipy.io import savemat
from stable_baselines3 import SAC

import drone_env as de
import yaw_env as ye
from export_policy import _policy_matrices

Ts = 0.05
N_FULL = 260                       # 13 s
T_TAKEOFF, T_LANDING, Z_HIGH = 1.0, 10.0, 0.5
S = np.array([1.0, -1.0, 1.0, -1.0])
OUT_DIR = os.path.join(de.HERE, "..", "rl_interface_yaw")
MEASUREMENTS = ("t", "z", "vz", "az_prev", "u_T_prev", "psi", "wz", "w_dot_z_prev", "u_yaw_prev")


def mission_z(t):
    """Altitude reference of the full mission (same Ts/2 shift as drone_env.mission)."""
    if t < T_TAKEOFF - Ts / 2:
        return 0.0
    if t < T_LANDING - Ts / 2:
        return Z_HIGH
    return de.Z_GROUND


mission_psi = ye.yaw_mission       # 90 deg steps at 2, 4, 6, 8 s (unwrapped: the last one is 2*pi)


def fly(alt_agent, yaw_agent, alt_plant, yaw_plant, noise=None, seed=0):
    """One flight of the full mission. noise: dict(z=, vz=, az=, psi=, wz=, w=) sensor noise std."""
    rng = np.random.default_rng(seed)
    n = {"z": 0, "vz": 0, "az": 0, "psi": 0, "wz": 0, "w": 0, **(noise or {})}
    obs_z_fn, obs_y_fn = de.Observer(), ye.YawObserver()
    az = alt_plant.reset(0.0)                  # motors off at t = 0: az = -g
    w = yaw_plant.reset()
    z = vz = psi = wz = 0.0
    az_prev, w_prev, uT_prev, uy_prev = az, w, 0.0, 0.0
    log = {k: [] for k in (*MEASUREMENTS, "z_ref", "psi_ref", "u_T", "u_yaw", "obs_z", "obs_yaw",
                           "z_true", "psi_true", "U")}
    for k in range(N_FULL):
        t = k * Ts
        m = {"t": t,
             "z": z + n["z"] * rng.standard_normal(), "vz": vz + n["vz"] * rng.standard_normal(),
             "az_prev": 0.0 if k == 0 else az_prev + n["az"] * rng.standard_normal(),
             "u_T_prev": uT_prev,
             "psi": psi + n["psi"] * rng.standard_normal(), "wz": wz + n["wz"] * rng.standard_normal(),
             "w_dot_z_prev": 0.0 if k == 0 else w_prev + n["w"] * rng.standard_normal(),
             "u_yaw_prev": uy_prev}
        z_ref, psi_ref = mission_z(t), mission_psi(t)
        # 1. altitude policy
        obs_z = obs_z_fn({"z_ref": z_ref, "z": m["z"], "vz": m["vz"], "az_prev": m["az_prev"], "u_prev": uT_prev})
        u_T = float(alt_agent.predict(obs_z.astype(np.float32), deterministic=True)[0][0])
        # 2. yaw policy (sees the thrust chosen at this step)
        obs_y = obs_y_fn({"psi_ref": psi_ref, "psi": m["psi"], "wz": m["wz"], "w_dot_z_prev": m["w_dot_z_prev"],
                          "u_yaw_prev": uy_prev, "u_T": u_T})
        u_yaw = float(yaw_agent.predict(obs_y.astype(np.float32), deterministic=True)[0][0])
        # 3. mixer
        margin = 1.0 - abs(u_T)
        u_yaw = min(max(u_yaw, -margin), margin)
        U = u_T + u_yaw * S

        for key in MEASUREMENTS:
            log[key].append(m[key])
        for key, v in (("z_ref", z_ref), ("psi_ref", psi_ref), ("u_T", u_T), ("u_yaw", u_yaw),
                       ("obs_z", obs_z.astype(np.float32).astype(np.float64)),
                       ("obs_yaw", obs_y.astype(np.float32).astype(np.float64)),
                       ("z_true", z), ("psi_true", psi), ("U", U)):
            log[key].append(v)

        # plants (identified models), same integration rules as the environments
        az1 = alt_plant.step(u_T)
        z += Ts * vz + Ts ** 2 * (2 * az + az1) / 6
        vz += Ts * (az + az1) / 2
        az_prev, az = az, az1
        w1 = yaw_plant.step(u_yaw, u_T)
        wz1 = wz + Ts * (w + w1) / 2
        psi += Ts * (wz + wz1) / 2
        wz, w_prev, w = wz1, w, w1
        uT_prev, uy_prev = u_T, u_yaw
    return {k: np.array(v) for k, v in log.items()}


def export(alt_agent, yaw_agent, traj, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    savemat(os.path.join(out_dir, "policy.mat"), _policy_matrices(alt_agent))
    savemat(os.path.join(out_dir, "policy_yaw.mat"), _policy_matrices(yaw_agent))
    keys = (*MEASUREMENTS, "z_ref", "psi_ref", "u_T", "u_yaw")
    savemat(os.path.join(out_dir, "traj_full.mat"),
            {**{k: traj[k].reshape(-1, 1) for k in keys}, "obs_z": traj["obs_z"], "obs_yaw": traj["obs_yaw"],
             "U": traj["U"]})
    print(f"policy.mat, policy_yaw.mat, traj_full.mat written in {os.path.abspath(out_dir)} ({len(traj['t'])} steps)")


def summary(o):
    e_z = o["z_ref"] - o["z_true"]
    land = o["t"] >= T_LANDING
    settle = ye.step_metrics({"t": o["t"], "psi": o["psi_true"]})
    return (f"altitude RMSE {np.sqrt(np.mean(e_z ** 2)):.3f} m, min z {o['z_true'].min():+.3f} m, "
            f"below ground {max(0.0, de.Z_GROUND - o['z_true'][land].min()) * 100:.1f} cm | "
            f"yaw settle {settle['settle_s']} s, overshoot {settle['overshoot_deg'].round(1)} deg, "
            f"final heading {np.degrees(o['psi_true'][-1]):.1f} deg")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    alt_agent = SAC.load(os.path.join(de.MODELS, "sac_nn.zip"), device="cpu")
    yaw_agent = SAC.load(os.path.join(ye.MODELS, "sac_yaw.zip"), device="cpu")
    alt_nn, yaw_nn = de.load_nn_plant(), ye.load_yaw_plant()

    nominal = fly(alt_agent, yaw_agent, alt_nn, yaw_nn)
    print("NN models, no noise      :", summary(nominal))
    tanh = fly(alt_agent, yaw_agent, de.TanhPlant(), ye.load_yaw_test_plant())
    print("tanh models (never seen) :", summary(tanh))
    noisy = fly(alt_agent, yaw_agent, alt_nn, yaw_nn,
                noise={"z": 0.005, "vz": 0.05, "az": 0.3, "psi": 0.005, "wz": 0.02, "w": 0.7}, seed=1)
    print("NN models + sensor noise :", summary(noisy))
    export(alt_agent, yaw_agent, nominal)

    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for o, lab, c in ((nominal, "NN models", "#2a78d6"), (tanh, "tanh models", "#eb6834"), (noisy, "NN + noise", "#8a8984")):
        ax[0].plot(o["t"], o["z_true"], color=c, label=lab)
        ax[1].plot(o["t"], np.degrees(o["psi_true"]), color=c)
        ax[2].step(o["t"], o["u_T"], where="post", color=c, lw=1)
        ax[3].step(o["t"], o["u_yaw"], where="post", color=c, lw=1)
    ax[0].step(nominal["t"], nominal["z_ref"], "--", where="post", color="k", lw=1, label="reference")
    ax[1].step(nominal["t"], np.degrees(nominal["psi_ref"]), "--", where="post", color="k", lw=1)
    ax[0].axhline(de.Z_GROUND, color="#8a8984", ls=":")
    for a, lab in zip(ax, ("z [m]", "psi [deg]", "u_T", "u_yaw")):
        a.set_ylabel(lab); a.grid(True)
    ax[0].legend(frameon=False, fontsize=8); ax[-1].set_xlabel("t [s]")
    ax[0].set_title("Full mission in Python: altitude + yaw policies, mixer", loc="left")
    plt.tight_layout(); os.makedirs(os.path.join(de.HERE, "figures"), exist_ok=True)
    plt.savefig(os.path.join(de.HERE, "figures", "full_mission.png"), dpi=150)
