"""
Full mission of advanced task 2: take off, move along x, land at a different position. Altitude policy
+ yaw policy (holds the heading) + x policy, used on BOTH horizontal axes, + mixer, flown on the
identified models (Python side of rl_interface_x/).

    0-1 s   on the platform (z_ref = 0, x_ref = 0)
    1-7 s   at 0.5 m; move to x = 1 m from t = 2 s
    7-11 s  land on the ground (z_ref = -0.25 m) at x = 1 m
    y is held at 0 the whole time

At each step (Ts = 0.05 s), in this order:
    1. altitude policy: obs_z   = Observer(z_ref, z, vz, az_prev, u_T_prev)                         -> u_T
    2. yaw policy:      obs_yaw = YawObserver(psi_ref = 0, psi, wz, w_dot_z_prev, u_yaw_prev, u_T)    -> u_yaw
    3. x policy:        obs_x   = XObserver(x_ref, x, vx - vx_ref, wy, w_dot_y_prev, ax_prev, az_prev, u_x_prev, u_T) -> u_x
    4. x policy on y:   obs_y   = XObserver(y_ref, y, vy - vy_ref, -wx, -w_dot_x_prev, ay_prev, az_prev, u_y_prev, u_T) -> u_y
    5. mixer: u_yaw clipped to +-(1 - |u_T|), u_x and u_y each to +-(1 - |u_T| - |u_yaw|) (other motors);
              U = u_T + u_yaw [1 -1 1 -1] + u_x [-1 0 1 0] + u_y [0 -1 0 1]

Why the x policy works on y: the probe of generate_x_data.m found the two axes exactly symmetric (motors
2/4 instead of 1/3, same response: 5.4602 m, 0.2159 rad). With S_Y = [0 -1 0 1], u_y > 0 rolls the drone
to phi < 0 and pushes it to y > 0: the "pitch" of the y axis is -phi, its rate -wx, and the accelerometer
tilt is atan2(ay, az + g). The identified models have no roll: y is held at 0 here by construction, the
y loop is what holds it on the simulator (without it the drone drifts ~8 cm sideways before touchdown).

vx_ref, vy_ref: velocity of the reference (0 for a fixed landing spot; the vehicle's velocity when the
drone follows it, task 3): the policy sees the velocity RELATIVE to the reference, so that in the frame
of the reference the problem is the one it was trained on.

The plants are coupled as identified in 6_identification_x.ipynb: the thrust T = az + g of the altitude
model is tilted (vertical az = T cos(theta) cos(phi) - g, horizontal pushes rotated by the heading psi),
and u_x / u_y add a yaw torque k (u_x^2 - u_y^2) (motors 1/3 and 2/4 spin in opposite directions).

    python full_mission_x.py    -> writes ../rl_interface_x/{policy.mat, policy_yaw.mat, policy_x.mat,
                                   traj_full_x.mat} and figures/full_mission_x.png
"""
import os
import joblib
import numpy as np
from scipy.io import savemat
from stable_baselines3 import SAC

import drone_env as de
import yaw_env as ye
import x_env as xe
from export_policy import _policy_matrices

Ts = 0.05
G = 9.81
N_FULL = 220                       # 11 s
T_TAKEOFF, T_LANDING, Z_HIGH = 1.0, 7.0, 0.5
S_YAW = np.array([1.0, -1.0, 1.0, -1.0])
S_X = np.array([-1.0, 0.0, 1.0, 0.0])
S_Y = np.array([0.0, -1.0, 0.0, 1.0])
OUT_DIR = os.path.join(de.HERE, "..", "rl_interface_x")
MEASUREMENTS = ("t", "z", "vz", "az_prev", "u_T_prev", "psi", "wz", "w_dot_z_prev", "u_yaw_prev",
                "x", "vx", "wy", "w_dot_y_prev", "ax_prev", "u_x_prev",
                "y", "vy", "wx", "w_dot_x_prev", "ay_prev", "u_y_prev")
REFERENCES = ("z_ref", "psi_ref", "x_ref", "vx_ref", "ax_ref", "y_ref", "vy_ref", "ay_ref")
NOISE_KEYS = ("z", "vz", "az", "psi", "wz", "w", "x", "vx", "wy", "wy_dot", "ax", "y", "vy", "wx", "wx_dot", "ay")


def mission_z(t):
    """Altitude reference (same Ts/2 shift as drone_env.mission)."""
    if t < T_TAKEOFF - Ts / 2:
        return 0.0
    if t < T_LANDING - Ts / 2:
        return Z_HIGH
    return de.Z_GROUND


ZERO = lambda t: 0.0
# land at x = 1 m: fixed landing spot, heading and y held at 0
MISSION_LAND = {"z": mission_z, "psi": ZERO, "x": xe.x_mission, "vx": ZERO, "y": ZERO, "vy": ZERO}


def fly(agents, plants, noise=None, seed=0, k_yaw=0.0, mission=None, n_steps=N_FULL, tracking=False):
    """One flight. agents: dict 'z', 'yaw', 'x' (the x policy flies both horizontal axes). plants: dict
    'z', 'yaw', 'x', 'y' ('y' is a second instance of the pitch model, for the roll). noise: sensor noise
    std per measurement (NOISE_KEYS). k_yaw: yaw torque per u_x^2 (notebook 6). mission: dict of reference
    functions of t for z, psi, x, vx, y, vy and optionally ax, ay (default: MISSION_LAND).
    tracking=True: agents['x'] is the tracking policy (models/sac_x_track.zip), whose observation also
    has the reference's acceleration (x_env.XObserver(tracking=True))."""
    rng = np.random.default_rng(seed)
    mis = {"ax": ZERO, "ay": ZERO, **(mission or MISSION_LAND)}
    n = {k: 0.0 for k in NOISE_KEYS}
    n.update(noise or {})
    obs_fn = {"z": de.Observer(), "yaw": ye.YawObserver(), "x": xe.XObserver(tracking), "y": xe.XObserver(tracking)}
    p_z, p_yaw, p_x, p_y = plants["z"], plants["yaw"], plants["x"], plants["y"]
    az_lvl = p_z.reset(0.0)                    # motors off at t = 0: az = -g
    az = az_lvl
    w_z, w_x, w_l = p_yaw.reset(), p_x.reset(), p_y.reset()
    z = vz = psi = wz = x = y = vx = vy = 0.0
    q_x = th_x = 0.0                           # pitch rate wy and pitch theta
    q_l = th_l = 0.0                           # the same for the y axis: -wx and -phi
    az_prev, wz_dot_prev, wx_dot_prev, wl_dot_prev, ax_prev, ay_prev = az, w_z, w_x, w_l, 0.0, 0.0
    uT_prev = uyaw_prev = ux_prev = uy_prev = 0.0
    log = {k: [] for k in (*MEASUREMENTS, *REFERENCES, "u_T", "u_yaw", "u_x", "u_y",
                           "obs_z", "obs_yaw", "obs_x", "obs_y",
                           "z_true", "psi_true", "x_true", "y_true", "theta_true", "phi_true", "U")}
    noisy = lambda v, key: v + n[key] * rng.standard_normal()
    act = lambda key, o: float(agents[key].predict(o.astype(np.float32), deterministic=True)[0][0])
    for k in range(n_steps):
        t = k * Ts
        m = {"t": t, "z": noisy(z, "z"), "vz": noisy(vz, "vz"),
             "az_prev": 0.0 if k == 0 else noisy(az_prev, "az"), "u_T_prev": uT_prev,
             "psi": noisy(psi, "psi"), "wz": noisy(wz, "wz"),
             "w_dot_z_prev": 0.0 if k == 0 else noisy(wz_dot_prev, "w"), "u_yaw_prev": uyaw_prev,
             "x": noisy(x, "x"), "vx": noisy(vx, "vx"), "wy": noisy(q_x, "wy"),
             "w_dot_y_prev": 0.0 if k == 0 else noisy(wx_dot_prev, "wy_dot"),
             "ax_prev": 0.0 if k == 0 else noisy(ax_prev, "ax"), "u_x_prev": ux_prev,
             "y": noisy(y, "y"), "vy": noisy(vy, "vy"), "wx": noisy(-q_l, "wx"),
             "w_dot_x_prev": 0.0 if k == 0 else noisy(-wl_dot_prev, "wx_dot"),
             "ay_prev": 0.0 if k == 0 else noisy(ay_prev, "ay"), "u_y_prev": uy_prev}
        ref = {f"{a}_ref": mis[a](t) for a in ("z", "psi", "x", "vx", "ax", "y", "vy", "ay")}
        # 1. altitude policy
        obs_z = obs_fn["z"]({"z_ref": ref["z_ref"], "z": m["z"], "vz": m["vz"], "az_prev": m["az_prev"],
                             "u_prev": uT_prev})
        u_T = act("z", obs_z)
        # 2. yaw policy, 3-4. x policy on x and on y (all see the thrust chosen at this step)
        obs_yaw = obs_fn["yaw"]({"psi_ref": ref["psi_ref"], "psi": m["psi"], "wz": m["wz"],
                                 "w_dot_z_prev": m["w_dot_z_prev"], "u_yaw_prev": uyaw_prev, "u_T": u_T})
        u_yaw = act("yaw", obs_yaw)
        obs_x = obs_fn["x"]({"t": t, "x_ref": ref["x_ref"], "x": m["x"], "vx": m["vx"] - ref["vx_ref"],
                             "wy": m["wy"], "w_dot_y_prev": m["w_dot_y_prev"], "ax_prev": m["ax_prev"],
                             "az_prev": m["az_prev"], "u_x_prev": ux_prev, "u_T": u_T, "ax_ref": ref["ax_ref"]})
        u_x = act("x", obs_x)
        obs_y = obs_fn["y"]({"t": t, "x_ref": ref["y_ref"], "x": m["y"], "vx": m["vy"] - ref["vy_ref"],
                             "wy": -m["wx"], "w_dot_y_prev": -m["w_dot_x_prev"], "ax_prev": m["ay_prev"],
                             "az_prev": m["az_prev"], "u_x_prev": uy_prev, "u_T": u_T, "ax_ref": ref["ay_ref"]})
        u_y = act("x", obs_y)
        # 5. mixer: thrust, then yaw, then x and y (on different motors: each has the remaining margin)
        margin = 1.0 - abs(u_T)
        u_yaw = min(max(u_yaw, -margin), margin)
        margin -= abs(u_yaw)
        u_x = min(max(u_x, -margin), margin)
        u_y = min(max(u_y, -margin), margin)
        U = u_T + u_yaw * S_YAW + u_x * S_X + u_y * S_Y

        for key in MEASUREMENTS:
            log[key].append(m[key])
        for key, v in (*ref.items(), ("u_T", u_T), ("u_yaw", u_yaw), ("u_x", u_x), ("u_y", u_y),
                       ("obs_z", obs_z.astype(np.float32).astype(np.float64)),
                       ("obs_yaw", obs_yaw.astype(np.float32).astype(np.float64)),
                       ("obs_x", obs_x.astype(np.float32).astype(np.float64)),
                       ("obs_y", obs_y.astype(np.float32).astype(np.float64)),
                       ("z_true", z), ("psi_true", psi), ("x_true", x), ("y_true", y),
                       ("theta_true", th_x), ("phi_true", -th_l), ("U", U)):
            log[key].append(v)

        # plants (identified models), same integration rules as the environments
        w_x1 = p_x.step(u_x, u_T)
        q_x1 = q_x + Ts * (w_x + w_x1) / 2
        th_x1 = th_x + Ts * (q_x + q_x1) / 2
        w_l1 = p_y.step(u_y, u_T)
        q_l1 = q_l + Ts * (w_l + w_l1) / 2
        th_l1 = th_l + Ts * (q_l + q_l1) / 2
        az_lvl1 = p_z.step(u_T)                        # thrust of the altitude model: T = az_lvl + g
        T0, T1 = az_lvl + G, az_lvl1 + G
        az1 = T1 * np.cos(th_x1) * np.cos(th_l1) - G   # tilted thrust lifts less
        w_z1 = p_yaw.step(u_yaw, u_T) + k_yaw * (u_x ** 2 - u_y ** 2)
        p_yaw.w = w_z1
        wz1 = wz + Ts * (w_z + w_z1) / 2
        psi1 = psi + Ts * (wz + wz1) / 2
        # horizontal pushes in the drone's frame, rotated by the heading into the world frame
        hx0, hx1 = T0 * np.sin(th_x), T1 * np.sin(th_x1)
        hy0, hy1 = T0 * np.sin(th_l), T1 * np.sin(th_l1)
        ax0, ax1 = hx0 * np.cos(psi) - hy0 * np.sin(psi), hx1 * np.cos(psi1) - hy1 * np.sin(psi1)
        ay0, ay1 = hx0 * np.sin(psi) + hy0 * np.cos(psi), hx1 * np.sin(psi1) + hy1 * np.cos(psi1)
        z += Ts * vz + Ts ** 2 * (2 * az + az1) / 6
        vz += Ts * (az + az1) / 2
        x += Ts * vx + Ts ** 2 * (2 * ax0 + ax1) / 6
        vx += Ts * (ax0 + ax1) / 2
        y += Ts * vy + Ts ** 2 * (2 * ay0 + ay1) / 6
        vy += Ts * (ay0 + ay1) / 2
        az_prev, az, az_lvl, ax_prev, ay_prev = az, az1, az_lvl1, ax0, ay0
        wz_dot_prev, w_z, wz, psi = w_z, w_z1, wz1, psi1
        wx_dot_prev, w_x, q_x, th_x = w_x, w_x1, q_x1, th_x1
        wl_dot_prev, w_l, q_l, th_l = w_l, w_l1, q_l1, th_l1
        uT_prev, uyaw_prev, ux_prev, uy_prev = u_T, u_yaw, u_x, u_y
    return {k: np.array(v) for k, v in log.items()}


def export(agents, traj, out_dir=OUT_DIR, name="traj_full_x.mat", x_policy="policy_x.mat"):
    os.makedirs(out_dir, exist_ok=True)
    savemat(os.path.join(out_dir, "policy.mat"), _policy_matrices(agents["z"]))
    savemat(os.path.join(out_dir, "policy_yaw.mat"), _policy_matrices(agents["yaw"]))
    savemat(os.path.join(out_dir, x_policy), _policy_matrices(agents["x"]))
    keys = (*MEASUREMENTS, *REFERENCES, "u_T", "u_yaw", "u_x", "u_y")
    savemat(os.path.join(out_dir, name),
            {**{k: traj[k].reshape(-1, 1) for k in keys},
             **{k: traj[k] for k in ("obs_z", "obs_yaw", "obs_x", "obs_y", "U")}})
    print(f"policy.mat, policy_yaw.mat, {x_policy}, {name} written in {os.path.abspath(out_dir)} "
          f"({len(traj['t'])} steps)")


def summary(o):
    e_z = o["z_ref"] - o["z_true"]
    land = o["t"] >= T_LANDING
    mv = xe.move_metrics({"t": o["t"], "x": o["x_true"]})
    return (f"altitude RMSE {np.sqrt(np.mean(e_z ** 2)):.3f} m, below ground "
            f"{max(0.0, de.Z_GROUND - o['z_true'][land].min()) * 100:.1f} cm | "
            f"x settle {mv['settle_s']:.2f} s, overshoot {mv['overshoot_cm']:.1f} cm, "
            f"landed at x = {o['x_true'][-1]:.3f} m, y = {o['y_true'][-1] * 100:.1f} cm, "
            f"max tilt {np.degrees(np.abs(o['theta_true']).max()):.1f} deg | "
            f"max |psi| {np.degrees(np.abs(o['psi_true']).max()):.2f} deg")


def load_agents():
    return {"z": SAC.load(os.path.join(de.MODELS, "sac_nn.zip"), device="cpu"),
            "yaw": SAC.load(os.path.join(ye.MODELS, "sac_yaw.zip"), device="cpu"),
            "x": SAC.load(os.path.join(xe.MODELS, "sac_x.zip"), device="cpu")}


def nn_plants():
    return {"z": de.load_nn_plant(), "yaw": ye.load_yaw_plant(), "x": xe.load_x_plant(), "y": xe.load_x_plant()}


def tanh_plants():
    return {"z": de.TanhPlant(), "yaw": ye.load_yaw_test_plant(), "x": xe.load_x_test_plant(),
            "y": xe.load_x_test_plant()}


K_YAW = float(joblib.load(os.path.join(xe.MODELS, "x_model_nn.joblib"))["k_yaw_ux2"])
BASE_NOISE = {"z": 0.005, "vz": 0.05, "az": 0.3, "psi": 0.005, "wz": 0.02, "w": 0.7,
              "x": 0.005, "vx": 0.005, "wy": 0.005, "wy_dot": 0.005, "ax": 0.005,
              "y": 0.005, "vy": 0.005, "wx": 0.005, "wx_dot": 0.005, "ay": 0.005}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    agents = load_agents()
    nominal = fly(agents, nn_plants(), k_yaw=K_YAW)
    print("NN models, no noise      :", summary(nominal))
    tanh = fly(agents, tanh_plants(), k_yaw=K_YAW)
    print("tanh models (never seen) :", summary(tanh))
    noisy = fly(agents, nn_plants(), k_yaw=K_YAW, seed=1, noise=BASE_NOISE)
    print("NN models + sensor noise :", summary(noisy))
    export(agents, nominal)

    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for o, lab, c in ((nominal, "NN models", "#2a78d6"), (tanh, "tanh models", "#eb6834"), (noisy, "NN + noise", "#8a8984")):
        ax[0].plot(o["t"], o["z_true"], color=c, label=lab)
        ax[1].plot(o["t"], o["x_true"], color=c)
        ax[2].plot(o["t"], np.degrees(o["theta_true"]), color=c)
        ax[3].step(o["t"], o["u_T"], where="post", color=c, lw=1)
        ax[4].step(o["t"], o["u_x"], where="post", color=c, lw=1)
    ax[0].step(nominal["t"], nominal["z_ref"], "--", where="post", color="k", lw=1, label="reference")
    ax[1].step(nominal["t"], nominal["x_ref"], "--", where="post", color="k", lw=1)
    ax[0].axhline(de.Z_GROUND, color="#8a8984", ls=":")
    for a, lab in zip(ax, ("z [m]", "x [m]", "theta [deg]", "u_T", "u_x")):
        a.set_ylabel(lab); a.grid(True)
    ax[0].legend(frameon=False, fontsize=8); ax[-1].set_xlabel("t [s]")
    ax[0].set_title("Full mission in Python: land at x = 1 m (altitude + yaw + x policies, mixer)", loc="left")
    plt.tight_layout(); os.makedirs(os.path.join(de.HERE, "figures"), exist_ok=True)
    plt.savefig(os.path.join(de.HERE, "figures", "full_mission_x.png"), dpi=150)
    plt.show()
