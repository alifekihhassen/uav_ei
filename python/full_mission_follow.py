"""
Advanced tasks 3 and 4: follow the ground vehicle, then land on it while it moves. The drone takes off
from the platform at the back of the vehicle and stays 0.5 m above it while the vehicle drives away, on
the identified models (Python side of rl_interface_x/, follow missions).

    0-1 s   on the platform, vehicle at rest
    1 s     take off to 0.5 m above the platform (the platform is z = 0, as on the course's Figure 1)
    2-4 s   the vehicle accelerates at 0.5 m/s^2 along x ...
    4-13 s  ... and drives at 1 m/s; the drone follows it (x_ref = vehicle), y and heading held at 0
    task 4: from T_LAND_VEH = 7 s (constant speed) the altitude reference goes down to the platform, z = 0,
            while the drone keeps following. The simulator has no contact: after touchdown the drone
            keeps "hovering" at z = 0 above the platform, as if it were sitting on it.

The vehicle sends its position, velocity and acceleration (x_env.vehicle_mission). The x axis is flown
by the TRACKING policy (models/sac_x_track.zip, XEnv(tracking=True)): its observation has the position
error, the velocity relative to the vehicle and the vehicle's acceleration (feedforward: theta ~ a / g).
The same policy holds y at 0. Everything else is full_mission_x.fly().

    python full_mission_follow.py        task 3 -> ../rl_interface_x/{policy_x_track.mat, traj_follow.mat},
                                                   figures/follow_vehicle.png
    python full_mission_follow.py land   task 4 -> ../rl_interface_x/traj_land_vehicle.mat,
                                                   figures/land_vehicle.png
"""
import os
import sys
import numpy as np
from stable_baselines3 import SAC

import drone_env as de
import x_env as xe
import full_mission_x as fm

Ts = 0.05
N_FOLLOW = 260                     # 13 s
T_TAKEOFF, Z_HIGH = 1.0, 0.5
T_LAND_VEH = 7.0                   # task 4: start of the descent onto the moving platform
Z_TOUCH = 0.01                     # touchdown: within 1 cm of the platform


def make_mission(t_land=np.inf):
    """References of the follow missions. t_land: start of the descent onto the platform (inf: task 3)."""
    def mission_z(t):
        """0.5 m above the platform from the take-off on, back to 0 from t_land (same Ts/2 shift as
        drone_env.mission)."""
        if t < T_TAKEOFF - Ts / 2:
            return 0.0
        return Z_HIGH if t < t_land - Ts / 2 else 0.0
    return {"z": mission_z, "psi": fm.ZERO,
            "x": lambda t: xe.vehicle_mission(t)[0], "vx": lambda t: xe.vehicle_mission(t)[1],
            "ax": lambda t: xe.vehicle_mission(t)[2],
            "y": fm.ZERO, "vy": fm.ZERO, "ay": fm.ZERO}


MISSION_FOLLOW = make_mission()
MISSION_LAND_VEH = make_mission(T_LAND_VEH)


def summary(o):
    """Error relative to the vehicle while it accelerates (and 1 s after) and at constant speed."""
    m = xe.track_metrics({"t": o["t"], "x_ref": o["x_ref"], "x": o["x_true"]})
    up = o["t"] >= 2.0
    return (f"x behind the vehicle: max {m['acc_max_cm']:.1f} cm while it accelerates, "
            f"{m['cruise_mean_cm']:+.1f} cm mean / {m['cruise_max_cm']:.1f} cm max at constant speed | "
            f"max |y| {100 * np.abs(o['y_true']).max():.1f} cm, altitude RMSE {100 * np.sqrt(np.mean((o['z_ref'] - o['z_true'])[up] ** 2)):.1f} cm, "
            f"max tilt {np.degrees(np.abs(o['theta_true']).max()):.1f} deg")


def touchdown(o, t_land=T_LAND_VEH):
    """Task 4: state at the first sample within Z_TOUCH of the platform after t_land, relative to the
    platform (velocities from the true trajectory), and how far the drone went below the platform."""
    t, z = o["t"], o["z_true"]
    after = t >= t_land - Ts / 2
    k = np.flatnonzero(after & (z <= Z_TOUCH))
    if not len(k):
        return None
    k = int(k[0])
    v = lambda p: (p[k + 1] - p[k - 1]) / (2 * Ts)
    return {"t": float(t[k]), "dx_cm": float(100 * (o["x_true"][k] - o["x_ref"][k])),
            "dy_cm": float(100 * o["y_true"][k]),
            "dvx": float(v(o["x_true"]) - o["vx_ref"][k]), "vy": float(v(o["y_true"])), "vz": float(v(z)),
            "below_cm": float(100 * max(0.0, -z[after].min())),
            "after_max_cm": float(100 * np.hypot(o["x_true"] - o["x_ref"], o["y_true"])[k:].max())}


def touchdown_summary(o):
    d = touchdown(o)
    if d is None:
        return "no touchdown"
    return (f"touchdown at t = {d['t']:.2f} s, {d['dx_cm']:+.1f} cm along x / {d['dy_cm']:+.1f} cm along y "
            f"from the centre of the platform, relative speed {d['dvx']:+.2f} / {d['vy']:+.2f} m/s, "
            f"descent {d['vz']:+.2f} m/s | below the platform {d['below_cm']:.1f} cm, "
            f"then within {d['after_max_cm']:.1f} cm of the centre")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    land = len(sys.argv) > 1 and sys.argv[1] == "land"
    mission = MISSION_LAND_VEH if land else MISSION_FOLLOW
    agents = fm.load_agents()
    agents_old = dict(agents)                                  # the task-2 policy, no feedforward
    agents["x"] = SAC.load(os.path.join(xe.MODELS, "sac_x_track.zip"), device="cpu")
    fly = lambda ag, pl, track, **kw: fm.fly(ag, pl(), k_yaw=fm.K_YAW, mission=mission, n_steps=N_FOLLOW,
                                             tracking=track, **kw)
    runs = {"tracking policy, NN": fly(agents, fm.nn_plants, True),
            "tracking policy, tanh": fly(agents, fm.tanh_plants, True),
            "tracking policy, NN + noise": fly(agents, fm.nn_plants, True, noise=fm.BASE_NOISE, seed=1),
            "task-2 policy, NN": fly(agents_old, fm.nn_plants, False)}
    for k, o in runs.items():
        print(f"{k:28s}: {touchdown_summary(o) if land else summary(o)}")
    if land:
        fm.export(agents, runs["tracking policy, NN"], name="traj_land_vehicle.mat", x_policy="policy_x_track.mat")
    else:
        fm.export(agents, runs["tracking policy, NN"], name="traj_follow.mat", x_policy="policy_x_track.mat")

    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    cols = ("#2a78d6", "#eb6834", "#8a8984", "#b5b4ae")
    o0 = runs["tracking policy, NN"]
    ax[0].plot(o0["t"], o0["x_ref"], "--", color="k", lw=1, label="vehicle (platform)")
    for (k, o), c in zip(runs.items(), cols):
        ls = ":" if k.startswith("task-2") else "-"
        ax[0].plot(o["t"], o["x_true"], ls, color=c, label=k)
        ax[1].plot(o["t"], 100 * (o["x_ref"] - o["x_true"]), ls, color=c)
        ax[2].plot(o["t"], o["z_true"], ls, color=c)
        ax[3].plot(o["t"], np.degrees(o["theta_true"]), ls, color=c)
        ax[4].step(o["t"], o["u_x"], ls, where="post", color=c, lw=1)
    ax[2].step(o0["t"], o0["z_ref"], "--", where="post", color="k", lw=1)
    if land:
        d = touchdown(o0)
        for a in ax:
            a.axvline(d["t"], color="#52514e", ls=":", lw=1)
        ax[2].text(d["t"] + 0.1, 0.3, "touchdown", fontsize=8, color="#52514e")
    for a, lab in zip(ax, ("x [m]", "vehicle - drone [cm]", "z [m] (platform = 0)", "theta [deg]", "u_x")):
        a.set_ylabel(lab); a.grid(True)
    ax[0].legend(frameon=False, fontsize=8); ax[-1].set_xlabel("t [s]")
    title = ("Task 4 in Python: land on the moving vehicle (descent from t = 7 s, vehicle at 1 m/s)" if land else
             "Task 3 in Python: follow the vehicle (0.5 m/s^2 up to 1 m/s from t = 2 s)")
    ax[0].set_title(title, loc="left")
    plt.tight_layout(); os.makedirs(os.path.join(de.HERE, "figures"), exist_ok=True)
    plt.savefig(os.path.join(de.HERE, "figures", "land_vehicle.png" if land else "follow_vehicle.png"), dpi=150)
    plt.show()
