"""Evaluation of the SAC policy against the PID baseline: on the training model (NN) and on a model it
has never seen (tanh Hammerstein).
Every function takes `plant`; when it is None, the NN plant (the training model) is used."""
import numpy as np, matplotlib.pyplot as plt
from drone_env import DroneAltitudeEnv, TanhPlant, PIDBaseline, rollout, load_nn_plant, Z_GROUND

C_SAC, C_PID, C_REF, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#8a8984"
BASE_NOISE = {"z": 0.005, "vz": 0.05, "az": 0.3}
plt.rcParams.update({"axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#b5b4ae", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "lines.linewidth": 1.6, "figure.dpi": 110})


_NN = None


def nominal_plant():
    """The training plant (NN of 2_identification_nn.ipynb), loaded once."""
    global _NN
    if _NN is None:
        _NN = load_nn_plant()
    return _NN


def sac_controller(agent):
    return lambda obs, m: float(agent.predict(obs, deterministic=True)[0][0])


def pid_controller():
    pid = PIDBaseline()
    def c(obs, m):
        if m["t"] == 0:
            pid.reset()
        return pid(m)
    return c


def compare(agent, plant=None, noise=None, seed=0):
    env = DroneAltitudeEnv(plant=plant if plant is not None else nominal_plant(), noise=noise)
    return {"SAC": rollout(env, sac_controller(agent), seed), "PID": rollout(env, pid_controller(), seed)}


def plot_flights(results, titles, fname=None, show=("SAC",)):
    fig, ax = plt.subplots(4, len(results), figsize=(5.2 * len(results), 8.8), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1, 1]}, squeeze=False)
    curves = [(k, c) for k, c in (("PID", C_PID), ("SAC", C_SAC)) if k in show]
    for j, (res, title) in enumerate(zip(results, titles)):
        a = ax[0, j]
        a.plot(res["SAC"]["t"], res["SAC"]["z_ref"], "--", color=C_REF, lw=1.2, label="reference")
        a.axhline(Z_GROUND, color=C_MUTED, ls=":", lw=1, label="ground")
        for k, c in curves:
            o = res[k]
            a.plot(o["t"], o["z"], color=c, label=f"{k}  RMSE {o['rmse']:.3f} m")
        a.set_title(title, fontsize=10, color=C_INK, loc="left"); a.set_ylabel("z [m]")
        a.legend(fontsize=8, frameon=False, loc="upper right")
        b = ax[1, j]
        for k, c in curves:
            b.step(res[k]["t"], res[k]["u"], where="post", color=c, lw=1.1, label=k)
        b.set_ylabel("u"); b.set_ylim(-1.05, 1.05)
        for i, (key, lab) in enumerate((("vz", "vz [m/s]"), ("az", "az [m/s²]")), start=2):
            for k, c in curves:
                ax[i, j].plot(res[k]["t"], res[k][key], color=c, lw=1.1, label=k)
            ax[i, j].set_ylabel(lab)
        ax[-1, j].set_xlabel("t [s]")
    fig.tight_layout()
    if fname: fig.savefig(fname, dpi=150)
    return fig
