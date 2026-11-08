"""
Track B - Gymnasium environment around the IDENTIFIED drone model (never the protected .p).

Plant  : what we identified from data in steps 1-3
         - NNPlant   : MLP  az[k+1] = F(az[k], u[k])            (2_identification_nn.ipynb)  -> training plant
         - TanhPlant : Hammerstein model  az[k+1] = a az[k] + b tanh(beta u[k]) + c
                       (1_identification_linear.ipynb)  -> PID baseline and "never seen" test drone
Kinematics : vz and z are integrals of az (known physics, not learned).

The measurement timing reproduces EINT_RL_closed_loop.slx exactly:
    t, z, vz : sampled at t_k (zero-order hold)
    az_prev  : az(t_{k-1})  (Unit Delay, initial output 0 at k = 0)
    u_prev   : u_{k-1}      (Unit Delay, initial output 0)
    u        : the same command on the 4 motors (collective thrust), held during Ts

mission() and Observer.__call__() are re-written line by line in rl_controller.m.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

Ts = 0.05
N_STEPS = 200                     # 10 s mission
Z_GROUND = -0.25                  # ground level (Figure 1)

# ---------------------------------------------------------------------------------------------
# Mission of Figure 1  (the SAME function is in rl_controller.m)
# ---------------------------------------------------------------------------------------------
T_TAKEOFF = 1.0                   # [s] stay on the platform (z = 0) until then
T_LANDING = 5.0                   # [s] go down to the ground (z = -0.25) from then on
Z_HIGH = 0.5


def mission(t):
    """Reference altitude of Figure 1. Thresholds are shifted by Ts/2 so that the sample at
    exactly t = 1 s / 5 s is classified the same way in Python and in Simulink (no float tie)."""
    if t < T_TAKEOFF - Ts / 2:
        return 0.0
    if t < T_LANDING - Ts / 2:
        return Z_HIGH
    return Z_GROUND


def random_mission(rng):
    """Training missions: random staircases inside the flight envelope [-0.25, 0.8] m, so the
    policy learns to track ANY step, not to memorise the 10 s of Figure 1."""
    t_sw = [0.0]
    levels = [0.0]                                     # always starts on the platform
    t = rng.uniform(0.0, 1.5)
    while t < N_STEPS * Ts:
        t_sw.append(t)
        levels.append(rng.uniform(Z_GROUND, 0.8) if rng.random() > 0.25 else Z_GROUND)
        t += rng.uniform(1.2, 4.0)
    t_sw, levels = np.array(t_sw), np.array(levels)

    def f(t):
        return float(levels[np.searchsorted(t_sw, t + 1e-9, side="right") - 1])
    return f


# ---------------------------------------------------------------------------------------------
# Observation (the SAME computation is in rl_controller.m, observation())
# ---------------------------------------------------------------------------------------------
I_MAX = 0.5                       # anti-windup bound of the error integral [m.s]
OBS_NAMES = ("e", "z", "vz/2", "az_prev/10", "u_prev", "I/I_MAX", "z_ref")


class Observer:
    """obs = [e, z, vz/2, az_prev/10, u_prev, I/I_MAX, z_ref], e = z_ref - z, clipped.
    I is the integral of e (memory = a persistent variable in MATLAB). It lets the policy remove
    the static error caused by a hover thrust slightly different from our model's."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.I = 0.0

    def __call__(self, m):
        e = m["z_ref"] - m["z"]
        self.I = min(max(self.I + Ts * e, -I_MAX), I_MAX)
        obs = np.array([min(max(e, -1.0), 1.0),
                        m["z"],
                        min(max(m["vz"] / 2.0, -5.0), 5.0),
                        min(max(m["az_prev"] / 10.0, -5.0), 5.0),
                        m["u_prev"],
                        self.I / I_MAX,
                        m["z_ref"]], dtype=np.float64)
        return obs


# ---------------------------------------------------------------------------------------------
# Plants (identified models)
# ---------------------------------------------------------------------------------------------
# coefficients identified in 1_identification_linear.ipynb (model C, tanh, beta = 0.9)
THETA_TANH = dict(a=0.7804, b=6.4373, c=-2.1545, beta=0.90)


class TanhPlant:
    """Hammerstein: static saturation tanh then first-order lag, plus gravity.
    Written in continuous parameters so it can be perturbed physically:
        tau = -Ts/ln(a),  K = b/(1-a)  (max thrust gain),  g0 = c/(1-a)  (= -g)"""
    name = "tanh (identified)"

    def __init__(self, a=THETA_TANH["a"], b=THETA_TANH["b"], c=THETA_TANH["c"], beta=THETA_TANH["beta"]):
        self.tau = -Ts / np.log(a)
        self.K = b / (1 - a)
        self.g0 = c / (1 - a)
        self.beta = beta
        self._set()
        self.az = self.g0

    def _set(self):
        self.a = np.exp(-Ts / self.tau)

    def perturbed(self, rng, k_tau=(0.8, 1.25), k_K=(0.85, 1.15), k_beta=(0.85, 1.15), d_g=0.4):
        p = TanhPlant.__new__(TanhPlant)
        p.tau = self.tau * rng.uniform(*k_tau)
        p.K = self.K * rng.uniform(*k_K)
        p.beta = self.beta * rng.uniform(*k_beta)
        p.g0 = self.g0 + rng.uniform(-d_g, d_g)
        p._set()
        p.az = p.g0
        return p

    def static(self, u):
        return self.K * np.tanh(self.beta * u) + self.g0

    def reset(self, u0=0.0):
        self.az = float(self.static(u0))      # at rest with u0 held for a long time
        return self.az

    def step(self, u):
        """u held during Ts -> az at the next sample (exact ZOH discretisation)."""
        self.az = self.a * self.az + (1 - self.a) * float(self.static(u))
        return self.az

    def hover_u(self):
        return float(np.arctanh(-self.g0 / self.K) / self.beta)


class NNPlant:
    """The MLP of 2_identification_nn.ipynb as a plant:  az[k+1] = F(az[k], u[k])  (2 -> 16 tanh -> 1, inputs and
    output standardised). Same indexing as TanhPlant.step(): u held during Ts -> next az.
    The weights are taken out of sklearn and run in numpy (sklearn.predict is ~50x slower per call).

    perturbed() gives the same physical knobs as TanhPlant, applied around the learned map:
        speed  k_tau : az + (F - az) / k_tau          (slower / faster motors)
        gain   k_K   : F evaluated in the frame (az - g0) / k_K, output scaled back by k_K
        offset d_g   : added to az (hover thrust different from the model's)"""
    name = "NN (identified)"
    G0 = -9.81

    def __init__(self, W1, b1, W2, b2, mx, sx, my, sy, k_tau=1.0, k_K=1.0, d_g=0.0, settle=400):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.mx, self.sx, self.my, self.sy = mx, sx, my, sy
        self.k_tau, self.k_K, self.d_g, self.settle = k_tau, k_K, d_g, settle
        self.az = self.G0

    @classmethod
    def from_sklearn(cls, net, sx, sy):
        return cls(net.coefs_[0], net.intercepts_[0], net.coefs_[1][:, 0], net.intercepts_[1][0],
                   sx.mean_, sx.scale_, sy.mean_[0], sy.scale_[0])

    def _F0(self, az, u):
        h = np.tanh(((np.array([az, u]) - self.mx) / self.sx) @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2) * self.sy + self.my

    def _F(self, az, u):
        a = self.G0 + (az - self.d_g - self.G0) / self.k_K           # back to the nominal frame
        f = self.G0 + self.k_K * (self._F0(a, u) - self.G0) + self.d_g
        return az + (f - az) / self.k_tau

    def perturbed(self, rng, k_tau=(0.8, 1.25), k_K=(0.85, 1.15), d_g=0.4):
        return NNPlant(self.W1, self.b1, self.W2, self.b2, self.mx, self.sx, self.my, self.sy,
                       k_tau=rng.uniform(*k_tau), k_K=rng.uniform(*k_K),
                       d_g=rng.uniform(-d_g, d_g), settle=self.settle)

    def reset(self, u0=0.0):
        """At rest with u0 held for a long time: the fixed point az = F(az, u0)."""
        az = self.G0
        for _ in range(self.settle):
            nxt = self._F(az, u0)
            if abs(nxt - az) < 1e-7:
                break
            az = nxt
        self.az = nxt
        return self.az

    def step(self, u):
        self.az = self._F(self.az, u)
        return self.az


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------
class DroneAltitudeEnv(gym.Env):
    """Altitude control of the drone on an identified model.

    randomize=True  (training): random missions, random initial states, perturbed model
                                parameters, process noise and sensor noise (domain randomisation).
    randomize=False (export / evaluation): Figure 1 mission, start exactly like Simulink,
                                nominal model, no noise (unless noise=... is given).
    """
    metadata = {"render_modes": []}

    def __init__(self, plant=None, randomize=False, noise=None, plant_sampler=None, fig1_prob=0.4):
        super().__init__()
        self.nominal = plant if plant is not None else TanhPlant()
        self.randomize = randomize
        self.fixed_noise = noise                  # dict(z=, vz=, az=) std, or None
        self.plant_sampler = plant_sampler        # optional callable rng -> plant (e.g. NN plant)
        self.fig1_prob = fig1_prob                # share of training flights on the Figure 1 mission
        self.observer = Observer()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)


    # -- helpers ---------------------------------------------------------------------------
    def _measure(self):
        n = self.noise
        r = self.np_random
        m = {"t": self.k * Ts,
             "z": self.z + n["z"] * r.standard_normal(),
             "vz": self.vz + n["vz"] * r.standard_normal(),
             "az_prev": (0.0 if self.k == 0 else self.az_prev + n["az"] * r.standard_normal()),
             "u_prev": self.u_prev}
        m["z_ref"] = self.ref(m["t"])
        return m

    def _obs(self):
        self.m = self._measure()
        return self.observer(self.m).astype(np.float32)

    # -- gym API -----------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.k = 0
        self.observer.reset()
        if self.randomize:
            self.ref = mission if r.random() < self.fig1_prob else random_mission(r)
            if self.plant_sampler is not None and r.random() < 0.5:
                self.plant = self.plant_sampler(r)
            else:
                self.plant = self.nominal.perturbed(r) if hasattr(self.nominal, "perturbed") else self.nominal
            lvl = r.uniform(0.0, 1.0)
            self.noise = {"z": 0.01 * lvl * r.random(), "vz": 0.1 * lvl * r.random(),
                          "az": 0.6 * lvl * r.random()}
            self.w_az = 0.15 * r.random()             # process noise (unmodelled disturbances)
        else:
            self.ref, self.plant = mission, self.nominal
            self.noise = self.fixed_noise or {"z": 0.0, "vz": 0.0, "az": 0.0}
            self.w_az = 0.0

        if self.randomize and self.fig1_prob < 1 and r.random() < 0.3:      # random flying state
            self.u_prev = float(r.uniform(-0.2, 0.9))
            self.az = self.plant.reset(self.u_prev)
            self.z, self.vz = float(r.uniform(-0.2, 0.7)), float(r.uniform(-0.8, 0.8))
            self.az_prev = self.az
        else:                                         # exactly like Simulink at t = 0
            self.u_prev = 0.0
            # motors off: az = -g (as at the start of every recorded experiment). In training, 25 %
            # of the flights start with the motors partly spun up, in case Simulink does not.
            u0 = r.uniform(0.0, 0.4) if (self.randomize and r.random() < 0.25) else 0.0
            self.az = self.plant.reset(u0)
            self.z, self.vz = 0.0, 0.0
            self.az_prev = self.az                    # (the Unit Delay still outputs 0 at k = 0)
        return self._obs(), {"m": dict(self.m)}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        az0 = self.az
        az1 = self.plant.step(u) + self.w_az * self.np_random.standard_normal()
        self.plant.az = az1                       # the process noise stays in the plant state
        # exact integration if az is linear between samples (good to << 1 mm here)
        self.z += Ts * self.vz + Ts ** 2 * (2 * az0 + az1) / 6
        self.vz += Ts * (az0 + az1) / 2
        self.az_prev, self.az = az0, az1
        du = u - self.u_prev
        self.u_prev = u
        self.k += 1

        z_ref = self.ref(self.k * Ts)
        e = z_ref - self.z
        below = max(0.0, Z_GROUND - self.z)
        reward = (1.0 - min(abs(e), 1.0)
                  - 0.5 * (1.0 - np.exp(-(e / 0.05) ** 2))     # precision term
                  - 0.05 * self.vz ** 2                          # no aggressive speeds
                  - 0.5 * du ** 2                                # smooth command (noise!)
                  - 10.0 * below)                                # never go through the ground
        terminated = bool(self.z < Z_GROUND - 0.25 or self.z > 3.0 or abs(self.vz) > 6.0)
        if terminated:
            reward -= 50.0
        truncated = self.k >= N_STEPS
        obs = self._obs()
        return obs, float(reward), terminated, truncated, {"m": dict(self.m), "z_true": self.z}


# ---------------------------------------------------------------------------------------------
# Linear baseline: PID with feedback linearisation through the identified tanh model
# ---------------------------------------------------------------------------------------------
class PIDBaseline:
    """a_cmd = Kp e - Kd vz + Ki int(e),  u = f^-1(a_cmd)  with f the identified static map.
    Uses only the measurements the RL policy has."""

    def __init__(self, Kp=12.0, Kd=8.0, Ki=0.5, model=None):
        self.Kp, self.Kd, self.Ki = Kp, Kd, Ki
        self.mdl = model or TanhPlant()
        self.I = 0.0

    def reset(self):
        self.I = 0.0

    def __call__(self, m):
        e = m["z_ref"] - m["z"]
        self.I = float(np.clip(self.I + Ts * e, -I_MAX, I_MAX))
        a_cmd = self.Kp * e - self.Kd * m["vz"] + self.Ki * self.I
        x = np.clip((a_cmd - self.mdl.g0) / self.mdl.K, -0.999, 0.999)
        return float(np.clip(np.arctanh(x) / self.mdl.beta, -1, 1))


def rollout(env, controller, seed=0):
    """controller: callable(obs, m) -> u. Returns a dict of arrays (true z, measurements, u)."""
    obs, info = env.reset(seed=seed)
    log = {k: [] for k in ("t", "z", "z_meas", "vz", "az", "u", "z_ref", "r")}
    z_true = env.z
    for _ in range(N_STEPS):
        u = controller(obs, info["m"])
        log["t"].append(info["m"]["t"]); log["z"].append(z_true); log["z_meas"].append(info["m"]["z"])
        log["vz"].append(env.vz); log["az"].append(env.az); log["z_ref"].append(info["m"]["z_ref"]); log["u"].append(u)
        obs, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        z_true = info["z_true"]
        log["r"].append(r)
        if term or trunc:
            break
    out = {k: np.array(v) for k, v in log.items()}
    out["rmse"] = float(np.sqrt(np.mean((out["z_ref"] - out["z"]) ** 2)))
    out["min_z"] = float(out["z"].min())
    out["crashed"] = bool(len(out["t"]) < N_STEPS or out["min_z"] < Z_GROUND - 0.25)
    out["du_rms"] = float(np.sqrt(np.mean(np.diff(out["u"]) ** 2)))
    return out


def load_nn_plant(path=os.path.join(MODELS, "drone_model_nn.joblib")):
    """The MLP saved by 2_identification_nn.ipynb, wrapped as a plant."""
    import joblib
    d = joblib.load(path)
    return NNPlant.from_sklearn(d["net"], d["sx"], d["sy"])
