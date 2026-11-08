"""
Yaw (rotation) - Gymnasium environment around the IDENTIFIED yaw model (never the protected .p).

Option A of the plan: the altitude policy (drone_env.py, models/sac_nn.zip) keeps controlling the
collective thrust u_T; this environment trains a SECOND policy that only controls the differential
command u_yaw. A fixed mixer (generate_yaw_data.m) turns the two commands into the 4 motor commands:

    u1 = u3 = u_T + u_yaw          u2 = u4 = u_T - u_yaw          (each must stay in [-1, 1])

Plant (4_identification_yaw.ipynb):
    - YawNNPlant   : MLP  w_dot_z[k+1] = F(w_dot_z[k], u_yaw[k], u_T[k])   -> training plant
    - YawTanhPlant : model C, w_dot_z[k+1] = a w_dot_z[k] + b [tanh(beta(u_T+u_yaw)) - tanh(beta(u_T-u_yaw))] + c
                     -> "never seen" test drone and PD baseline
Kinematics: wz and psi are integrals of w_dot_z (trapezoid: the rule that matches the simulator).

Measurement timing, chosen like EINT_RL_closed_loop.slx for the altitude:
    t, psi, wz        : sampled at t_k (zero-order hold)
    w_dot_z_prev      : w_dot_z(t_{k-1}) (Unit Delay, 0 at k = 0)
    u_yaw_prev        : u_yaw[k-1]       (Unit Delay, 0 at k = 0)
    u_T               : the thrust command of the altitude policy at the same step
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

Ts = 0.05
YAW_MAX = 0.1                     # |u_yaw| range of the identification data (beyond: extrapolation)
UT_DATA = (0.0, 0.8)              # u_T range of the identification data
U_HOVER = 0.387                   # hover thrust (altitude model)
N_STEPS = 260                     # 13 s: the full mission with the rotation

# ---------------------------------------------------------------------------------------------
# Mission: the rotation at the top of the flight (the SAME function goes into rl_controller.m)
#   0-1 s on the platform, 1-2 s climb to 0.5 m, then four quarter turns every 2 s
#   (90, 180, 270, 360 deg at t = 2, 4, 6, 8 s), landing from 10 s, end at 13 s.
# ---------------------------------------------------------------------------------------------
T_YAW = (2.0, 4.0, 6.0, 8.0)
PSI_STEP = np.pi / 2


def yaw_mission(t):
    """Reference heading [rad], unwrapped (the last step is 2*pi, a full turn, not 0).
    Thresholds shifted by Ts/2, as for the altitude, so Python and Simulink agree on t = 2, 4 ... s."""
    return PSI_STEP * sum(t >= tk - Ts / 2 for tk in T_YAW)


def random_yaw_mission(rng):
    """Training references: random staircases, mostly quarter turns (either direction) and some
    arbitrary angles, holds of 1 to 4 s. The policy learns to track any step, not Figure 1 only."""
    t_sw, levels = [0.0], [0.0]
    t, psi = rng.uniform(0.5, 3.0), 0.0
    while t < N_STEPS * Ts:
        if rng.random() < 0.7:
            psi += PSI_STEP * rng.choice([-2, -1, 1, 1, 2])
        else:
            psi += rng.uniform(-np.pi, np.pi)
        t_sw.append(t); levels.append(psi)
        t += rng.uniform(1.0, 4.0)
    t_sw, levels = np.array(t_sw), np.array(levels)
    return lambda t: float(levels[np.searchsorted(t_sw, t + 1e-9, side="right") - 1])


def random_thrust(rng):
    """The thrust u_T chosen by the altitude policy, as seen by the yaw controller: mostly near hover,
    sometimes anywhere in the identified range, changing every 0.5 to 3 s. The yaw gain depends on
    u_T, so the policy must work whatever the altitude policy is doing."""
    u = np.zeros(N_STEPS + 1)
    k = int(round(rng.uniform(0, 1.0) / Ts)) if rng.random() < 0.5 else 0   # motors off at start
    while k <= N_STEPS:
        L = int(rng.integers(10, 60))
        lvl = np.clip(rng.normal(U_HOVER, 0.05), *UT_DATA) if rng.random() < 0.7 else rng.uniform(*UT_DATA)
        u[k:k + L] = lvl
        k += L
    return u


def hover_thrust():
    """Evaluation thrust: motors off at t = 0, then hover (what the altitude policy does on average)."""
    u = np.full(N_STEPS + 1, U_HOVER)
    u[0] = 0.0
    return u


# ---------------------------------------------------------------------------------------------
# Observation (the SAME computation goes into rl_controller.m)
# ---------------------------------------------------------------------------------------------
I_MAX = 0.5                       # anti-windup bound of the heading-error integral [rad.s]
OBS_NAMES = ("e", "wz/5", "w_dot_z_prev/20", "u_yaw_prev/YAW_MAX", "I/I_MAX", "u_T")


class YawObserver:
    """obs = [e, wz/5, w_dot_z_prev/20, u_yaw_prev/YAW_MAX, I/I_MAX, u_T],  e = psi_ref - psi (clipped +-pi).
    No absolute angle: the dynamics do not depend on the heading, so the policy only needs the error.
    (wz, w_dot_z_prev, u_yaw_prev, u_T) is the state of the identified model with its delay: Markov.
    The integral I removes the static error caused by a residual torque (model bias, asymmetric motors)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.I = 0.0

    def __call__(self, m):
        e = m["psi_ref"] - m["psi"]
        self.I = min(max(self.I + Ts * e, -I_MAX), I_MAX)
        return np.array([min(max(e, -np.pi), np.pi),
                         min(max(m["wz"] / 5.0, -5.0), 5.0),
                         min(max(m["w_dot_z_prev"] / 20.0, -5.0), 5.0),
                         m["u_yaw_prev"] / YAW_MAX,
                         self.I / I_MAX,
                         m["u_T"]], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Plants (identified models)
# ---------------------------------------------------------------------------------------------
class YawNNPlant:
    """The MLP of 4_identification_yaw.ipynb, run in numpy (sklearn.predict is ~50x slower per call).
    perturbed() gives the same physical knobs as for the altitude:
        speed  k_tau : w + (F - w) / k_tau      (slower / faster motors)
        gain   k_K   : F evaluated in the frame w / k_K, output scaled back by k_K (yaw authority)
        offset d     : constant torque added to F (asymmetric motors, model bias)"""
    name = "yaw NN (identified)"

    def __init__(self, W1, b1, W2, b2, mx, sx, my, sy, k_tau=1.0, k_K=1.0, d=0.0):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.mx, self.sx, self.my, self.sy = mx, sx, my, sy
        self.k_tau, self.k_K, self.d = k_tau, k_K, d
        self.w = 0.0

    @classmethod
    def from_sklearn(cls, net, sx, sy):
        return cls(net.coefs_[0], net.intercepts_[0], net.coefs_[1][:, 0], net.intercepts_[1][0],
                   sx.mean_, sx.scale_, sy.mean_[0], sy.scale_[0])

    def _F0(self, w, uy, uT):
        uT = min(max(uT, UT_DATA[0]), UT_DATA[1])          # stay in the identified range
        h = np.tanh(((np.array([w, uy, uT]) - self.mx) / self.sx) @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2) * self.sy + self.my

    def _F(self, w, uy, uT):
        f = self.k_K * self._F0(w / self.k_K, uy, uT) + self.d
        return w + (f - w) / self.k_tau

    def perturbed(self, rng, k_tau=(0.8, 1.25), k_K=(0.85, 1.15), d=0.3):
        return YawNNPlant(self.W1, self.b1, self.W2, self.b2, self.mx, self.sx, self.my, self.sy,
                          k_tau=rng.uniform(*k_tau), k_K=rng.uniform(*k_K), d=rng.uniform(-d, d))

    def reset(self):
        self.w = 0.0                                       # motors off: no torque
        return self.w

    def step(self, uy, uT):
        """u_yaw, u_T held during Ts -> w_dot_z at the next sample."""
        self.w = self._F(self.w, uy, uT)
        return self.w


class YawTanhPlant:
    """Model C of 4_identification_yaw.ipynb: first-order lag on the torque difference of the two
    motor pairs, each a saturating tanh. Physical form: tau = -Ts/ln(a), K = b/(1-a), c0 = c/(1-a)."""
    name = "yaw tanh (model C)"

    def __init__(self, theta, beta):
        a, b, c = theta
        self.tau, self.K, self.c0, self.beta = -Ts / np.log(a), b / (1 - a), c / (1 - a), beta
        self.a = np.exp(-Ts / self.tau)
        self.w = 0.0

    def torque(self, uy, uT):
        return self.K * (np.tanh(self.beta * (uT + uy)) - np.tanh(self.beta * (uT - uy))) + self.c0

    def gain(self, uT):
        """d w_dot_z / d u_yaw at u_yaw = 0 (used by the PD baseline)."""
        return 2 * self.K * self.beta / np.cosh(self.beta * uT) ** 2

    def perturbed(self, rng, k_tau=(0.8, 1.25), k_K=(0.85, 1.15), k_beta=(0.85, 1.15), d=0.3):
        p = YawTanhPlant.__new__(YawTanhPlant)
        p.tau, p.K = self.tau * rng.uniform(*k_tau), self.K * rng.uniform(*k_K)
        p.beta, p.c0 = self.beta * rng.uniform(*k_beta), self.c0 + rng.uniform(-d, d)
        p.a, p.w = np.exp(-Ts / p.tau), 0.0
        return p

    def reset(self):
        self.w = 0.0
        return self.w

    def step(self, uy, uT):
        self.w = self.a * self.w + (1 - self.a) * float(self.torque(uy, uT))
        return self.w


def load_yaw_plant(path=os.path.join(MODELS, "yaw_model_nn.joblib")):
    import joblib
    d = joblib.load(path)
    return YawNNPlant.from_sklearn(d["net"], d["sx"], d["sy"])


def load_yaw_test_plant(path=os.path.join(MODELS, "yaw_model_nn.joblib")):
    import joblib
    d = joblib.load(path)
    return YawTanhPlant(d["theta_C"], d["beta_C"])


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------
class YawEnv(gym.Env):
    """Heading control on the identified yaw model.

    randomize=True  (training): random references, random thrust profiles, random initial yaw rate,
                                perturbed model, process and sensor noise (domain randomisation).
    randomize=False (evaluation / export): the mission of yaw_mission(), thrust = hover_thrust()
                                (or `thrust`), nominal model, no noise (unless noise=... is given).
    The action is u_yaw itself, in [-YAW_MAX, YAW_MAX]; the mixer then clips it to the margin 1 - |u_T|
    left by the thrust, so no motor command ever leaves [-1, 1] (thrust has priority)."""
    metadata = {"render_modes": []}

    def __init__(self, plant=None, randomize=False, noise=None, thrust=None, fig_prob=0.4):
        super().__init__()
        self.nominal = plant if plant is not None else load_yaw_plant()
        self.randomize, self.fixed_noise, self.fixed_thrust, self.fig_prob = randomize, noise, thrust, fig_prob
        self.observer = YawObserver()
        self.action_space = spaces.Box(-YAW_MAX, YAW_MAX, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)

    def _measure(self):
        n, r = self.noise, self.np_random
        m = {"t": self.k * Ts,
             "psi": self.psi + n["psi"] * r.standard_normal(),
             "wz": self.wz + n["wz"] * r.standard_normal(),
             "w_dot_z_prev": 0.0 if self.k == 0 else self.w_prev + n["w"] * r.standard_normal(),
             "u_yaw_prev": self.u_prev,
             "u_T": float(self.uT[self.k])}
        m["psi_ref"] = self.ref(m["t"])
        return m

    def _obs(self):
        self.m = self._measure()
        return self.observer(self.m).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.k, self.u_prev = 0, 0.0
        self.observer.reset()
        if self.randomize:
            self.ref = yaw_mission if r.random() < self.fig_prob else random_yaw_mission(r)
            self.plant = self.nominal.perturbed(r)
            self.uT = random_thrust(r)
            lvl = r.uniform(0.0, 1.0)
            self.noise = {"psi": 0.01 * lvl * r.random(), "wz": 0.1 * lvl * r.random(), "w": 1.0 * lvl * r.random()}
            self.w_proc = 0.2 * r.random()                 # process noise on w_dot_z
        else:
            self.ref, self.plant = yaw_mission, self.nominal
            self.uT = self.fixed_thrust if self.fixed_thrust is not None else hover_thrust()
            self.noise = self.fixed_noise or {"psi": 0.0, "wz": 0.0, "w": 0.0}
            self.w_proc = 0.0
        self.w = self.plant.reset()
        self.psi, self.wz = 0.0, 0.0
        if self.randomize and r.random() < 0.3:            # already turning
            self.wz = float(r.uniform(-2.0, 2.0))
        self.w_prev = self.w
        return self._obs(), {"m": dict(self.m)}

    def step(self, action):
        uT = float(self.uT[self.k])
        margin = 1.0 - abs(uT)
        uy = float(np.clip(np.asarray(action).reshape(-1)[0], -YAW_MAX, YAW_MAX))
        uy = min(max(uy, -margin), margin)                 # mixer: thrust has priority
        w0 = self.w
        w1 = self.plant.step(uy, uT) + self.w_proc * self.np_random.standard_normal()
        self.plant.w = w1
        wz1 = self.wz + Ts * (w0 + w1) / 2                 # trapezoid, as in the simulator
        self.psi += Ts * (self.wz + wz1) / 2
        self.wz, self.w_prev, self.w = wz1, w0, w1
        du = (uy - self.u_prev) / YAW_MAX
        self.u_prev = uy
        self.k += 1

        e = self.ref(self.k * Ts) - self.psi
        reward = (1.0 - min(abs(e), np.pi) / np.pi            # tracking (and alive bonus)
                  - 0.5 * (1.0 - np.exp(-(e / 0.05) ** 2))      # precision: within ~3 deg
                  - 0.01 * self.wz ** 2                         # no violent spinning
                  - 0.2 * du ** 2)                              # smooth command (noise!)
        terminated = bool(abs(self.wz) > 15.0)                 # spinning out of control
        if terminated:
            reward -= 50.0
        truncated = self.k >= N_STEPS
        obs = self._obs()
        return obs, float(reward), terminated, truncated, {"m": dict(self.m), "psi_true": self.psi, "u_yaw": uy}


# ---------------------------------------------------------------------------------------------
# Linear baseline: PD + integral with feedback linearisation through model C
# ---------------------------------------------------------------------------------------------
class YawPD:
    """w_cmd = Kp e - Kd wz + Ki int(e),  u_yaw = (w_cmd - c0) / gain(u_T)  (model C linearised at
    u_yaw = 0, so the loop gain is the same at any thrust). Uses only the measurements the policy has.
    Gains grid-searched on the NN and tanh models under the constraint overshoot < 5 deg."""

    def __init__(self, Kp=12.0, Kd=8.0, Ki=2.0, model=None):
        self.Kp, self.Kd, self.Ki = Kp, Kd, Ki
        self.mdl = model or load_yaw_test_plant()
        self.I = 0.0

    def reset(self):
        self.I = 0.0

    def __call__(self, m):
        e = m["psi_ref"] - m["psi"]
        self.I = float(np.clip(self.I + Ts * e, -I_MAX, I_MAX))
        w_cmd = self.Kp * e - self.Kd * m["wz"] + self.Ki * self.I
        return float(np.clip((w_cmd - self.mdl.c0) / self.mdl.gain(m["u_T"]), -YAW_MAX, YAW_MAX))


# ---------------------------------------------------------------------------------------------
# Rollout and metrics
# ---------------------------------------------------------------------------------------------
def rollout(env, controller, seed=0):
    """controller: callable(obs, m) -> u_yaw. Returns a dict of arrays + metrics."""
    obs, info = env.reset(seed=seed)
    log = {k: [] for k in ("t", "psi", "psi_meas", "psi_ref", "wz", "w", "u", "u_T", "r")}
    psi_true = env.psi
    for _ in range(N_STEPS):
        u = controller(obs, info["m"])
        m = info["m"]
        log["t"].append(m["t"]); log["psi"].append(psi_true); log["psi_meas"].append(m["psi"])
        log["psi_ref"].append(m["psi_ref"]); log["wz"].append(env.wz); log["w"].append(env.w)
        log["u_T"].append(m["u_T"])
        obs, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        log["u"].append(info["u_yaw"]); log["r"].append(r)
        psi_true = info["psi_true"]
        if term or trunc:
            break
    out = {k: np.array(v) for k, v in log.items()}
    e = out["psi_ref"] - out["psi"]
    out["rmse_deg"] = float(np.degrees(np.sqrt(np.mean(e ** 2))))
    out["final_err_deg"] = float(np.degrees(np.abs(e[-20:]).mean()))
    out["du_rms"] = float(np.sqrt(np.mean(np.diff(out["u"]) ** 2)))
    out["max_wz"] = float(np.abs(out["wz"]).max())
    out["crashed"] = bool(len(out["t"]) < N_STEPS)
    out.update(step_metrics(out))
    return out


def step_metrics(o, tol_deg=5.0):
    """For each quarter turn of the mission: settling time (|e| < tol for good) and overshoot."""
    t, psi = o["t"], o["psi"]
    settle, over = [], []
    for j, tk in enumerate(T_YAW):
        target = PSI_STEP * (j + 1)
        t_end = T_YAW[j + 1] if j + 1 < len(T_YAW) else t[-1] + Ts
        win = (t >= tk - Ts / 2) & (t < t_end - Ts / 2)
        if not win.any():
            settle.append(np.nan); over.append(np.nan); continue
        e = np.degrees(np.abs(target - psi[win]))
        out_tol = np.flatnonzero(e > tol_deg)
        settle.append(np.nan if len(out_tol) and out_tol[-1] == len(e) - 1
                      else (out_tol[-1] + 1) * Ts if len(out_tol) else 0.0)
        over.append(float(max(0.0, np.degrees((psi[win] - target).max()))))
    return {"settle_s": np.array(settle), "overshoot_deg": np.array(over)}
