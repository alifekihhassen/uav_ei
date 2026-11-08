"""
Motion along x (advanced task 2: land at a different position) - Gymnasium environment around the
IDENTIFIED pitch model (never the protected .p).

Same plan as the yaw (yaw_env.py): the altitude policy keeps controlling the collective thrust u_T;
this environment trains a THIRD policy that only controls the differential command u_x on motors 1
and 3. The mixer (generate_x_data.m) turns the commands into the 4 motor commands:

    u1 = u_T - u_x,   u3 = u_T + u_x,   u2 = u4 = u_T           (each must stay in [-1, 1])

Plant (6_identification_x.ipynb):
    - XNNPlant   : MLP  w_dot_y[k+1] = F(w_dot_y[k], u_x[k], u_T[k])   -> training plant
    - XTanhPlant : model C, w_dot_y[k+1] = a w_dot_y[k] + b [tanh(beta(u_T+u_x)) - tanh(beta(u_T-u_x))] + c
                   -> "never seen" test drone and cascade PD baseline
Kinematics (known physics, checked in the notebook): wy and theta are integrals of w_dot_y (trapezoid),
the tilted thrust gives ax = T sin(theta) with T = az + g the thrust acceleration of the altitude model,
vx and x are integrals of ax. No restoring torque, no damping, no drag: a chain of four integrators.

Measurement timing, chosen like EINT_RL_closed_loop.slx for the altitude:
    t, x, vx, wy      : sampled at t_k (zero-order hold)
    w_dot_y_prev      : w_dot_y(t_{k-1}) (Unit Delay, 0 at k = 0)
    ax_prev, az_prev  : ax(t_{k-1}), az(t_{k-1}) (Unit Delays, 0 at k = 0), for the tilt estimate
    u_x_prev          : u_x[k-1]         (Unit Delay, 0 at k = 0)
    u_T               : the thrust command of the altitude policy at the same step
The theta OUTPUT of the simulator is unusable noise: the observer estimates it from wy and the
accelerometer (TiltEstimator).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import drone_env as de

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

Ts = 0.05
G = 9.81
X_MAX = 0.02                      # |u_x| range where the network matches model C (notebook 6, section 6)
UT_DATA = (0.2, 0.7)              # u_T range of the identification data
U_HOVER = 0.387                   # hover thrust (altitude model)
N_STEPS = 220                     # 11 s

# ---------------------------------------------------------------------------------------------
# Mission: move to the landing spot at the top of the flight (the SAME function goes into MATLAB)
#   0-1 s on the platform, 1-2 s climb to 0.5 m, move to x = X_LAND from 2 s, landing from 7 s.
# ---------------------------------------------------------------------------------------------
T_MOVE = 2.0
X_LAND = 1.0


def x_mission(t):
    """Reference position [m]. Threshold shifted by Ts/2, as for the altitude and the yaw."""
    return X_LAND if t >= T_MOVE - Ts / 2 else 0.0


def random_x_mission(rng):
    """Training references: random staircases of steps from 0.3 to 1.5 m (either direction), holds of
    2 to 5 s, inside +-2 m. The policy learns to go anywhere, not to memorise one move."""
    t_sw, levels = [0.0], [0.0]
    t, x = rng.uniform(0.5, 3.0), 0.0
    while t < N_STEPS * Ts:
        step = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])
        x = float(np.clip(x + step, -2.0, 2.0))
        t_sw.append(t); levels.append(x)
        t += rng.uniform(2.0, 5.0)
    t_sw, levels = np.array(t_sw), np.array(levels)
    return lambda t: float(levels[np.searchsorted(t_sw, t + 1e-9, side="right") - 1])


# ---------------------------------------------------------------------------------------------
# Task 3: follow the ground vehicle. The reference is the platform: position, velocity AND acceleration
# (the vehicle sends them). In the frame of the vehicle the problem is the one of the fixed landing spot,
# plus the vehicle's acceleration, which the tracking policy sees (feedforward: theta ~ a / g).
#   vehicle_mission: at rest until T_VEH, then A_VEH up to V_VEH, then constant speed (the SAME function
#   goes into MATLAB)
# ---------------------------------------------------------------------------------------------
T_VEH, A_VEH, V_VEH = 2.0, 0.5, 1.0
V_MAX_VEH, A_MAX_VEH = 1.5, 0.8   # training range of the random vehicle profiles


def vehicle_mission(t):
    """(x, v, a) of the platform. Thresholds shifted by Ts/2, as the other missions."""
    t1 = T_VEH + V_VEH / A_VEH
    if t < T_VEH - Ts / 2:
        return 0.0, 0.0, 0.0
    if t < t1 - Ts / 2:
        return 0.5 * A_VEH * (t - T_VEH) ** 2, A_VEH * (t - T_VEH), A_VEH
    return 0.5 * A_VEH * (t1 - T_VEH) ** 2 + V_VEH * (t - t1), V_VEH, 0.0


def random_vehicle(rng):
    """Training vehicle profiles as arrays (x, v, a) over N_STEPS + 1 samples: piecewise-constant
    accelerations (70 % in +-A_MAX_VEH, 30 % zero) of 1 to 4 s, speed limited to +-V_MAX_VEH; with
    probability 0.3 the drone is also asked to move by 0.3-1 m relative to the platform at a random time."""
    n = N_STEPS + 1
    acc = np.zeros(n)
    k = int(rng.integers(0, 60))
    while k < n:
        L = int(rng.integers(20, 80))
        acc[k:k + L] = rng.uniform(-A_MAX_VEH, A_MAX_VEH) if rng.random() < 0.7 else 0.0
        k += L
    vel, pos = np.zeros(n), np.zeros(n)
    for k in range(n - 1):
        vel[k + 1] = np.clip(vel[k] + Ts * acc[k], -V_MAX_VEH, V_MAX_VEH)
        acc[k] = (vel[k + 1] - vel[k]) / Ts                # what the speed limit leaves of it
        pos[k + 1] = pos[k] + Ts * (vel[k] + vel[k + 1]) / 2
    if rng.random() < 0.3:
        k0 = int(rng.integers(20, n - 40))
        pos[k0:] += rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
    return pos, vel, acc


def reference_arrays(f):
    """(x, v, a) arrays of a reference given as a function of t: x(t) only (a fixed spot, v = a = 0) or
    (x, v, a)(t)."""
    vals = [f(k * Ts) for k in range(N_STEPS + 1)]
    if np.ndim(vals[0]) == 0:
        return np.array(vals, float), np.zeros(len(vals)), np.zeros(len(vals))
    return tuple(np.array(c, float) for c in zip(*vals))


def random_thrust(rng):
    """The thrust u_T chosen by the altitude policy, as seen by the x controller: mostly near hover,
    sometimes anywhere in the identified range, changing every 0.5 to 3 s. Both the pitch gain and
    the thrust that pushes the drone along x depend on u_T."""
    u = np.zeros(N_STEPS + 1)
    k = int(round(rng.uniform(0, 1.0) / Ts)) if rng.random() < 0.5 else 0   # motors off at start
    while k <= N_STEPS:
        L = int(rng.integers(10, 60))
        lvl = np.clip(rng.normal(U_HOVER, 0.05), *UT_DATA) if rng.random() < 0.7 else rng.uniform(*UT_DATA)
        u[k:k + L] = lvl
        k += L
    return u


def hover_thrust():
    """Evaluation thrust: motors off at t = 0, then hover."""
    u = np.full(N_STEPS + 1, U_HOVER)
    u[0] = 0.0
    return u


# ---------------------------------------------------------------------------------------------
# Observation (the SAME computation goes into the MATLAB controller)
# ---------------------------------------------------------------------------------------------
I_MAX = 0.5                       # anti-windup bound of the position-error integral [m.s]
TH_MAX = 0.3                      # tilt scale of the observation [rad]
OBS_NAMES = ("e", "vx/2", "theta/TH_MAX", "wy", "w_dot_y_prev/5", "u_x_prev/X_MAX", "I/I_MAX", "u_T")
TAU_TILT = 1.0                    # time constant of the accelerometer correction of the tilt [s]
T_MIN_TILT = 2.0                  # below this thrust acceleration [m/s^2] ax/(az+g) says nothing


class TiltEstimator:
    """Complementary filter for theta (the simulator's theta output is noise):
        - short term: integral of wy (trapezoid), exact but it integrates the gyro noise too: a random
          walk (11 cm of drift at landing with 0.005 rad/s of noise on wy, full mission);
        - long term : the accelerometer, tan(theta) = ax / (az + g) (tilted thrust, notebook 6), with no
          drift; it is the tilt of the PREVIOUS sample (ax_prev, az_prev through Unit Delays), so it
          corrects theta[k-1] before the gyro step to k. Used only when there is thrust.
    theta[k-1] <- a theta[k-1] + (1 - a) atan2(ax_prev, az_prev + g),   a = exp(-Ts / TAU_TILT)
    theta[k]    = theta[k-1] + Ts (wy[k-1] + wy[k]) / 2"""
    ALPHA = float(np.exp(-Ts / TAU_TILT))

    def __init__(self):
        self.reset()

    def reset(self):
        self.theta, self.wy_last = 0.0, 0.0

    def __call__(self, m):
        if m["t"] > 0:
            T_prev = m["az_prev"] + G
            if T_prev > T_MIN_TILT:
                self.theta = self.ALPHA * self.theta + (1 - self.ALPHA) * np.arctan2(m["ax_prev"], T_prev)
            self.theta += Ts * (self.wy_last + m["wy"]) / 2
        self.wy_last = m["wy"]
        return self.theta


OBS_NAMES_TRACK = (*OBS_NAMES, "ax_ref")


class XObserver:
    """obs = [e, (vx - vx_ref)/2, theta/TH_MAX, wy, w_dot_y_prev/5, u_x_prev/X_MAX, I/I_MAX, u_T]
    (+ [ax_ref] with tracking=True), e = x_ref - x (clipped). vx_ref, ax_ref: velocity and acceleration of
    the reference (0 for a fixed spot, the platform's when following the vehicle; missing keys = 0): in the
    frame of the reference the problem is the same, and ax_ref is the push the policy must anticipate.
    theta from TiltEstimator (gyro + accelerometer). No absolute position: the dynamics do not depend on x.
    (vx, theta, wy, w_dot_y_prev, u_x_prev, u_T) is the state of the identified chain with its delay:
    Markov. I removes a static error caused by a residual torque."""

    def __init__(self, tracking=False):
        self.tracking = tracking
        self.tilt = TiltEstimator()
        self.reset()

    def reset(self):
        self.I = 0.0
        self.tilt.reset()

    def __call__(self, m):
        e = m["x_ref"] - m["x"]
        self.I = min(max(self.I + Ts * e, -I_MAX), I_MAX)
        theta = self.tilt(m)
        obs = [min(max(e, -2.0), 2.0),
               min(max((m["vx"] - m.get("vx_ref", 0.0)) / 2.0, -5.0), 5.0),
               min(max(theta / TH_MAX, -5.0), 5.0),
               min(max(m["wy"], -5.0), 5.0),
               min(max(m["w_dot_y_prev"] / 5.0, -5.0), 5.0),
               m["u_x_prev"] / X_MAX,
               self.I / I_MAX,
               m["u_T"]]
        if self.tracking:
            obs.append(min(max(m.get("ax_ref", 0.0), -5.0), 5.0))
        return np.array(obs, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Plants (identified models)
# ---------------------------------------------------------------------------------------------
class XNNPlant:
    """The MLP of 6_identification_x.ipynb, run in numpy. Same physical knobs as YawNNPlant:
        speed  k_tau : w + (F - w) / k_tau      (slower / faster motors)
        gain   k_K   : F evaluated in the frame w / k_K, output scaled back by k_K (pitch authority)
        offset d     : constant torque added to F (asymmetric motors, model bias)"""
    name = "x NN (identified)"

    def __init__(self, W1, b1, W2, b2, mx, sx, my, sy, k_tau=1.0, k_K=1.0, d=0.0):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.mx, self.sx, self.my, self.sy = mx, sx, my, sy
        self.k_tau, self.k_K, self.d = k_tau, k_K, d
        self.w = 0.0

    @classmethod
    def from_sklearn(cls, net, sx, sy):
        return cls(net.coefs_[0], net.intercepts_[0], net.coefs_[1][:, 0], net.intercepts_[1][0],
                   sx.mean_, sx.scale_, sy.mean_[0], sy.scale_[0])

    def _F0(self, w, ux, uT):
        uT = min(max(uT, UT_DATA[0]), UT_DATA[1])          # stay in the identified range
        h = np.tanh(((np.array([w, ux, uT]) - self.mx) / self.sx) @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2) * self.sy + self.my

    def _F(self, w, ux, uT):
        f = self.k_K * self._F0(w / self.k_K, ux, uT) + self.d
        return w + (f - w) / self.k_tau

    def perturbed(self, rng, k_tau=(0.8, 1.25), k_K=(0.85, 1.15), d=0.05):
        return XNNPlant(self.W1, self.b1, self.W2, self.b2, self.mx, self.sx, self.my, self.sy,
                        k_tau=rng.uniform(*k_tau), k_K=rng.uniform(*k_K), d=rng.uniform(-d, d))

    def reset(self):
        self.w = 0.0
        return self.w

    def step(self, ux, uT):
        """u_x, u_T held during Ts -> w_dot_y at the next sample."""
        self.w = self._F(self.w, ux, uT)
        return self.w


class XTanhPlant:
    """Model C of 6_identification_x.ipynb: first-order lag on the torque difference of motors 3 and 1,
    each a saturating tanh. Physical form: tau = -Ts/ln(a), K = b/(1-a), c0 = c/(1-a)."""
    name = "x tanh (model C)"

    def __init__(self, theta, beta):
        a, b, c = theta
        self.tau, self.K, self.c0, self.beta = -Ts / np.log(a), b / (1 - a), c / (1 - a), beta
        self.a = np.exp(-Ts / self.tau)
        self.w = 0.0

    def torque(self, ux, uT):
        return self.K * (np.tanh(self.beta * (uT + ux)) - np.tanh(self.beta * (uT - ux))) + self.c0

    def gain(self, uT):
        """d w_dot_y / d u_x at u_x = 0 (used by the PD baseline)."""
        return 2 * self.K * self.beta / np.cosh(self.beta * uT) ** 2

    def perturbed(self, rng, k_tau=(0.8, 1.25), k_K=(0.85, 1.15), k_beta=(0.85, 1.15), d=0.05):
        p = XTanhPlant.__new__(XTanhPlant)
        p.tau, p.K = self.tau * rng.uniform(*k_tau), self.K * rng.uniform(*k_K)
        p.beta, p.c0 = self.beta * rng.uniform(*k_beta), self.c0 + rng.uniform(-d, d)
        p.a, p.w = np.exp(-Ts / p.tau), 0.0
        return p

    def reset(self):
        self.w = 0.0
        return self.w

    def step(self, ux, uT):
        self.w = self.a * self.w + (1 - self.a) * float(self.torque(ux, uT))
        return self.w


def load_x_plant(path=os.path.join(MODELS, "x_model_nn.joblib")):
    import joblib
    d = joblib.load(path)
    return XNNPlant.from_sklearn(d["net"], d["sx"], d["sy"])


def load_x_test_plant(path=os.path.join(MODELS, "x_model_nn.joblib")):
    import joblib
    d = joblib.load(path)
    return XTanhPlant(d["theta_C"], d["beta_C"])


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------
class XEnv(gym.Env):
    """Position control along x on the identified pitch model.

    randomize=True  (training): random references, random thrust profiles, random initial speed and
                                tilt, perturbed pitch and thrust models, process and sensor noise.
    randomize=False (evaluation / export): the mission of x_mission(), thrust = hover_thrust()
                                (or `thrust`), nominal models, no noise (unless noise=... is given).
    The action is u_x itself, in [-X_MAX, X_MAX]; the mixer then clips it to the margin 1 - |u_T|.
    thrust_plant: the altitude model that gives the thrust acceleration T = az + g (default: its NN).
    du_weight: weight of the smoothness term of the reward.
    tracking=True: the reference moves (task 3: follow the vehicle). The observation gets the reference's
    acceleration (XObserver(tracking=True)); training mixes the move mission in both directions, random
    staircases, random vehicle profiles and the vehicle mission. mission: the evaluation reference, a
    function of t returning x or (x, v, a) (default: x_mission, or vehicle_mission when tracking)."""
    metadata = {"render_modes": []}

    def __init__(self, plant=None, randomize=False, noise=None, thrust=None, fig_prob=0.4, thrust_plant=None,
                 du_weight=0.2, tracking=False, mission=None):
        super().__init__()
        self.nominal = plant if plant is not None else load_x_plant()
        self.nominal_T = thrust_plant if thrust_plant is not None else de.load_nn_plant()
        self.randomize, self.fixed_noise, self.fixed_thrust, self.fig_prob = randomize, noise, thrust, fig_prob
        self.du_weight, self.tracking = du_weight, tracking
        self.mission = mission or (vehicle_mission if tracking else x_mission)
        self.observer = XObserver(tracking)
        self.action_space = spaces.Box(-X_MAX, X_MAX, shape=(1,), dtype=np.float32)
        n_obs = len(OBS_NAMES_TRACK if tracking else OBS_NAMES)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    def _measure(self):
        n, r = self.noise, self.np_random
        m = {"t": self.k * Ts,
             "x": self.x + n["x"] * r.standard_normal(),
             "vx": self.vx + n["vx"] * r.standard_normal(),
             "wy": self.wy + n["wy"] * r.standard_normal(),
             "w_dot_y_prev": 0.0 if self.k == 0 else self.w_prev + n["w"] * r.standard_normal(),
             "ax_prev": 0.0 if self.k == 0 else self.ax_prev + n.get("ax", 0.0) * r.standard_normal(),
             "az_prev": 0.0 if self.k == 0 else self.az_prev + n.get("az", 0.0) * r.standard_normal(),
             "u_x_prev": self.u_prev,
             "u_T": float(self.uT[self.k])}
        m["x_ref"], m["vx_ref"], m["ax_ref"] = (float(a[self.k]) for a in (self.xr, self.vr, self.ar))
        return m

    def _obs(self):
        self.m = self._measure()
        return self.observer(self.m).astype(np.float32)

    def _training_reference(self, r):
        if not self.tracking:
            return reference_arrays(x_mission if r.random() < self.fig_prob else random_x_mission(r))
        u, sign = r.random(), float(r.choice([-1, 1]))       # both directions: no bias towards +x
        if u < 0.25:
            return reference_arrays(lambda t: sign * x_mission(t))
        if u < 0.5:
            return reference_arrays(random_x_mission(r))
        if u < 0.8:
            return random_vehicle(r)
        return tuple(sign * a for a in reference_arrays(vehicle_mission))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.k, self.u_prev = 0, 0.0
        self.observer.reset()
        if self.randomize:
            self.xr, self.vr, self.ar = self._training_reference(r)
            self.plant, self.T_plant = self.nominal.perturbed(r), self.nominal_T.perturbed(r)
            self.uT = random_thrust(r)
            lvl = r.uniform(0.0, 1.0)
            self.noise = {"x": 0.01 * lvl * r.random(), "vx": 0.05 * lvl * r.random(),
                          "wy": 0.02 * lvl * r.random(), "w": 0.2 * lvl * r.random(),
                          "ax": 0.05 * lvl * r.random(), "az": 0.3 * lvl * r.random()}
            self.w_proc = 0.02 * r.random()                # process noise on w_dot_y
        else:
            self.xr, self.vr, self.ar = reference_arrays(self.mission)
            self.plant, self.T_plant = self.nominal, self.nominal_T
            self.uT = self.fixed_thrust if self.fixed_thrust is not None else hover_thrust()
            self.noise = self.fixed_noise or {"x": 0.0, "vx": 0.0, "wy": 0.0, "w": 0.0}
            self.w_proc = 0.0
        self.w = self.plant.reset()
        self.az = self.T_plant.reset(float(self.uT[0]))
        self.x = self.vx = self.wy = self.theta = 0.0
        if self.randomize and r.random() < 0.3:            # already moving / tilted
            self.vx, self.theta = float(r.uniform(-0.5, 0.5)), float(r.uniform(-0.1, 0.1))
        self.ax = (self.az + G) * np.sin(self.theta)
        self.w_prev, self.ax_prev, self.az_prev = self.w, 0.0, 0.0
        return self._obs(), {"m": dict(self.m)}

    def step(self, action):
        uT = float(self.uT[self.k])
        margin = 1.0 - abs(uT)
        ux = float(np.clip(np.asarray(action).reshape(-1)[0], -X_MAX, X_MAX))
        ux = min(max(ux, -margin), margin)                 # mixer: thrust has priority
        w0 = self.w
        w1 = self.plant.step(ux, uT) + self.w_proc * self.np_random.standard_normal()
        self.plant.w = w1
        wy1 = self.wy + Ts * (w0 + w1) / 2                 # trapezoid, as in the simulator
        th1 = self.theta + Ts * (self.wy + wy1) / 2
        az1 = self.T_plant.step(uT)
        ax1 = (az1 + G) * np.sin(th1)                      # tilted thrust
        self.ax_prev = self.ax                             # accelerometer through a Unit Delay:
        self.az_prev = (self.az + G) * np.cos(self.theta) - G   # tilted thrust, vertical part
        self.x += Ts * self.vx + Ts ** 2 * (2 * self.ax + ax1) / 6
        self.vx += Ts * (self.ax + ax1) / 2
        self.wy, self.theta, self.ax, self.az, self.w_prev, self.w = wy1, th1, ax1, az1, w0, w1
        du = (ux - self.u_prev) / X_MAX
        self.u_prev = ux
        self.k += 1

        e = self.xr[self.k] - self.x
        reward = (1.0 - min(abs(e), 2.0) / 2.0                 # tracking (and alive bonus)
                  - 0.5 * (1.0 - np.exp(-(e / 0.05) ** 2))      # precision: within ~5 cm
                  - 2.0 * self.theta ** 2                       # no violent tilt
                  - self.du_weight * du ** 2)                   # smooth command (noise!)
        terminated = bool(abs(self.theta) > 0.6 or abs(e) > 4.0)   # out of the identified envelope
        if terminated:
            reward -= 50.0
        truncated = self.k >= N_STEPS
        obs = self._obs()
        return obs, float(reward), terminated, truncated, {"m": dict(self.m), "x_true": self.x,
                                                           "theta_true": self.theta, "u_x": ux}


# ---------------------------------------------------------------------------------------------
# Linear baseline: cascade PD (position -> tilt -> torque) with feedback linearisation through model C
# ---------------------------------------------------------------------------------------------
class XPD:
    """Outer loop: ax_cmd = Kp e - Kd vx + Ki int(e), theta_cmd = atan(ax_cmd / g) clipped to TH_CMD.
    Inner loop: w_cmd = Kth (theta_cmd - theta) - Kw wy,  u_x = w_cmd / gain(u_T)  (model C linearised at
    u_x = 0). theta from the same TiltEstimator as the observer. Uses only the measurements the policy has.
    Gains grid-searched on the NN and tanh models (mission move of 1 m): settles within 5 cm in 2.55 s,
    overshoot 4.6 cm. Ki = 0: any integral winds up during the move and doubles the overshoot; a residual
    torque d leaves a static error g d / (Kth Kp) instead (1.6 cm for d = 0.05 rad/s^2)."""
    TH_CMD = 0.25

    def __init__(self, Kp=1.0, Kd=1.5, Ki=0.0, Kth=30.0, Kw=12.0, model=None):
        self.Kp, self.Kd, self.Ki, self.Kth, self.Kw = Kp, Kd, Ki, Kth, Kw
        self.mdl = model or load_x_test_plant()
        self.tilt = TiltEstimator()
        self.reset()

    def reset(self):
        self.I = 0.0
        self.tilt.reset()

    def __call__(self, m):
        e = m["x_ref"] - m["x"]
        self.I = float(np.clip(self.I + Ts * e, -I_MAX, I_MAX))
        self.theta = self.tilt(m)
        ax_cmd = self.Kp * e - self.Kd * m["vx"] + self.Ki * self.I
        th_cmd = float(np.clip(np.arctan(ax_cmd / G), -self.TH_CMD, self.TH_CMD))
        w_cmd = self.Kth * (th_cmd - self.theta) - self.Kw * m["wy"]
        gain = self.mdl.gain(min(max(m["u_T"], UT_DATA[0]), UT_DATA[1]))
        return float(np.clip((w_cmd - self.mdl.c0) / gain, -X_MAX, X_MAX))


# ---------------------------------------------------------------------------------------------
# Rollout and metrics
# ---------------------------------------------------------------------------------------------
def rollout(env, controller, seed=0):
    """controller: callable(obs, m) -> u_x. Returns a dict of arrays + metrics."""
    obs, info = env.reset(seed=seed)
    log = {k: [] for k in ("t", "x", "x_meas", "x_ref", "vx", "theta", "wy", "u", "u_T", "r")}
    x_true, th_true = env.x, env.theta
    for _ in range(N_STEPS):
        u = controller(obs, info["m"])
        m = info["m"]
        log["t"].append(m["t"]); log["x"].append(x_true); log["x_meas"].append(m["x"])
        log["x_ref"].append(m["x_ref"]); log["vx"].append(env.vx); log["theta"].append(th_true)
        log["wy"].append(env.wy); log["u_T"].append(m["u_T"])
        obs, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        log["u"].append(info["u_x"]); log["r"].append(r)
        x_true, th_true = info["x_true"], info["theta_true"]
        if term or trunc:
            break
    out = {k: np.array(v) for k, v in log.items()}
    e = out["x_ref"] - out["x"]
    out["rmse"] = float(np.sqrt(np.mean(e ** 2)))
    out["final_err_cm"] = float(100 * np.abs(e[-20:]).mean())
    out["du_rms"] = float(np.sqrt(np.mean(np.diff(out["u"]) ** 2)))
    out["max_tilt_deg"] = float(np.degrees(np.abs(out["theta"]).max()))
    out["crashed"] = bool(len(out["t"]) < N_STEPS)
    out.update(move_metrics(out))
    return out


def move_metrics(o, tol=0.05):
    """For the move of the mission (from T_MOVE): settling time (|e| < tol for good) and overshoot [cm]."""
    t, x = o["t"], o["x"]
    win = t >= T_MOVE - Ts / 2
    if not win.any():
        return {"settle_s": np.nan, "overshoot_cm": np.nan}
    e = np.abs(X_LAND - x[win])
    out_tol = np.flatnonzero(e > tol)
    settle = (np.nan if len(out_tol) and out_tol[-1] == len(e) - 1
              else (out_tol[-1] + 1) * Ts if len(out_tol) else 0.0)
    return {"settle_s": settle, "overshoot_cm": float(100 * max(0.0, (x[win] - X_LAND).max()))}


def track_metrics(o):
    """Following the vehicle mission: largest error while the vehicle accelerates (and 1 s after), mean
    and largest error at constant speed (from 2 s after the end of the acceleration) [cm]."""
    t, e = o["t"], o["x_ref"] - o["x"]
    t1 = T_VEH + V_VEH / A_VEH
    acc, cru = (t >= T_VEH) & (t < t1 + 1), t >= t1 + 2
    return {"acc_max_cm": float(100 * np.abs(e[acc]).max()), "cruise_mean_cm": float(100 * e[cru].mean()),
            "cruise_max_cm": float(100 * np.abs(e[cru]).max())}
