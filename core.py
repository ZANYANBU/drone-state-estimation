"""
Drone state estimation under turbulent wind - Linear Dynamical System + Kalman filter.
ML Unit 4 project.  Core model, simulator, filter, smoother and EM.

State (9D), three identical per-axis blocks [position, velocity, wind]:
    x = [px,py,pz, vx,vy,vz, wx,wy,wz]

Per axis the dynamics are LINEAR, which is what lets us use a Kalman filter exactly:
    p_{t+1} = p_t + v_t*dt
    v_{t+1} = v_t + (a_cmd - k*(v_t - w_t))*dt      <- drag acts on AIRSPEED (v - w)
    w_{t+1} = phi*w_t + eps,  phi = exp(-dt/tau)    <- Ornstein-Uhlenbeck gust model
"""
import numpy as np

AX = 3                      # x, y, z
NS = 3 * AX                 # 9 states: 3 position, 3 velocity, 3 wind


# ----------------------------------------------------------------- model

def build_model(dt=0.01, k_drag=0.8, tau=5.0, sigma_gust=1.5, sigma_imu=0.15):
    """Return A, B, Q for the augmented linear dynamical system."""
    phi = np.exp(-dt / tau)
    # steady-state wind variance is sigma_gust^2, so the driving noise must be:
    q_w = sigma_gust ** 2 * (1 - phi ** 2)

    A = np.zeros((NS, NS))
    B = np.zeros((NS, AX))
    Q = np.zeros((NS, NS))
    for i in range(AX):
        p, v, w = i, AX + i, 2 * AX + i
        A[p, p], A[p, v] = 1.0, dt
        A[v, v], A[v, w] = 1.0 - k_drag * dt, k_drag * dt
        A[w, w] = phi
        B[p, i], B[v, i] = 0.5 * dt ** 2, dt
        # IMU accelerometer noise enters as control noise -> velocity/position
        Q[p, p] = (0.5 * dt ** 2 * sigma_imu) ** 2
        Q[v, v] = (dt * sigma_imu) ** 2
        Q[p, v] = Q[v, p] = (0.5 * dt ** 2 * sigma_imu) * (dt * sigma_imu)
        Q[w, w] = q_w
    return A, B, Q


# Measurement models. Each sensor sees a different part of the state, and none
# of them sees px or py - that is the whole point of the observability finding.
H_ALT = np.zeros((1, NS)); H_ALT[0, 2] = 1.0            # altimeter -> pz
H_FLOW = np.zeros((2, NS)); H_FLOW[0, 3] = H_FLOW[1, 4] = 1.0   # optical flow -> vx, vy
H_GPS = np.zeros((2, NS)); H_GPS[0, 0] = H_GPS[1, 1] = 1.0      # (extension) GPS -> px, py


def commanded_accel(t):
    """What the flight controller asks for: a climbing, turning path."""
    return np.array([
        1.2 * np.cos(0.35 * t),
        1.2 * np.sin(0.35 * t) + 0.4 * np.sin(0.11 * t),
        0.25 * np.sin(0.18 * t),
    ])


# ------------------------------------------------------------- simulation

def simulate(T=60.0, dt=0.01, seed=7, sigma_imu=0.15, sigma_alt=0.60,
             sigma_flow=0.12, f_alt=10, f_flow=30, f_gps=0, sigma_gps=2.5,
             **model_kw):
    """Generate ground truth and noisy, multi-rate sensor readings."""
    rng = np.random.default_rng(seed)
    A, B, Q = build_model(dt=dt, sigma_imu=sigma_imu, **model_kw)
    n = int(T / dt)
    t = np.arange(n) * dt

    phi = A[2 * AX, 2 * AX]
    q_w = Q[2 * AX, 2 * AX]

    X = np.zeros((n, NS))
    X[0, 2] = 5.0                       # start 5 m up
    X[0, 2 * AX:] = rng.normal(0, np.sqrt(q_w / (1 - phi ** 2)), AX)

    u_true = np.zeros((n, AX))
    imu = np.zeros((n, AX))
    for i in range(n - 1):
        a = commanded_accel(t[i])
        u_true[i] = a
        x = X[i]
        p, v, w = x[:AX], x[AX:2 * AX], x[2 * AX:]
        X[i + 1, :AX] = p + v * dt
        X[i + 1, AX:2 * AX] = v + (a - 0.8 * (v - w)) * dt
        X[i + 1, 2 * AX:] = phi * w + rng.normal(0, np.sqrt(q_w), AX)
    u_true[-1] = commanded_accel(t[-1])
    # the IMU reports commanded acceleration corrupted by noise: it is the INPUT u
    imu = u_true + rng.normal(0, sigma_imu, (n, AX))

    # multi-rate sensors; None where that sensor did not fire on this step
    step_alt = max(1, int(round(1 / (f_alt * dt))))
    step_flow = max(1, int(round(1 / (f_flow * dt))))
    step_gps = max(1, int(round(1 / (f_gps * dt)))) if f_gps else 0

    z_alt = [None] * n; z_flow = [None] * n; z_gps = [None] * n
    for i in range(n):
        if i % step_alt == 0:
            z_alt[i] = X[i, 2] + rng.normal(0, sigma_alt)
        if i % step_flow == 0:
            # optical flow gets noisier the higher you fly (fewer ground features)
            s = sigma_flow * (1 + 0.06 * max(0.0, X[i, 2] - 5.0))
            z_flow[i] = X[i, AX:AX + 2] + rng.normal(0, s, 2)
        if step_gps and i % step_gps == 0:
            z_gps[i] = X[i, :2] + rng.normal(0, sigma_gps, 2)

    return dict(t=t, X=X, imu=imu, u_true=u_true, z_alt=z_alt, z_flow=z_flow,
                z_gps=z_gps, A=A, B=B, Q=Q, dt=dt,
                sigma_alt=sigma_alt, sigma_flow=sigma_flow, sigma_gps=sigma_gps)


# ------------------------------------------------------- Kalman filtering

def flow_scale(alt):
    """Optical flow sees fewer ground features the higher you are."""
    return 1.0 + 0.06 * max(0.0, alt - 5.0)


def kalman_filter(sim, R_alt=None, R_flow=None, R_gps=None, Q=None,
                  use_gps=False, collect=True, adaptive_flow=False):
    """
    Forward pass. This is the forward (alpha) recursion of Unit 4, written for
    continuous Gaussian states: predict = sum over previous states, update = Bayes.
    """
    A, B = sim["A"], sim["B"]
    Q = sim["Q"] if Q is None else Q
    n = len(sim["t"])
    R_alt = np.array([[sim["sigma_alt"] ** 2]]) if R_alt is None else R_alt
    R_flow = np.eye(2) * sim["sigma_flow"] ** 2 if R_flow is None else R_flow
    R_gps = np.eye(2) * sim["sigma_gps"] ** 2 if R_gps is None else R_gps

    x = np.zeros(NS); x[2] = 5.0
    P = np.eye(NS) * 1.0
    P[2 * AX:, 2 * AX:] = np.eye(AX) * 2.0 ** 2

    xf = np.zeros((n, NS)); Pf = np.zeros((n, NS, NS))
    xp = np.zeros((n, NS)); Pp = np.zeros((n, NS, NS))
    loglik = 0.0
    gains, nis = [], []

    for i in range(n):
        if i > 0:                                   # ---- predict
            x = A @ x + B @ sim["imu"][i - 1]
            P = A @ P @ A.T + Q
        xp[i], Pp[i] = x, P

        for z, H, R, tag in ((sim["z_alt"][i], H_ALT, R_alt, "alt"),
                             (sim["z_flow"][i], H_FLOW, R_flow, "flow"),
                             (sim["z_gps"][i] if use_gps else None, H_GPS, R_gps, "gps")):
            if z is None:
                continue
            z = np.atleast_1d(z)
            if tag == "flow" and adaptive_flow:
                # scale R by the CURRENT ESTIMATED altitude - we never use truth
                R = R_flow * flow_scale(x[2]) ** 2
            y = z - H @ x                           # innovation
            S = H @ P @ H.T + R                     # innovation covariance
            K = P @ H.T @ np.linalg.inv(S)          # Kalman gain
            x = x + K @ y
            P = (np.eye(NS) - K @ H) @ P
            P = 0.5 * (P + P.T)                     # keep it symmetric
            # log-likelihood: the continuous analogue of Unit 4's scaling factors
            sign, logdet = np.linalg.slogdet(2 * np.pi * S)
            loglik += -0.5 * (logdet + y @ np.linalg.inv(S) @ y)
            if collect:
                if tag == "alt":
                    gains.append(K[2, 0])
                nis.append((i, tag, float(y @ np.linalg.inv(S) @ y), len(z)))
        xf[i], Pf[i] = x, P

    return dict(xf=xf, Pf=Pf, xp=xp, Pp=Pp, loglik=loglik,
                gains=np.array(gains), nis=nis, Q=Q,
                R_alt=R_alt, R_flow=R_flow)


def rts_smoother(sim, filt):
    """
    Backward pass: the Rauch-Tung-Striebel smoother. This is exactly the
    backward (beta) half of forward-backward, for continuous Gaussian states.
    """
    A = sim["A"]
    xf, Pf, xp, Pp = filt["xf"], filt["Pf"], filt["xp"], filt["Pp"]
    n = len(xf)
    xs = xf.copy(); Ps = Pf.copy()
    Js = np.zeros((n, NS, NS))
    for i in range(n - 2, -1, -1):
        J = Pf[i] @ A.T @ np.linalg.inv(Pp[i + 1])
        Js[i] = J
        xs[i] = xf[i] + J @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + J @ (Ps[i + 1] - Pp[i + 1]) @ J.T
        Ps[i] = 0.5 * (Ps[i] + Ps[i].T)
    return dict(xs=xs, Ps=Ps, J=Js)


# ---------------------------------------------------------------- EM (M-step)

def em_learn_noise(sim, iters=12, verbose=True):
    """
    EM for the unknown measurement noises, exactly the Unit 4 recipe:
      E-step: run filter + smoother to get the posterior over hidden states.
      M-step: set each variance to the mean squared smoothed residual.
    We start deliberately wrong (10x too large) and watch it find the truth.
    """
    R_alt = np.array([[(sim["sigma_alt"] * 3.0) ** 2]])
    R_flow = np.eye(2) * (sim["sigma_flow"] * 3.0) ** 2
    hist = []
    for it in range(iters):
        filt = kalman_filter(sim, R_alt=R_alt, R_flow=R_flow, collect=False)
        sm = rts_smoother(sim, filt)
        xs = sm["xs"]; Ps = sm["Ps"]

        # M-step: E[(z - Hx)^2] = residual^2 + H P H^T
        num_a, den_a = 0.0, 0
        num_f, den_f = 0.0, 0
        for i in range(len(xs)):
            if sim["z_alt"][i] is not None:
                r = sim["z_alt"][i] - H_ALT @ xs[i]
                num_a += float(np.squeeze(r) ** 2 + np.squeeze(H_ALT @ Ps[i] @ H_ALT.T)); den_a += 1
            if sim["z_flow"][i] is not None:
                r = np.atleast_1d(sim["z_flow"][i]) - H_FLOW @ xs[i]
                num_f += float(r @ r + np.trace(H_FLOW @ Ps[i] @ H_FLOW.T)); den_f += 2
        R_alt = np.array([[num_a / den_a]])
        R_flow = np.eye(2) * (num_f / den_f)
        hist.append((it, filt["loglik"], float(np.sqrt(R_alt[0, 0])),
                     float(np.sqrt(R_flow[0, 0]))))
        if verbose:
            print(f"   EM {it:2d}:  loglik={filt['loglik']:12.1f}   "
                  f"sigma_alt={np.sqrt(R_alt[0,0]):.3f} (true {sim['sigma_alt']:.3f})   "
                  f"sigma_flow={np.sqrt(R_flow[0,0]):.3f} (true {sim['sigma_flow']:.3f})")
    return R_alt, R_flow, hist


def dead_reckon(sim):
    """IMU-only integration: no filtering. The baseline we must beat."""
    A, B = sim["A"], sim["B"]
    n = len(sim["t"])
    x = np.zeros(NS); x[2] = 5.0
    out = np.zeros((n, NS))
    for i in range(n):
        if i > 0:
            x = A @ x + B @ sim["imu"][i - 1]
        out[i] = x
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
