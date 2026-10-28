# Drone 3-D State Estimation in Turbulent Wind

**Linear Dynamical Systems + Kalman Filtering** — Machine Learning, Unit 4 (Sequential Data)

**Team Cash Guard** · V. Anbu Chelvan · S. Roopesh · B. Charith Reddy · Y. Rupesh Reddy

> Recover a drone's true position, velocity **and the wind acting on it** from three sensors that
> are each individually inadequate — and from a wind that nothing on board measures.
>
> **Result: 238× more accurate than integrating the IMU.** Position error falls from 103.4 m to
> 0.43 m, and the wind is recovered to 2.14 m/s without ever being observed.

---

## Quick start

```
Colab → File → Upload notebook → Drone_Kalman_Unit4.ipynb → Runtime → Run all
```

Runs in about a minute. It installs `pyulog` itself and downloads the flight log it needs.
No API key, no account, nothing to configure.

---

## The problem

| Sensor | Rate | Measures | Weakness |
|---|---|---|---|
| IMU | 100 Hz | acceleration | integrating twice turns small bias into huge error — **drifts 103 m in 60 s** |
| Altimeter | 10 Hz | height only | σ = 0.60 m, and says nothing about horizontal position |
| Optical flow | 30 Hz | horizontal velocity | σ = 0.12 m/s, and **degrades with altitude** |

On top of this, turbulent wind pushes the aircraft around. Gusts are not white noise — they are
correlated in time, so a gust now predicts a gust a second from now.

## Why this is a Unit 4 problem

A Linear Dynamical System is an HMM whose hidden state is continuous. Everything from the
discrete case carries over:

| Concept | HMM (discrete) | LDS (this project) |
|---|---|---|
| Hidden state | Rainy / Sunny | position, velocity, wind ∈ ℝ⁹ |
| Forward pass | α recursion | **Kalman filter** — predict, then update |
| Forward–backward | γ = αβ / P(O) | **RTS smoother** |
| Underflow fix | scaling factors sₜ | log-likelihood accumulated from innovations |
| Unknown parameters | EM / Baum–Welch | **EM**, with the smoother in the E-step |
| Viterbi | best discrete path | the smoothed mean *is* the MAP path (Gaussian ⇒ unimodal) |

Because the state is continuous and every relationship is linear with Gaussian noise, the sum
over states becomes an integral with a closed form. The posterior stays Gaussian, so a mean and a
covariance are the whole answer — we never enumerate paths.

## The model

State is 9-dimensional: three identical per-axis blocks of `[position, velocity, wind]`.

```
p(t+1) = p(t) + v(t)·Δt
v(t+1) = v(t) + ( a − k·(v(t) − w(t)) )·Δt      ← drag acts on AIRSPEED (v − w)
w(t+1) = φ·w(t) + ε(t),        φ = exp(−Δt/τ)   ← Ornstein–Uhlenbeck gust model
```

Two decisions worth defending:

1. **Drag acts on airspeed `(v − w)`, not ground speed.** This is what couples wind into the
   dynamics and makes it *observable* — we infer the gust from how the drone is pushed off course.
   Write drag against ground speed instead and the wind becomes unrecoverable in principle.

2. **Gusts are an Ornstein–Uhlenbeck process, not white noise.** Real turbulence persists for
   seconds. Crucially this is **still linear**, so the Kalman filter remains exactly optimal —
   no EKF or UKF is needed. We gain realism without losing optimality.

## Real data, from two independent sources

This is not a project where every number was chosen by the authors.

**1 — Real IMU noise, from a real flight (`core.py`)**
A PX4 ULog from an actual quadrotor flight is downloaded and parsed with `pyulog`. We measure the
accelerometer noise from the log rather than assuming it:

```
rate         248 Hz over 69 s, 17 070 samples
accel sigma  [0.293 0.411 0.162] m/s²
accel bias   [1.091 −0.442 0.189] m/s²   ← real, uncorrected
sigma_imu    0.289 m/s²   (MEASURED, not assumed) → feeds Q
velocity random walk  0.01835 (m/s²)/√Hz
```

**2 — Real turbulence, from measured weather (`wind.py`)**
168 hourly wind and gust observations from the Open-Meteo archive. The flight runs in the
genuinely gustiest hour in the record — 21 Oct 2026, 14:00 at Kallakurichi: mean 2.44 m/s with
**9.69 m/s gusts**. The gust factor sets the turbulence intensity σ = 2.90 m/s driving the OU wind state — so
**neither the sensor noise nor the wind is a number we chose.**

```python
from wind import fetch_real_wind, wind_profile_from_real
w = fetch_real_wind(lat=11.70, lon=79.00, d0="2026-10-20", d1="2026-10-26")
```

## The database

Everything goes through SQLite — sensors, ground truth, estimates and metrics. Accuracy is
computed **by SQL JOIN inside the database**, not by numpy:

```sql
SELECT sqrt(avg((e.px-g.px)*(e.px-g.px) + (e.py-g.py)*(e.py-g.py) + (e.pz-g.pz)*(e.pz-g.pz)))
FROM estimates e JOIN ground_truth g ON e.run_id=g.run_id AND e.step=g.step
WHERE e.run_id=? AND e.kind=?
```

| table | rows | holds |
|---|---|---|
| `runs` | 1 | parameters and final metrics per experiment |
| `imu_calibration` | 1 | noise measured from the PX4 log |
| `imu_raw` | 4 268 | raw samples from the real flight |
| `ground_truth` | 6 000 | the true state, for scoring only |
| `sensor_readings` | 8 660 | every IMU / altimeter / flow / GPS reading |
| `estimates` | 18 000 | filter, smoother and dead-reckoning output with variances |

Schema in [`schema.sql`](schema.sql). The database itself is gitignored — the notebook rebuilds it.

## Results

60 s of flight, real measured turbulence:

| method | position RMSE | velocity RMSE | |
|---|---|---|---|
| IMU only (dead reckoning) | 103.394 m | 4.601 m/s | unusable |
| **Kalman filter** | **0.434 m** | **0.683 m/s** | real time |
| **RTS smoother** | **0.318 m** | **0.200 m/s** | after the flight |

**238× better than dead reckoning.** Wind recovered to **2.14 m/s RMSE** — though no sensor
measures wind. EM recovered the altimeter noise as 0.593 against a true 0.600, with the
log-likelihood rising monotonically at every iteration.

## Three findings

**1. A consistency test caught an error the RMSE could not.**
Optical flow gets noisier with altitude; our filter assumed constant noise. Position error looked
perfectly healthy — but NIS was **10.15 when it should be 2**, meaning the filter's claimed
uncertainty was about five times too small. It was accurate *and dishonest about being accurate*.

| R for optical flow | NIS (want 2) | position RMSE |
|---|---|---|
| constant (wrong) | 10.15 | 0.434 m |
| altitude-scaled | **1.95** | 0.434 m |

Scaling R with the *estimated* altitude fixed the consistency without moving the accuracy at all.
Looking only at RMSE, we would never have found it.

**2. The covariance admits what it cannot know.**
No sensor observes horizontal position, so σₓ and σᵧ grow without bound while σ_z stays pinned by
the altimeter. That is observability showing up in the mathematics — not a bug.

**3. Adding a sensor made things worse.**
A 2.5 m GPS *hurt* over a short flight, because optical flow was already better than that. It only
begins to help once dead-reckoned drift exceeds the GPS noise:

| flight | no GPS | with GPS | |
|---|---|---|---|
| 60 s | 0.434 m | 0.750 m | GPS hurts (+73%) |
| 120 s | 0.430 m | 0.604 m | GPS hurts (+41%) |
| 300 s | 1.558 m | 0.673 m | GPS helps (−57%) |
| 600 s | 2.105 m | 0.901 m | GPS helps (−57%) |

More sensors is not automatically better. Fusion helps only when the new sensor beats what the
estimator already has.

## Files

| file | what it is |
|---|---|
| `Drone_Kalman_Unit4.ipynb` | the whole project, runnable top to bottom in Colab |
| `core.py` | LDS model, PX4 log parsing, simulator, Kalman filter, RTS smoother, EM |
| `wind.py` | real measured wind and gusts from the Open-Meteo archive |
| `db.py` | SQLite layer — schema, writes, and the SQL-JOIN accuracy query |
| `schema.sql` | the database schema on its own |
| `figures/` | generated plots |
| `run_output.txt` | console output of a full run, for reference |

## Limitations

Stated plainly, because they matter:

- **Linear drag and small-angle flight.** Aggressive manoeuvres would need an EKF or UKF.
- **The IMU channel carries thrust acceleration only.** A real accelerometer also senses drag;
  we leave that out deliberately so the filter must *infer* wind rather than read it.
- **Wind is spatially uniform** across the airframe.
- **The flight is simulated.** What is real is the IMU noise (measured from a PX4 log) and the
  turbulence statistics (measured weather). The trajectory itself is generated.

## The lesson we would keep

**Accuracy and honesty are different properties of an estimator.** Our filter reported an
excellent RMSE while being significantly over-confident about its own uncertainty, and only a
consistency test revealed it. An estimator that does not know what it does not know is dangerous
in a way that no error metric will show you.

## Contributions

| Member | Contribution |
|---|---|
| V. Anbu Chelvan | LDS model formulation, wind-augmented state, Kalman filter + RTS smoother |
| S. Roopesh | Real PX4 log retrieval, IMU noise calibration, SQLite database |
| B. Charith Reddy | NEES/NIS consistency testing, GPS crossover study, observability analysis |
| Y. Rupesh Reddy | EM noise learning, figures, slide deck |

## References

- Bishop, *Pattern Recognition and Machine Learning*, ch. 13 — sequential data, HMMs and LDS.
- Rauch, Tung & Striebel (1965), "Maximum likelihood estimates of linear dynamic systems".
- Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation* — NEES and NIS.
- PX4 flight log, [logs.px4.io](https://logs.px4.io). Weather: [Open-Meteo archive](https://open-meteo.com).
