"""
Real relational store for the drone estimation pipeline (SQLite).

The database is not decoration: the simulator WRITES the sensor stream into it,
the Kalman filter READS its measurements back out with SQL, and the estimates and
per-run metrics are written back. Nothing is passed between stages in memory.

    runs            one row per experiment (parameters + headline metrics)
    ground_truth    the true 9-D state, for evaluation only
    sensor_readings the multi-rate, noisy sensor stream  <- the filter's only input
    estimates       filtered and smoothed state + its variance
"""
import sqlite3, json, numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs(
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created     TEXT NOT NULL,
    params      TEXT NOT NULL,
    pos_rmse_dr REAL, pos_rmse_kf REAL, pos_rmse_rts REAL,
    vel_rmse_kf REAL, wind_rmse_kf REAL, loglik REAL,
    nees REAL, nis_alt REAL, nis_flow REAL, notes TEXT);

CREATE TABLE IF NOT EXISTS ground_truth(
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL,
    px REAL, py REAL, pz REAL, vx REAL, vy REAL, vz REAL,
    wx REAL, wy REAL, wz REAL,
    PRIMARY KEY(run_id, step));

CREATE TABLE IF NOT EXISTS sensor_readings(
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL,
    sensor TEXT NOT NULL,            -- 'imu' | 'altimeter' | 'optical_flow' | 'gps'
    c0 REAL, c1 REAL, c2 REAL,       -- up to 3 channels
    sigma REAL,                      -- the sensor's stated 1-sigma at this instant
    PRIMARY KEY(run_id, step, sensor));
CREATE INDEX IF NOT EXISTS idx_sensor ON sensor_readings(run_id, sensor, step);

CREATE TABLE IF NOT EXISTS imu_calibration(
    source      TEXT PRIMARY KEY,   -- URL the log was fetched from
    fetched     TEXT NOT NULL,
    rate_hz     REAL, duration_s REAL, n_samples INTEGER,
    acc_sigma_x REAL, acc_sigma_y REAL, acc_sigma_z REAL,
    gyr_sigma_x REAL, gyr_sigma_y REAL, gyr_sigma_z REAL,
    acc_bias_x  REAL, acc_bias_y  REAL, acc_bias_z  REAL,
    vrw         REAL);

CREATE TABLE IF NOT EXISTS imu_raw(          -- real samples, straight off the aircraft
    source TEXT NOT NULL, n INTEGER NOT NULL, t REAL,
    ax REAL, ay REAL, az REAL, gx REAL, gy REAL, gz REAL,
    PRIMARY KEY(source, n));

CREATE TABLE IF NOT EXISTS estimates(
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL,
    kind TEXT NOT NULL,              -- 'filter' | 'smoother' | 'dead_reckon'
    px REAL, py REAL, pz REAL, vx REAL, vy REAL, vz REAL,
    wx REAL, wy REAL, wz REAL,
    var_px REAL, var_py REAL, var_pz REAL,
    PRIMARY KEY(run_id, step, kind));
"""


def connect(path="drone.db"):
    con = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def new_run(con, params, notes=""):
    cur = con.execute(
        "INSERT INTO runs(created, params, notes) VALUES(datetime('now'), ?, ?)",
        (json.dumps(params), notes))
    return cur.lastrowid


def write_truth(con, run_id, t, X):
    con.executemany(
        "INSERT OR REPLACE INTO ground_truth VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        [(run_id, i, float(t[i]), *[float(v) for v in X[i]]) for i in range(len(X))])


def write_sensors(con, run_id, t, imu, z_alt, z_flow, z_gps, s_alt, s_flow_list, s_gps):
    rows = []
    for i in range(len(t)):
        rows.append((run_id, i, float(t[i]), "imu",
                     float(imu[i][0]), float(imu[i][1]), float(imu[i][2]), None))
        if z_alt[i] is not None:
            rows.append((run_id, i, float(t[i]), "altimeter",
                         float(z_alt[i]), None, None, s_alt))
        if z_flow[i] is not None:
            rows.append((run_id, i, float(t[i]), "optical_flow",
                         float(z_flow[i][0]), float(z_flow[i][1]), None,
                         float(s_flow_list[i])))
        if z_gps[i] is not None:
            rows.append((run_id, i, float(t[i]), "gps",
                         float(z_gps[i][0]), float(z_gps[i][1]), None, s_gps))
    con.executemany("INSERT OR REPLACE INTO sensor_readings VALUES(?,?,?,?,?,?,?,?)", rows)


def read_sensor_stream(con, run_id, n):
    """
    The filter's ONLY input. Pull the whole multi-rate stream back out with SQL and
    re-assemble it into per-step slots. Steps with no reading simply have none.
    """
    imu = np.zeros((n, 3))
    z_alt = [None] * n
    z_flow = [None] * n
    z_gps = [None] * n
    s_flow = [None] * n
    for r in con.execute(
            "SELECT step, sensor, c0, c1, c2, sigma FROM sensor_readings "
            "WHERE run_id=? ORDER BY step", (run_id,)):
        i, s = r["step"], r["sensor"]
        if s == "imu":
            imu[i] = (r["c0"], r["c1"], r["c2"])
        elif s == "altimeter":
            z_alt[i] = r["c0"]
        elif s == "optical_flow":
            z_flow[i] = np.array([r["c0"], r["c1"]]); s_flow[i] = r["sigma"]
        elif s == "gps":
            z_gps[i] = np.array([r["c0"], r["c1"]])
    return imu, z_alt, z_flow, z_gps, s_flow


def write_estimates(con, run_id, t, kind, X, P=None):
    rows = []
    for i in range(len(X)):
        v = [float(x) for x in X[i]]
        if P is None:
            var = (None, None, None)
        else:
            var = tuple(float(P[i][j, j]) for j in range(3))
        rows.append((run_id, i, float(t[i]), kind, *v, *var))
    con.executemany(
        "INSERT OR REPLACE INTO estimates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def rmse_by_sql(con, run_id, kind):
    """Compute the error in the database itself, with a JOIN - no numpy."""
    q = """
    SELECT sqrt(avg((e.px-g.px)*(e.px-g.px)
                  + (e.py-g.py)*(e.py-g.py)
                  + (e.pz-g.pz)*(e.pz-g.pz))) AS pos_rmse,
           sqrt(avg((e.vx-g.vx)*(e.vx-g.vx)
                  + (e.vy-g.vy)*(e.vy-g.vy)
                  + (e.vz-g.vz)*(e.vz-g.vz))) AS vel_rmse
    FROM estimates e JOIN ground_truth g
      ON e.run_id=g.run_id AND e.step=g.step
    WHERE e.run_id=? AND e.kind=?"""
    r = con.execute(q, (run_id, kind)).fetchone()
    return r["pos_rmse"], r["vel_rmse"]


def finish_run(con, run_id, **metrics):
    sets = ", ".join(f"{k}=?" for k in metrics)
    con.execute(f"UPDATE runs SET {sets} WHERE run_id=?",
                (*metrics.values(), run_id))
