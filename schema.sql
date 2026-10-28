CREATE TABLE runs(
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created     TEXT NOT NULL,
    params      TEXT NOT NULL,
    pos_rmse_dr REAL, pos_rmse_kf REAL, pos_rmse_rts REAL,
    vel_rmse_kf REAL, wind_rmse_kf REAL, loglik REAL,
    nees REAL, nis_alt REAL, nis_flow REAL, notes TEXT);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE ground_truth(
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL,
    px REAL, py REAL, pz REAL, vx REAL, vy REAL, vz REAL,
    wx REAL, wy REAL, wz REAL,
    PRIMARY KEY(run_id, step));
CREATE TABLE sensor_readings(
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL,
    sensor TEXT NOT NULL,            -- 'imu' | 'altimeter' | 'optical_flow' | 'gps'
    c0 REAL, c1 REAL, c2 REAL,       -- up to 3 channels
    sigma REAL,                      -- the sensor's stated 1-sigma at this instant
    PRIMARY KEY(run_id, step, sensor));
CREATE INDEX idx_sensor ON sensor_readings(run_id, sensor, step);
CREATE TABLE imu_calibration(
    source      TEXT PRIMARY KEY,   -- URL the log was fetched from
    fetched     TEXT NOT NULL,
    rate_hz     REAL, duration_s REAL, n_samples INTEGER,
    acc_sigma_x REAL, acc_sigma_y REAL, acc_sigma_z REAL,
    gyr_sigma_x REAL, gyr_sigma_y REAL, gyr_sigma_z REAL,
    acc_bias_x  REAL, acc_bias_y  REAL, acc_bias_z  REAL,
    vrw         REAL);
CREATE TABLE imu_raw(          -- real samples, straight off the aircraft
    source TEXT NOT NULL, n INTEGER NOT NULL, t REAL,
    ax REAL, ay REAL, az REAL, gx REAL, gy REAL, gz REAL,
    PRIMARY KEY(source, n));
CREATE TABLE estimates(
    run_id INTEGER NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL,
    kind TEXT NOT NULL,              -- 'filter' | 'smoother' | 'dead_reckon'
    px REAL, py REAL, pz REAL, vx REAL, vy REAL, vz REAL,
    wx REAL, wy REAL, wz REAL,
    var_px REAL, var_py REAL, var_pz REAL,
    PRIMARY KEY(run_id, step, kind));