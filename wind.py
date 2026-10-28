"""Real measured wind from the Open-Meteo archive.

Second real-data source in this project. The PX4 log (core.py) gives us the real IMU
noise; this gives us real turbulence. Hourly mean wind and gusts for any location and
date, free and without an API key. The gust factor sets the turbulence intensity that
drives the Ornstein-Uhlenbeck wind state.
"""
import json, urllib.request

import numpy as np

ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date={d0}&end_date={d1}&hourly=wind_speed_10m,wind_gusts_10m,"
           "wind_direction_10m&timezone=auto")


def fetch_real_wind(lat=11.70, lon=79.00, d0="2026-10-20", d1="2026-10-26", timeout=30):
    """Hourly measured wind. Returns dict with speed/gust (m/s) and direction (deg)."""
    url = ARCHIVE.format(lat=lat, lon=lon, d0=d0, d1=d1)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        j = json.load(r)
    h = j["hourly"]
    kmh = 1 / 3.6
    return dict(time=h["time"],
                speed=np.array(h["wind_speed_10m"], float) * kmh,
                gust=np.array(h["wind_gusts_10m"], float) * kmh,
                direction=np.array(h["wind_direction_10m"], float),
                lat=j["latitude"], lon=j["longitude"])


def wind_profile_from_real(wind, hour_index, dt, n, seed=0):
    """
    Turn one real hour into a flight-length turbulence profile.
    Mean wind sets the steady component; the measured gust factor sets turbulence
    intensity, following the usual sigma ~ (gust - mean)/2.5 rule of thumb.
    """
    rng = np.random.default_rng(seed)
    mean = float(wind["speed"][hour_index])
    gust = float(wind["gust"][hour_index])
    sigma = max(0.15, (gust - mean) / 2.5)
    theta = np.deg2rad(float(wind["direction"][hour_index]))
    # meteorological direction = where it blows FROM
    unit = np.array([-np.sin(theta), -np.cos(theta), 0.0])
    return dict(mean_vec=mean * unit, sigma=sigma, mean=mean, gust=gust,
                direction=float(wind["direction"][hour_index]))
