"""A small orbit-fitting task abstracted from GD-1 stream modelling.

The stream track is approximated by the orbit of its progenitor, as in Koposov et al. (2010).
Seven parameters: the circular velocity and vertical flattening of a logarithmic halo potential,
and the progenitor's phase-space position at phi1 = 0 (phi2, distance, two proper motions and the
line-of-sight velocity). The data are the GD-1 track measurements of Koposov et al. (2010),
Tables 1-4, as transcribed in Bovy (2016, mwhalo-shape-2016, BSD licence): 16 positions,
6 distances, 5 x 2 proper motions and 27 radial velocities. Mock scenarios keep the same phi1
nodes and error bars and replace the measured values with a known truth plus Gaussian noise.
"""
import numpy as np

NAMES = ("vc", "q", "phi2", "dist", "pmphi1", "pmphi2", "vlos")
UNITS = ("km/s", "", "deg", "kpc", "mas/yr", "mas/yr", "km/s")
PRIOR = {"vc": (150.0, 300.0), "q": (0.6, 1.5), "phi2": (-2.0, 0.5), "dist": (7.0, 13.0),
         "pmphi1": (-12.0, -5.0), "pmphi2": (-5.0, 1.0), "vlos": (-400.0, -150.0)}
R0, V_PECULIAR = 8.0, np.array([11.1, 24.0, 7.25])  # Sun: kpc from the centre; (U, V, W) km/s, U towards the centre
KPC_PER_MYR = 1.0227e-3  # 1 km/s in kpc/Myr
K_PM = 4.74047  # km/s per (mas/yr * kpc)

# phi1, phi2, phi2 error (deg) | phi1, distance, error (kpc) | phi1, pmphi1, pmphi2, error (mas/yr) | phi1, vlos, error (km/s)
POSITION = np.array([[-60, -0.64, 0.15], [-56, -0.89, 0.27], [-54, -0.45, 0.15], [-48, -0.08, 0.13],
                     [-44, 0.01, 0.14], [-40, -0.00, 0.09], [-36, 0.04, 0.10], [-34, 0.06, 0.13],
                     [-32, 0.04, 0.06], [-30, 0.08, 0.10], [-28, 0.03, 0.12], [-24, 0.06, 0.05],
                     [-22, 0.06, 0.13], [-18, -0.05, 0.11], [-12, -0.29, 0.16], [-2, -0.87, 0.07]])
DISTANCE = np.array([[-55, 7.20, 0.30], [-45, 7.59, 0.40], [-35, 7.83, 0.30], [-25, 8.69, 0.40],
                     [-15, 8.91, 0.40], [0, 9.86, 0.50]])
PROPER = np.array([[-55, -13.60, -5.70, 1.30], [-45, -13.10, -3.30, 0.70], [-35, -12.20, -3.10, 1.00],
                   [-25, -12.60, -2.70, 1.40], [-15, -10.80, -2.80, 1.00]])
VELOCITY = np.array([[-45.23, 28.8, 6.9], [-43.17, 29.3, 10.2], [-39.54, 2.9, 8.7], [-39.25, -5.2, 6.5],
                     [-37.95, 1.1, 5.6], [-37.96, -11.7, 11.2], [-35.49, -50.4, 5.2], [-35.27, -30.9, 12.8],
                     [-34.92, -35.3, 7.5], [-34.74, -30.9, 9.2], [-33.74, -74.3, 9.8], [-32.90, -71.5, 9.6],
                     [-32.25, -71.5, 9.2], [-29.95, -92.7, 8.7], [-26.61, -114.2, 7.3], [-25.45, -67.8, 7.1],
                     [-24.86, -111.2, 17.8], [-21.21, -144.4, 10.5], [-14.47, -179.0, 10.0], [-13.73, -191.4, 7.5],
                     [-13.02, -162.9, 9.6], [-12.68, -217.2, 10.7], [-12.55, -172.2, 6.6], [-56.0, 122.0, 14.0],
                     [-47.0, 44.8, 10.0], [-28.0, -77.8, 6.0], [-24.0, -114.5, 9.0]])
GD1_DATA = {"phi2": POSITION, "dist": DISTANCE, "pmphi1": PROPER[:, [0, 1, 3]],
            "pmphi2": PROPER[:, [0, 2, 3]], "vlos": VELOCITY}

# Galactic -> ICRS (transpose of the Hipparcos ICRS -> Galactic matrix), then ICRS -> GD-1 (Koposov et al. 2010)
ICRS_TO_GALACTIC = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                             [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
                             [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])
ICRS_TO_GD1 = np.array([[-0.4776303088, -0.1738432154, 0.8611897727],
                        [0.510844589, -0.8524449229, 0.111245042],
                        [0.7147776536, 0.4930681392, 0.4959603976]])
GALACTIC_TO_GD1 = ICRS_TO_GD1 @ ICRS_TO_GALACTIC.T


def _observe(position, velocity, vc):
    """Galactocentric position (kpc) and velocity (km/s), (..., 3) -> phi1, phi2, dist, pmphi1, pmphi2, vlos."""
    sun_velocity = V_PECULIAR + np.array([0.0, 1.0, 0.0]) * vc[..., None]
    x = (position + np.array([R0, 0.0, 0.0])) @ GALACTIC_TO_GD1.T
    v = (velocity - sun_velocity) @ GALACTIC_TO_GD1.T
    dist = np.linalg.norm(x, axis=-1)
    phi1 = np.arctan2(x[..., 1], x[..., 0])
    phi2 = np.arcsin(x[..., 2] / dist)
    radial = np.stack([np.cos(phi2) * np.cos(phi1), np.cos(phi2) * np.sin(phi1), np.sin(phi2)], -1)
    along = np.stack([-np.sin(phi1), np.cos(phi1), np.zeros_like(phi1)], -1)
    across = np.stack([-np.sin(phi2) * np.cos(phi1), -np.sin(phi2) * np.sin(phi1), np.cos(phi2)], -1)
    return (np.degrees(phi1), np.degrees(phi2), dist, (v * along).sum(-1) / (K_PM * dist),
            (v * across).sum(-1) / (K_PM * dist), (v * radial).sum(-1))


def _start(theta):
    """Progenitor at phi1 = 0 -> Galactocentric position and velocity."""
    vc, phi2, dist, pm1, pm2, vlos = (theta[:, i] for i in (0, 2, 3, 4, 5, 6))
    phi2 = np.radians(phi2)
    radial = np.stack([np.cos(phi2), np.zeros_like(phi2), np.sin(phi2)], -1)
    along = np.stack([np.zeros_like(phi2), np.ones_like(phi2), np.zeros_like(phi2)], -1)
    across = np.stack([-np.sin(phi2), np.zeros_like(phi2), np.cos(phi2)], -1)
    x = dist[:, None] * radial
    v = vlos[:, None] * radial + K_PM * dist[:, None] * (pm1[:, None] * along + pm2[:, None] * across)
    sun_velocity = V_PECULIAR + np.array([0.0, 1.0, 0.0]) * vc[:, None]
    return x @ GALACTIC_TO_GD1 - np.array([R0, 0.0, 0.0]), v @ GALACTIC_TO_GD1 + sun_velocity


def _integrate(position, velocity, vc, q, dt, steps):
    """Leapfrog in the potential (vc^2 / 2) ln(x^2 + y^2 + z^2 / q^2); returns (steps + 1, n, 3) arrays."""
    scale = np.array([1.0, 1.0, 0.0])[None, :] + np.array([0.0, 0.0, 1.0])[None, :] / q[:, None] ** 2

    def accel(x):
        return -(vc ** 2)[:, None] * x * scale / (x * x * scale).sum(-1, keepdims=True) * KPC_PER_MYR

    xs, vs = [position], [velocity]
    x, v = position.copy(), velocity.copy()
    for _ in range(steps):
        v = v + 0.5 * dt * accel(x)
        x = x + dt * v * KPC_PER_MYR
        v = v + 0.5 * dt * accel(x)
        xs.append(x)
        vs.append(v)
    return np.array(xs), np.array(vs)


def track(theta, dt=0.1, steps=700):
    """Orbit track through the progenitor, integrated 70 Myr each way; returns (n, 2 * steps + 1, 6)."""
    theta = np.atleast_2d(np.asarray(theta, float))
    x, v = _start(theta)
    vc, q = theta[:, 0], theta[:, 1]
    xf, vf = _integrate(x, v, vc, q, dt, steps)
    xb, vb = _integrate(x, v, vc, q, -dt, steps)
    xs = np.concatenate([xb[::-1], xf[1:]]).transpose(1, 0, 2)
    vs = np.concatenate([vb[::-1], vf[1:]]).transpose(1, 0, 2)
    return np.stack(_observe(xs, vs, vc[:, None]), -1)


def segment(t):
    """The part of one track where phi1 changes monotonically through the progenitor, sorted by phi1;
    None unless it covers phi1 from -62 to +2 degrees, the span of the data."""
    centre = len(t) // 2
    step = np.sign(np.diff(t[:, 0]))
    direction = step[centre]
    lo, hi = centre, centre
    while lo > 0 and step[lo - 1] == direction:
        lo -= 1
    while hi < len(step) and step[hi] == direction:
        hi += 1
    part = t[lo:hi + 1] if direction > 0 else t[lo:hi + 1][::-1]
    if part[0, 0] > -62 or part[-1, 0] < 2 or not np.isfinite(part).all():
        return None
    return part


def chi2(theta, data=GD1_DATA):
    """Chi-square per observable (phi2, dist, pmphi1, pmphi2, vlos) for each parameter vector; an orbit
    without a usable segment gets 1e6 on every observable."""
    theta = np.atleast_2d(np.asarray(theta, float))
    out = np.full((len(theta), 5), 1e6)
    for row, t in enumerate(track(theta)):
        part = segment(t)
        if part is None:
            continue
        for column, name in enumerate(("phi2", "dist", "pmphi1", "pmphi2", "vlos")):
            table = data[name]
            model = np.interp(table[:, 0], part[:, 0], part[:, column + 1])
            out[row, column] = (((model - table[:, 1]) / table[:, 2]) ** 2).sum()
    return out


def mock(truth, seed):
    """Mock data with the GD-1 phi1 nodes and error bars, values from the truth plus Gaussian noise;
    None if the truth has no usable track."""
    rng = np.random.default_rng(seed)
    part = segment(track(np.asarray(truth, float)[None])[0])
    if part is None:
        return None
    data = {}
    for column, name in enumerate(("phi2", "dist", "pmphi1", "pmphi2", "vlos")):
        table = GD1_DATA[name].copy()
        table[:, 1] = np.interp(table[:, 0], part[:, 0], part[:, column + 1]) + rng.normal(0, table[:, 2])
        data[name] = table
    return data
