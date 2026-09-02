"""Build deterministic, offline teaching fixtures for six structured domains.

The fixtures are generated from documented equations or geometric prototypes.
They are intentionally small and are not scientific benchmarks.
"""

import argparse
import json
from pathlib import Path

import numpy as np


SEED = 20260830
N_SAMPLES = 96
TRAIN_SAMPLES = 80


def split(values):
    return values[:TRAIN_SAMPLES], values[TRAIN_SAMPLES:]


def write_fixture(root, domain, arrays, meta):
    domain_dir = root / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(domain_dir / "fixture.npz", **arrays)
    payload = {
        "mode": "structured",
        "teaching_only": True,
        "generator": "scripts/build_structured_fixtures.py",
        "seed": SEED,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": N_SAMPLES - TRAIN_SAMPLES,
        **meta,
    }
    (domain_dir / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def weather_fixture(rng):
    axis = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values, labels = [], []
    for _ in range(N_SAMPLES):
        x0, y0 = rng.uniform(-0.45, 0.45, size=2)
        vx, vy = rng.uniform(-0.12, 0.12, size=2)
        sigma = rng.uniform(0.18, 0.35)
        amplitude = rng.uniform(0.7, 1.3)
        frames = []
        for time in range(4):
            cx, cy = x0 + time * vx, y0 + time * vy
            frame = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
            frame += rng.normal(0.0, 0.01, frame.shape)
            frames.append(frame.astype(np.float32))
        values.append(frames)
        labels.append(float(np.hypot(vx, vy)))
    x_train, x_val = split(np.asarray(values, dtype=np.float32))
    y_train, y_val = split(np.asarray(labels, dtype=np.float32))
    return {"train_x": x_train, "val_x": x_val, "train_y": y_train, "val_y": y_val}


def image_fixture(rng):
    axis = np.arange(16, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values, labels = [], []
    for _ in range(N_SAMPLES):
        count = int(rng.integers(1, 5))
        image = np.zeros((16, 16), dtype=np.float32)
        for _ in range(count):
            cx, cy = rng.uniform(2.0, 13.0, size=2)
            sigma = rng.uniform(0.7, 1.4)
            flux = rng.uniform(0.7, 1.4)
            image += flux * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        image += rng.normal(0.0, 0.02, image.shape)
        values.append(image[None, ...])
        labels.append(float(count))
    x_train, x_val = split(np.asarray(values, dtype=np.float32))
    y_train, y_val = split(np.asarray(labels, dtype=np.float32))
    return {"train_x": x_train, "val_x": x_val, "train_y": y_train, "val_y": y_val}


def spectrum_fixture(rng):
    wavelength = np.linspace(350e-9, 1000e-9, 128, dtype=np.float64)
    c2 = 1.438776877e-2
    values, labels = [], []
    for _ in range(N_SAMPLES):
        temperature = float(rng.uniform(3500.0, 8500.0))
        exponent = np.clip(c2 / (wavelength * temperature), 1e-5, 80.0)
        continuum = 1.0 / (wavelength**5 * np.expm1(exponent))
        continuum /= continuum.max()
        spectrum = continuum.copy()
        for center, width in ((430e-9, 8e-9), (517e-9, 10e-9), (656e-9, 7e-9)):
            depth = 0.08 + 0.12 * (temperature - 3500.0) / 5000.0
            spectrum *= 1.0 - depth * np.exp(-0.5 * ((wavelength - center) / width) ** 2)
        spectrum += rng.normal(0.0, 0.008, spectrum.shape)
        values.append(spectrum.astype(np.float32)[None, :])
        labels.append(temperature)
    x_train, x_val = split(np.asarray(values, dtype=np.float32))
    y_train, y_val = split(np.asarray(labels, dtype=np.float32))
    return {"train_x": x_train, "val_x": x_val, "train_y": y_train, "val_y": y_val}


def field_fixture(rng):
    position = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    values, labels = [], []
    for _ in range(N_SAMPLES):
        diffusivity = float(rng.uniform(0.02, 0.15))
        coefficients = rng.normal(0.0, 1.0, size=3)
        frames = []
        for time in range(4):
            field = sum(
                coefficients[k - 1]
                * np.exp(-diffusivity * (k**2) * time)
                * np.sin(k * position)
                for k in (1, 2, 3)
            )
            field += rng.normal(0.0, 0.01, field.shape)
            frames.append(field.astype(np.float32))
        values.append(frames)
        labels.append(diffusivity)
    x_train, x_val = split(np.asarray(values, dtype=np.float32))
    y_train, y_val = split(np.asarray(labels, dtype=np.float32))
    return {"train_x": x_train, "val_x": x_val, "train_y": y_train, "val_y": y_val}


def random_rotation(rng):
    matrix = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def structure_fixture(rng):
    theta = np.linspace(0.0, 4.0 * np.pi, 16, dtype=np.float32)
    values, labels = [], []
    for _ in range(N_SAMPLES):
        radius = float(rng.uniform(0.8, 1.3))
        pitch = float(rng.uniform(0.5, 2.0))
        points = np.stack(
            [radius * np.cos(theta), radius * np.sin(theta), pitch * theta / (2.0 * np.pi)],
            axis=1,
        )
        points = points @ random_rotation(rng).T + rng.uniform(-3.0, 3.0, size=(1, 3))
        points += rng.normal(0.0, 0.01, points.shape)
        values.append(points.astype(np.float32))
        labels.append(pitch)
    x_train, x_val = split(np.asarray(values, dtype=np.float32))
    y_train, y_val = split(np.asarray(labels, dtype=np.float32))
    return {"train_x": x_train, "val_x": x_val, "train_y": y_train, "val_y": y_val}


def crystal_fixture(rng):
    prototypes = [
        (np.array([11, 17]), np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])),
        (np.array([6, 6]), np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])),
        (np.array([14, 14, 8]), np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])),
        (np.array([29, 29, 8, 8]), np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]])),
    ]
    masses = {6: 12.011, 8: 15.999, 11: 22.990, 14: 28.085, 17: 35.45, 29: 63.546}
    atomic_numbers = np.zeros((N_SAMPLES, 8), dtype=np.int64)
    fractional = np.zeros((N_SAMPLES, 8, 3), dtype=np.float32)
    masks = np.zeros((N_SAMPLES, 8), dtype=bool)
    lattices = np.zeros((N_SAMPLES, 3, 3), dtype=np.float32)
    labels = np.zeros(N_SAMPLES, dtype=np.float32)
    for index in range(N_SAMPLES):
        numbers, coordinates = prototypes[index % len(prototypes)]
        count = len(numbers)
        length = float(rng.uniform(3.0, 6.0))
        atomic_numbers[index, :count] = numbers
        fractional[index, :count] = (coordinates + rng.normal(0.0, 0.005, coordinates.shape)) % 1.0
        masks[index, :count] = True
        lattices[index] = np.eye(3, dtype=np.float32) * length
        labels[index] = sum(masses[int(number)] for number in numbers) / length**3
    return {
        "train_atomic_numbers": atomic_numbers[:TRAIN_SAMPLES],
        "val_atomic_numbers": atomic_numbers[TRAIN_SAMPLES:],
        "train_fractional": fractional[:TRAIN_SAMPLES],
        "val_fractional": fractional[TRAIN_SAMPLES:],
        "train_mask": masks[:TRAIN_SAMPLES],
        "val_mask": masks[TRAIN_SAMPLES:],
        "train_lattice": lattices[:TRAIN_SAMPLES],
        "val_lattice": lattices[TRAIN_SAMPLES:],
        "train_y": labels[:TRAIN_SAMPLES],
        "val_y": labels[TRAIN_SAMPLES:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
    )
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    rng = np.random.default_rng(SEED)
    specs = {
        "weather": (
            weather_fixture(rng),
            {
                "representation": "spatiotemporal_patches",
                "pretraining": "masked patch reconstruction",
                "task_name": "advection speed teaching regression",
                "task_type": "regression",
                "sample_unit": "four 16x16 scalar-field frames",
                "target_unit": "grid units per frame",
                "patch_size": 4,
            },
        ),
        "crystal": (
            crystal_fixture(rng),
            {
                "representation": "periodic_graph",
                "pretraining": "masked atomic-number reconstruction",
                "task_name": "unit-cell mass density proxy regression",
                "task_type": "regression",
                "sample_unit": "one periodic crystal cell",
                "target_unit": "atomic-mass proxy per cubic angstrom",
            },
        ),
        "structure3d": (
            structure_fixture(rng),
            {
                "representation": "pairwise_distance_tokens",
                "pretraining": "masked distance-row reconstruction",
                "task_name": "helix pitch teaching regression",
                "task_type": "regression",
                "sample_unit": "one 16-point three-dimensional backbone",
                "target_unit": "coordinate units per turn",
                "patch_size": 1,
            },
        ),
        "image": (
            image_fixture(rng),
            {
                "representation": "image_patches",
                "pretraining": "masked image-patch reconstruction",
                "task_name": "astronomical source-count teaching regression",
                "task_type": "regression",
                "sample_unit": "one 16x16 single-band image",
                "target_unit": "sources",
                "patch_size": 4,
            },
        ),
        "spectrum": (
            spectrum_fixture(rng),
            {
                "representation": "wavelength_patches",
                "pretraining": "masked wavelength-patch reconstruction",
                "task_name": "blackbody temperature teaching regression",
                "task_type": "regression",
                "sample_unit": "one 128-bin normalized spectrum",
                "target_unit": "kelvin",
                "patch_size": 8,
            },
        ),
        "field": (
            field_fixture(rng),
            {
                "representation": "space_time_patches",
                "pretraining": "masked space-time patch reconstruction",
                "task_name": "diffusion coefficient teaching regression",
                "task_type": "regression",
                "sample_unit": "four 64-point one-dimensional field states",
                "target_unit": "inverse time units",
                "patch_size": 8,
            },
        ),
    }
    for domain, (arrays, meta) in specs.items():
        write_fixture(root, domain, arrays, meta)
        print(f"built {domain}: {root / domain / 'fixture.npz'}")


if __name__ == "__main__":
    main()
