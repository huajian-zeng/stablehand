"""Shared validation for public CLI parameters and numerical configuration."""

import argparse
import math


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return value


def random_seed(value):
    value = int(value)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("must be an integer in [0, 2**32)")
    return value


def validate_positive_values(values, label):
    if not len(values) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError(f"{label} must contain only finite values greater than zero")


def sigma_vector(value):
    try:
        values = [float(part.strip()) for part in value.split(",")]
        validate_positive_values(values, "sigma")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return values
