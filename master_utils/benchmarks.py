from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

def branin(x1: float, x2: float) -> float:
    """Branin-Hoo benchmark function.

    Common domain:
        x1 in [-5, 10]
        x2 in [0, 15]

    Global minimum:
        approximately 0.397887
    """
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)

    return float(
        a * (x2 - b * x1**2 + c * x1 - r) ** 2
        + s * (1 - t) * np.cos(x1)
        + s
    )


def hartmann(x: ArrayLike, dim: int = 6) -> float:
    """
    Evaluate the Hartmann function.

    The Hartmann function is a standard non-convex optimization
    benchmark function. It is usually evaluated on the unit hypercube:

        x_i in [0, 1]

    Supported dimensions are 3, 4, and 6.

    Approximate global minima
    --------------------------
    dim=3:
        x* = [0.1145889, 0.5556489, 0.8525470]
        f(x*) ≈ -3.862779787

    dim=4:
        x* = [0.1873953, 0.1941515, 0.5579178, 0.2647796]
        f(x*) ≈ -3.729840584

    dim=6:
        x* = [0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657300]
        f(x*) ≈ -3.322368011

    Parameters
    ----------
    x:
        Input point of shape (dim,). Values are usually in [0, 1].
    dim:
        Dimension of the Hartmann function. Supported values: 3, 4, 6.

    Returns
    -------
    float
        Hartmann function value.
    """

    x_arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)

    if dim == 3:
        alpha: NDArray[np.float64] = np.array(
            [1.0, 1.2, 3.0, 3.2],
            dtype=np.float64,
        )

        A: NDArray[np.float64] = np.array(
            [
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
            ],
            dtype=np.float64,
        )

        P: NDArray[np.float64] = 1e-4 * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ],
            dtype=np.float64,
        )

    elif dim == 4:
        alpha = np.array(
            [1.0, 1.2, 3.0, 3.2],
            dtype=np.float64,
        )

        A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5],
                [0.05, 10.0, 17.0, 0.1],
                [3.0, 3.5, 1.7, 10.0],
                [17.0, 8.0, 0.05, 10.0],
            ],
            dtype=np.float64,
        )

        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ],
            dtype=np.float64,
        )

    elif dim == 6:
        alpha = np.array(
            [1.0, 1.2, 3.0, 3.2],
            dtype=np.float64,
        )

        A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ],
            dtype=np.float64,
        )

        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ],
            dtype=np.float64,
        )

    else:
        raise ValueError("Only dim=3, dim=4, and dim=6 are supported.")

    if x_arr.shape != (dim,):
        raise ValueError(
            f"x must be a 1D array of length {dim}, got shape {x_arr.shape}."
        )

    inner_sums: NDArray[np.float64] = np.sum(A * (x_arr - P) ** 2, axis=1)
    value: np.float64 = -np.sum(alpha * np.exp(-inner_sums))

    return float(value)