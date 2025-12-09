import time
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular

"""
date:9 December 2025
use GPU for:
1.Matrix assembly
2.Per-step mat-vec multiplication
3.linear solve
4.Large-scale pointwise ops
"""

def european_call_price_1D_gpu(
    S0,  # current price
    K=10.0,  # strike price
    T=1.0,  # maturity
    r=0.10,  # risk-free interest rate
    sigma=0.20,  # volatility
    S_max_factor=4.0,  # Right-side spatial truncation multiple
    m=200,       # finite elements
    n=200,       # time steps
    use_float32=False,
):
    """
    GPU version using CuPy.
    Linear system is solved via:
    1) One-time Cholesky factorization: A = L L^T
    2) Each time step: two triangular solves (L y = b, L^T x = y)
    """
    dtype = cp.float32 if use_float32 else cp.float64

    t_start_total = time.time()

    # 1. preprocessing & variable transformation parameters
    S_max = S_max_factor * K
    S_min = 1e-6 * K

    # x = ln(S/K)
    x_min = float(np.log(S_min / K))
    x_max = float(np.log(S_max / K))

    # k1 = 2r / sigma^2
    k1 = 2.0 * r / (sigma ** 2)

    # tau = 0.5 * sigma^2 * (T - t), tau_max = tau(t=0)
    tau_max = 0.5 * sigma ** 2 * T

    # space and time mesh
    dx = (x_max - x_min) / m
    x = x_min + dx * cp.arange(m + 1, dtype=dtype)   # (m+1,)

    dtau = tau_max / n
    tau_grid = dtau * cp.arange(n + 1, dtype=dtype)

    # 2. initial condition v(x, 0)
    # v(x,0) = max( e^{0.5(k1+1)x} - e^{0.5(k1-1)x}, 0 )
    v0 = cp.maximum(
        cp.exp(0.5 * (k1 + 1.0) * x) - cp.exp(0.5 * (k1 - 1.0) * x),
        0.0,
    ).astype(dtype)

    # 3. assemble global mass matrix C and stiffness matrix K_mat on GPU
    t_assemble_start = time.time()

    C_e = (dx / 3.0) * cp.array([[1.0, 0.5],
                                 [0.5, 1.0]], dtype=dtype)
    K_e = (1.0 / dx) * cp.array([[1.0, -1.0],
                                 [-1.0, 1.0]], dtype=dtype)

    C = cp.zeros((m + 1, m + 1), dtype=dtype)
    K_mat = cp.zeros((m + 1, m + 1), dtype=dtype)

    # Matrix assembly (still Python loop, but everything on GPU; m=200~几千没问题)
    for e in range(m):
        idx = cp.array([e, e + 1])
        C[cp.ix_(idx, idx)] += C_e
        K_mat[cp.ix_(idx, idx)] += K_e

    cp.cuda.Stream.null.synchronize()
    t_assemble_end = time.time()
    assemble_time = t_assemble_end - t_assemble_start

    # 4. boundary conditions
    # remove left Dirichlet node: use nodes 1..m
    C_hat = C[1:, 1:].copy()    # m×m
    K_hat = K_mat[1:, 1:].copy()

    v_hat = v0[1:].copy()       # (m,)

    # A = C_hat/dtau + K_hat  (constant in time)
    A = C_hat / dtau + K_hat

    # 4.5 One-time Cholesky factorization on GPU
    t_factor_start = time.time()
    L = cp.linalg.cholesky(A)   # A = L @ L.T, L lower-triangular
    cp.cuda.Stream.null.synchronize()
    t_factor_end = time.time()
    factor_time = t_factor_end - t_factor_start

    # 5. time stepping
    t_solve_total = 0.0

    for i in range(1, n + 1):
        tau = tau_grid[i]

        # build F on GPU
        F = cp.zeros(m + 1, dtype=dtype)
        x_inf = x_max

        # Neumann right boundary derivative ∂v/∂x(x_inf, τ)
        dvdx_right = 0.5 * (
            (k1 + 1.0) * cp.exp(x_inf) - (k1 - 1.0) * cp.exp(-k1 * tau)
        ) * cp.exp(
            0.5 * (k1 - 1.0) * x_inf + 0.25 * (k1 + 1.0) ** 2 * tau
        )

        F[-1] = dvdx_right
        F_hat = F[1:]   # length m

        # right-hand side b = (C_hat/dtau)v_{i-1} + F_hat
        b = (C_hat @ v_hat) / dtau + F_hat

        # Solve A v_hat = b via Cholesky: A = L L^T
        #   1) L y = b
        #   2) L^T v_hat = y
        t_solve_start = time.time()
        y = solve_triangular(L, b, lower=True, check_finite=False)
        v_hat = solve_triangular(L.T, y, lower=False, check_finite=False)
        cp.cuda.Stream.null.synchronize()
        t_solve_end = time.time()
        t_solve_total += (t_solve_end - t_solve_start)

    # 6. reassemble full v_T0 (including left Dirichlet node)
    v_T0 = cp.zeros(m + 1, dtype=dtype)
    v_T0[0] = 0.0
    v_T0[1:] = v_hat

    # transform v(x, tau_max) into V(S, t=0)
    S_grid = K * cp.exp(x)
    tau_T0 = tau_max

    V_T0 = K * v_T0 * cp.exp(
        -0.5 * (k1 - 1.0) * x - 0.25 * (k1 + 1.0) ** 2 * tau_T0
    )

    # 7. timing
    cp.cuda.Stream.null.synchronize()
    t_total_end = time.time()
    total_runtime = t_total_end - t_start_total

    print(f"\n--- FEM Timing Report (GPU / CuPy + Cholesky) ---")
    print(f"Matrix assembly time:      {assemble_time:.6f} s")
    print(f"Cholesky factor time:      {factor_time:.6f} s")
    print(f"Triangular solves (total): {t_solve_total:.6f} s")
    print(f"Total runtime:             {total_runtime:.6f} s")
    print(f"Time steps: {n}, Elements: {m}")
    print("-------------------------------------------------\n")

    # 8. interpolate price at S0 (use NumPy on host for simplicity)
    S_grid_host = cp.asnumpy(S_grid)
    V_T0_host = cp.asnumpy(V_T0)

    if S0 <= S_grid_host[0]:
        return float(V_T0_host[0])
    if S0 >= S_grid_host[-1]:
        return float(V_T0_host[-1])

    price = float(np.interp(S0, S_grid_host, V_T0_host))

    metrics = {
        "S0": S0,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "S_max_factor": S_max_factor,
        "m": m,
        "n": n,
        "assembly_time_gpu": assemble_time,
        "solve_time_gpu": t_solve_total,
        "total_time": total_runtime,
        "price": price,
    }

    return price, metrics


