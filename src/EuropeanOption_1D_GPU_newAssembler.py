import time
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular


def european_call_price_1D_gpu_newAssembler(
        S0, K=10.0, T=1.0, r=0.10, sigma=0.20,
        S_max_factor=4.0, m=1000, n=1000, use_float32=False
):
    dtype = cp.float32 if use_float32 else cp.float64
    t_start_total = time.time()

    # 1. preprocessing
    S_max, S_min = S_max_factor * K, 1e-6 * K
    x_min, x_max = float(np.log(S_min / K)), float(np.log(S_max / K))
    k1 = 2.0 * r / (sigma ** 2)
    tau_max = 0.5 * sigma ** 2 * T
    dx = (x_max - x_min) / m
    dtau = tau_max / n

    x = cp.linspace(x_min, x_max, m + 1, dtype=dtype)

    # 2. initial
    v_hat = cp.maximum(cp.exp(0.5 * (k1 + 1.0) * x) - cp.exp(0.5 * (k1 - 1.0) * x), 0.0)[1:].astype(dtype)

    # 3. assemble
    t_assemble_start = time.time()

    main_idx = cp.arange(m + 1)
    off_idx = cp.arange(m)

    # mass matrix C
    diag_C = cp.full(m, 2.0 * dx / 3.0, dtype=dtype)
    diag_C[-1] = dx / 3.0
    off_diag_C = cp.full(m - 1, dx / 6.0, dtype=dtype)

    # stiffness  matrix K
    diag_K = cp.full(m, 2.0 / dx, dtype=dtype)
    diag_K[-1] = 1.0 / dx
    off_diag_K = cp.full(m - 1, -1.0 / dx, dtype=dtype)

    # A = C/dtau + K
    A_diag = diag_C / dtau + diag_K
    A_off = off_diag_C / dtau + off_diag_K

    # Cholesky
    A = cp.diag(A_diag) + cp.diag(A_off, k=1) + cp.diag(A_off, k=-1)

    C_hat = cp.diag(diag_C) + cp.diag(off_diag_C, k=1) + cp.diag(off_diag_C, k=-1)

    cp.cuda.Stream.null.synchronize()
    assemble_time = time.time() - t_assemble_start

    # 4. Cholesky
    t_factor_start = time.time()
    L = cp.linalg.cholesky(A)
    cp.cuda.Stream.null.synchronize()
    factor_time = time.time() - t_factor_start

    # 5. time_step
    t_solve_start_total = time.time()

    x_inf = x_max
    c1 = 0.5 * (k1 + 1.0) * cp.exp(x_inf)
    c2 = 0.5 * (k1 - 1.0)
    c3 = 0.5 * (k1 - 1.0) * x_inf
    c4 = 0.25 * (k1 + 1.0) ** 2

    b = cp.zeros(m, dtype=dtype)

    for i in range(1, n + 1):
        tau = i * dtau
        # Neumann
        dvdx_right = (c1 - c2 * cp.exp(-k1 * tau)) * cp.exp(c3 + c4 * tau)

        b = (C_hat @ v_hat) / dtau
        b[-1] += dvdx_right

        # solver
        y = solve_triangular(L, b, lower=True, check_finite=False)
        v_hat = solve_triangular(L.T, y, lower=False, check_finite=False)

    cp.cuda.Stream.null.synchronize()
    t_solve_total = time.time() - t_solve_start_total

    # 6. retransfer
    v_total = cp.concatenate([cp.array([0.0], dtype=dtype), v_hat])
    V_T0 = K * v_total * cp.exp(-0.5 * (k1 - 1.0) * x - 0.25 * (k1 + 1.0) ** 2 * tau_max)

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
    price = float(np.interp(S0, cp.asnumpy(K * cp.exp(x)), cp.asnumpy(V_T0)))

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