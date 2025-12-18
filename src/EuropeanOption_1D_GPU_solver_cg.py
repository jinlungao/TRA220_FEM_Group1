import time
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags
from cupyx.scipy.sparse.linalg import cg


def european_call_price_1D_gpu_cg(
    S0,                 # current price
    K=10.0,             # strike
    T=1.0,              # maturity
    r=0.10,             # risk-free rate
    sigma=0.20,         # volatility
    S_max_factor=4.0,   # right truncation multiple
    m=2000,             # elements (=> unknowns m after removing left Dirichlet)
    n=200,              # time steps
    use_float32=False,
    cg_tol=1e-10,       # CG tolerance
    cg_maxiter=5000,    # CG max iterations per time step
    cg_use_prev_as_x0=True,  # warm start CG using previous solution
):
    """
    1D FEM in transformed variables on GPU (CuPy), but solving each time step with
    Conjugate Gradient (CG) on a sparse tridiagonal matrix.
    Key idea:
      - For 1D linear FEM, C_hat and K_hat are tridiagonal.
      - Build A = C_hat/dtau + K_hat as sparse CSR once.
      - Each time step: b = (C_hat/dtau) v_prev + F_hat, then solve A v = b with CG.
    """
    dtype = cp.float32 if use_float32 else cp.float64
    t_start_total = time.time()

    # -----------------------------
    # 1) Preprocessing and grids
    # -----------------------------
    S_max = S_max_factor * K
    S_min = 1e-6 * K

    x_min = float(np.log(S_min / K))
    x_max = float(np.log(S_max / K))

    k1 = 2.0 * r / (sigma ** 2)
    tau_max = 0.5 * sigma ** 2 * T

    dx = (x_max - x_min) / m
    dtau = tau_max / n

    # Node grid (m+1 nodes)
    x = x_min + dx * cp.arange(m + 1, dtype=dtype)

    # -----------------------------
    # 2) Initial condition v(x, 0)
    # -----------------------------
    v0 = cp.maximum(
        cp.exp(0.5 * (k1 + 1.0) * x) - cp.exp(0.5 * (k1 - 1.0) * x),
        0.0
    ).astype(dtype)

    # Remove left Dirichlet node -> unknowns are nodes 1..m (length m)
    v_hat = v0[1:].copy()

    # -----------------------------------------
    # 3) Assemble tridiagonal C_hat and K_hat
    t_assemble_start = time.time()
    diag_C = cp.full(m, 2.0 * dx / 3.0, dtype=dtype)
    off_C  = cp.full(m - 1, dx / 6.0, dtype=dtype)

    diag_K = cp.full(m, 2.0 / dx, dtype=dtype)
    off_K  = cp.full(m - 1, -1.0 / dx, dtype=dtype)

    # Build sparse matrices (CSR) once
    C_sp = diags([off_C, diag_C, off_C], offsets=[-1, 0, 1], format="csr")
    K_sp = diags([off_K, diag_K, off_K], offsets=[-1, 0, 1], format="csr")

    # Constant system matrix A = C/dtau + K
    A_sp = (C_sp / dtau) + K_sp

    cp.cuda.Stream.null.synchronize()
    assemble_time = time.time() - t_assemble_start

    # -----------------------------------------
    # 4) Time stepping (CG solve each step)
    # -----------------------------------------
    t_solve_total = 0.0
    t_cg_iters_total = 0
    t_cg_fail_count = 0

    # Precompute constants for right Neumann boundary term
    x_inf = x_max
    c1 = 0.5 * (k1 + 1.0) * cp.exp(x_inf)
    c2 = 0.5 * (k1 - 1.0)
    c3 = 0.5 * (k1 - 1.0) * x_inf
    c4 = 0.25 * (k1 + 1.0) ** 2

    # Small helper to record CG iterations (SciPy-style callback)
    cg_iter_counter = {"k": 0}
    def _cg_callback(_xk):
        cg_iter_counter["k"] += 1

    for i in range(1, n + 1):
        tau = i * dtau

        # Neumann right boundary derivative ∂v/∂x(x_inf, τ)
        dvdx_right = (c1 - c2 * cp.exp(-k1 * tau)) * cp.exp(c3 + c4 * tau)

        # RHS: b = (C/dtau) v_prev + F_hat
        # Here F_hat is zero except last entry, where we add the Neumann contribution.
        b = (C_sp @ v_hat) / dtau
        b[-1] += dvdx_right

        # Warm start: use previous solution as x0 (often speeds up CG a lot)
        x0 = v_hat if cg_use_prev_as_x0 else None

        # Reset iteration counter
        cg_iter_counter["k"] = 0

        t_step_start = time.time()
        v_new, info = cg(
            A_sp, b,
            x0=x0,
            tol=cg_tol,
            maxiter=cg_maxiter,
            callback=_cg_callback
        )
        cp.cuda.Stream.null.synchronize()
        t_step_end = time.time()

        t_solve_total += (t_step_end - t_step_start)
        t_cg_iters_total += cg_iter_counter["k"]

        if info != 0:
            # info > 0: convergence not achieved within maxiter
            # info < 0: numerical breakdown
            t_cg_fail_count += 1
            raise RuntimeError(f"CG failed at step {i}/{n}, info={info} (iters={cg_iter_counter['k']})")

        v_hat = v_new

    # 5) Reconstruct full solution and transform back
    v_T0 = cp.zeros(m + 1, dtype=dtype)
    v_T0[0] = 0.0
    v_T0[1:] = v_hat

    S_grid = K * cp.exp(x)
    tau_T0 = tau_max

    V_T0 = K * v_T0 * cp.exp(
        -0.5 * (k1 - 1.0) * x - 0.25 * (k1 + 1.0) ** 2 * tau_T0
    )

    # 6) Timing + interpolation on CPU
    cp.cuda.Stream.null.synchronize()
    total_runtime = time.time() - t_start_total

    # Host interpolation for the requested S0
    S_grid_host = cp.asnumpy(S_grid)
    V_T0_host = cp.asnumpy(V_T0)
    price = float(np.interp(S0, S_grid_host, V_T0_host))

    print(f"\n--- FEM Timing Report (GPU / CuPy + Sparse CG) ---")
    print(f"Tridiag assembly time:       {assemble_time:.6f} s")
    print(f"CG solve time (total):       {t_solve_total:.6f} s")
    print(f"Total runtime:               {total_runtime:.6f} s")
    print(f"Time steps: {n}, Elements: {m}")
    print(f"CG tol: {cg_tol}, maxiter: {cg_maxiter}")
    print(f"CG total iters: {t_cg_iters_total}, avg/step: {t_cg_iters_total / max(n,1):.2f}")
    print(f"CG failures: {t_cg_fail_count}")
    print("-------------------------------------------------\n")

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
        "cg_tol": cg_tol,
        "cg_maxiter": cg_maxiter,
        "cg_total_iters": int(t_cg_iters_total),
        "cg_avg_iters_per_step": float(t_cg_iters_total / max(n, 1)),
        "price": price,
    }
    return price, metrics
