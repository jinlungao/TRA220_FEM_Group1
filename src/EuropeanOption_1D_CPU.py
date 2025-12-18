import time

import numpy as np
import platform

"""
date: 9 December 2025
reference paper
https://www.sciencedirect.com/science/article/pii/S0898122111002331?ref=pdf_download&fr=RR-2&rr=9ab4d928285c92eb
- Black–Scholes PDE >> v_tau - v_xx = 0
- The spatial domain is discretized using 1D linear finite elements.
- The temporal domain is discretized using backward difference.
"""

def european_call_1D_price(
    S0,  #current price
    K=10.0, #strike price
    T=1.0,  #maturity
    r=0.10, #risk-free interest rate
    sigma=0.20,  #volatility
    S_max_factor=4.0,  #Right-side spatial truncation multiple  w
    m=200,       # finite elements
    n=200        # time steps
):
    """
    # timer
    # We would like to obtain the following metrics: 1.Total Run Time, 2.Assembly Time 3.Solve Time.
    """
    t_start_total = time.time()



    # 1.preprocessing & Variable transformation parameters
    # S ∈ (0, S_max)
    S_max = S_max_factor * K
    S_min = 1e-6 * K

    # x = ln(S/K), τ = σ^2/2 * (T - t)
    x_min = np.log(S_min / K)
    x_max = np.log(S_max / K)

    # k1 = r / (σ^2/2) = 2r / σ^2（equation(9)）
    k1 = 2.0 * r / (sigma ** 2)

    #max τ >> t = 0
    tau_max = 0.5 * sigma ** 2 * T

    # space and time mesh（equation(17)-(20)）
    dx = (x_max - x_min) / m
    x = x_min + dx * np.arange(m + 1)   # m+1 nodes

    dtau = tau_max / n
    tau_grid = dtau * np.arange(n + 1)

    # 2.initial condition v(x, 0)  (equation(11))
    # payoff：V(S,T) = max(S-K, 0) transfer to  v(x,0) （equation(11)）
    # v(x,0) = max( e^{0.5(k1+1)x} - e^{0.5(k1-1)x}, 0 )
    v0 = np.maximum(
        np.exp(0.5 * (k1 + 1.0) * x) - np.exp(0.5 * (k1 - 1.0) * x),
        0.0
    )

    # 3.Construct the global quality matrix C and the stiffness matrix K
    # Use the linear 2-node element (in the form of (32) in the paper)
    # Unit local matrix：
    """
    # assemble matrix/ can be accelerated by GPU
    """
    #assemble_start time
    t_assemble_start = time.time()

    C_e = (dx / 3.0) * np.array([[1.0, 0.5],
                                 [0.5, 1.0]])
    K_e = (1.0 / dx) * np.array([[1.0, -1.0],
                                 [-1.0, 1.0]])


    C = np.zeros((m + 1, m + 1))
    K_mat = np.zeros((m + 1, m + 1))

    for e in range(m):
        idx = np.array([e, e + 1])
        C[np.ix_(idx, idx)] += C_e
        K_mat[np.ix_(idx, idx)] += K_e

    #assemble end time
    t_assemble_end = time.time()
    assemble_time = t_assemble_end - t_assemble_start


    # 4. boundary conditions
    # left: x_min → S≈0：eu call, V≈0 → v≈0，Dirichlet
    # right: x_max：Neumann
    # get submatrix（nodes 1..m）
    C_hat = C[1:, 1:].copy()   # m×m
    K_hat = K_mat[1:, 1:].copy()

    # initialize v node (i=0 >> τ=0)
    v_hat = v0[1:].copy()      # m

    # A = C_hat/dtau + K_hat
    A = C_hat / dtau + K_hat

    # 5. time step：from τ=0 to τ=tau_max
    # for each step, calulate(C_hat/dtau + K_hat) v_i = (C_hat/dtau) v_{i-1} + F_i_hat
    # F_i comes from Neumann condition（equation(40)）
    # -----------------------------
    # total solve time
    t_solve_total = 0.0

    for i in range(1, n + 1):
        tau = tau_grid[i]
        # vector F（reference paper p.8 equation(40)）
        F = np.zeros(m + 1)
        x_inf = x_max

        # right boundary的 ∂v/∂x(x_inf, τ)（equation(13)）
        # F_{m+1} = 0.5 * ((k1+1)e^{x_inf} - (k1-1)e^{-k1 τ})
        #           * exp( 0.5 (k1-1)x_inf + 0.25 (k1+1)^2 τ )
        dvdx_right = 0.5 * (
            (k1 + 1.0) * np.exp(x_inf) - (k1 - 1.0) * np.exp(-k1 * tau)
        ) * np.exp(
            0.5 * (k1 - 1.0) * x_inf + 0.25 * (k1 + 1.0)**2 * tau
        )

        #For the last unit of the linear element, the Neumann boundary condition only acts on the last node.
        F[-1] = dvdx_right

        # remove 0st node
        F_hat = F[1:]

        # right hand vector
        b = (C_hat @ v_hat) / dtau + F_hat
        """
        # solve linear equations//can be accelerated by GPU
        # maybe use different solvers to evaluate performance
        """
        t_solve_start = time.time()
        v_hat = np.linalg.solve(A, b)
        t_solve_total += (time.time() - t_solve_start)

    """
    # Reassemble into v//can be accelerated by GPU
    """
    v_T0 = np.zeros(m + 1)
    v_T0[0] = 0.0               #Left Dirichlet boundary
    v_T0[1:] = v_hat

    # 6. transfer v(x, τ_max) into V(S, t=0)
    # equation(8):
    # v = (1/K) V * exp( 0.5(k1-1)x + 0.25(k1+1)^2 τ )
    # ⇒ V = K * v * exp( -0.5(k1-1)x - 0.25(k1+1)^2 τ )
    # -----------------------------
    S_grid = K * np.exp(x)
    tau_T0 = tau_max

    V_T0 = K * v_T0 * np.exp(
        -0.5 * (k1 - 1.0) * x - 0.25 * (k1 + 1.0)**2 * tau_T0
    )

    # total run time
    t_total_end = time.time()
    total_runtime = t_total_end - t_start_total


    # print time
    print(f"\n--- FEM Timing Report ---")
    print(f"Matrix assembly time:     {assemble_time:.6f} s")
    print(f"Linear solve time:        {t_solve_total:.6f} s")
    print(f"Total runtime:            {total_runtime:.6f} s")
    print(f"Time steps: {n}, Elements: {m}")
    print("----------------------------\n")


    # 7. The interpolation yields the price corresponding to the given S0. If S0 exceeds the grid range, it is truncated.
    if S0 <= S_grid[0]:
        return float(V_T0[0])
    if S0 >= S_grid[-1]:
        return float(V_T0[-1])

    # linear interpolation
    price = float(np.interp(S0, S_grid, V_T0))
    cpu_model = platform.processor()

    #data record
    metrics = {
        "S0": S0,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "S_max_factor": S_max_factor,
        "m": m,
        "n": n,
        "assembly_time": assemble_time,
        "solve_time": t_solve_total,
        "total_time": total_runtime,
        "cpu_model": cpu_model,
        "price": price,
    }

    return price, metrics






