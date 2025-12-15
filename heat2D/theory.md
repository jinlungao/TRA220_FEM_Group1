# Finite Element Formulation: 2D Heat Equation (Backward Euler)

This document outlines the mathematical derivation for solving the 2D heat equation using the Finite Element Method (FEM) with implicit time stepping (Backward Euler).

## 1. Strong Form (The PDE)

We start with the transient heat equation in a domain $\Omega \subset \mathbb{R}^2$. Let $u(\mathbf{x}, t)$ represent the temperature field.

$$
\frac{\partial u}{\partial t} - \nabla \cdot (\kappa \nabla u) = f \quad \text{in } \Omega
$$

With the following conditions:
* **Dirichlet Boundary Conditions:** $u = 0$ on $\partial \Omega$ (the boundary).
* **Initial Condition:** $u(\mathbf{x}, 0) = u_0(\mathbf{x})$ (Gaussian hump).
* $\kappa$: Thermal diffusivity (constant).
* $f$: Source term (assumed $0$ for simple cooling, but kept for generality).

---

## 2. The Weak Formulation

To derive the weak form, we multiply the strong form by a smooth **test function** $v(\mathbf{x})$ (which vanishes on the boundary $\partial \Omega$ where Dirichlet BCs are applied) and integrate over the domain $\Omega$:

$$
\int_{\Omega} \left( \frac{\partial u}{\partial t} - \nabla \cdot (\kappa \nabla u) \right) v \, d\Omega = \int_{\Omega} f v \, d\Omega
$$

We can split the integral:

$$
\int_{\Omega} \frac{\partial u}{\partial t} v \, d\Omega - \int_{\Omega} \nabla \cdot (\kappa \nabla u) v \, d\Omega = \int_{\Omega} f v \, d\Omega
$$

### Integration by Parts (Green's First Identity)
We apply integration by parts to the diffusion term (the Laplacian). The rule is:
$$
-\int_{\Omega} \nabla \cdot (\kappa \nabla u) v \, d\Omega = \int_{\Omega} \kappa \nabla u \cdot \nabla v \, d\Omega - \int_{\partial \Omega} \kappa (\nabla u \cdot \mathbf{n}) v \, d\Gamma
$$

Since we have homogeneous Dirichlet boundary conditions ($u=0$), our test functions $v$ must also be $0$ on the boundary $\partial \Omega$. Therefore, the boundary integral term vanishes.

### Final Weak Form
Find $u$ such that for all valid test functions $v$:

$$
\int_{\Omega} \frac{\partial u}{\partial t} v \, d\Omega + \int_{\Omega} \kappa \nabla u \cdot \nabla v \, d\Omega = \int_{\Omega} f v \, d\Omega
$$

---

## 3. Spatial Discretization (Galerkin FEM)

We approximate the solution $u(\mathbf{x},t)$ as a linear combination of basis functions (shape functions) $\phi_j(\mathbf{x})$ with time-dependent coefficients $u_j(t)$:

$$
u(\mathbf{x}, t) \approx u_h(\mathbf{x}, t) = \sum_{j=1}^{N} u_j(t) \phi_j(\mathbf{x})
$$

We also choose the test function $v$ to be one of these basis functions, $v = \phi_i(\mathbf{x})$.

Substituting these into the weak form:

$$
\int_{\Omega} \left( \sum_{j=1}^{N} \frac{d u_j}{dt} \phi_j \right) \phi_i \, d\Omega + \int_{\Omega} \kappa \nabla \left( \sum_{j=1}^{N} u_j \phi_j \right) \cdot \nabla \phi_i \, d\Omega = \int_{\Omega} f \phi_i \, d\Omega
$$

Since summation and differentiation are linear, we can pull the coefficients out:

$$
\sum_{j=1}^{N} \frac{d u_j}{dt} \underbrace{\int_{\Omega} \phi_j \phi_i \, d\Omega}_{M_{ij}} + \sum_{j=1}^{N} u_j \underbrace{\int_{\Omega} \kappa \nabla \phi_j \cdot \nabla \phi_i \, d\Omega}_{K_{ij}} = \underbrace{\int_{\Omega} f \phi_i \, d\Omega}_{F_i}
$$

### Matrix Definitions

1.  **Mass Matrix ($M$):** Represents the overlap of basis functions.
    $$
    M_{ij} = \int_{\Omega} \phi_i \phi_j \, d\Omega
    $$

2.  **Stiffness Matrix ($K$):** Represents the diffusion/conduction.
    $$
    K_{ij} = \int_{\Omega} \kappa (\nabla \phi_i \cdot \nabla \phi_j) \, d\Omega = \int_{\Omega} \kappa \left( \frac{\partial \phi_i}{\partial x}\frac{\partial \phi_j}{\partial x} + \frac{\partial \phi_i}{\partial y}\frac{\partial \phi_j}{\partial y} \right) d\Omega
    $$

3.  **Load Vector ($\mathbf{F}$):** Represents the source term.
    $$
    F_i = \int_{\Omega} f \phi_i \, d\Omega
    $$

### Semi-Discrete System
This results in a system of Ordinary Differential Equations (ODEs) in time:

$$
M \mathbf{\dot{u}}(t) + K \mathbf{u}(t) = \mathbf{F}(t)
$$

---

## 4. Temporal Discretization (Backward Euler)

We use the **Backward Euler** method (implicit) to solve the ODE.
We approximate the time derivative at time step $t^n$ as:

$$
\mathbf{\dot{u}}(t^n) \approx \frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t}
$$

Because it is an implicit method, we evaluate the spatial terms ($K \mathbf{u}$ and $\mathbf{F}$) at the current time step $t^{n+1}$.

$$
M \left( \frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t} \right) + K \mathbf{u}^{n+1} = \mathbf{F}^{n+1}
$$

### Rearranging for the Linear Solver

We need to solve for the unknown vector $\mathbf{u}^{n+1}$ (temperature at current step).

1.  Multiply by $\Delta t$:
    $$
    M (\mathbf{u}^{n+1} - \mathbf{u}^{n}) + \Delta t K \mathbf{u}^{n+1} = \Delta t \mathbf{F}^{n+1}
    $$

2.  Group terms with $\mathbf{u}^{n+1}$ on the Left Hand Side (LHS) and known terms ($\mathbf{u}^{n}$) on the Right Hand Side (RHS):
    $$
    M \mathbf{u}^{n+1} + \Delta t K \mathbf{u}^{n+1} = M \mathbf{u}^{n} + \Delta t \mathbf{F}^{n+1}
    $$

3.  Factor out $\mathbf{u}^{n+1}$:
    $$
    (M + \Delta t K) \mathbf{u}^{n+1} = M \mathbf{u}^{n} + \Delta t \mathbf{F}^{n+1}
    $$

## 5. Final System to Solve

At every time step, you are solving a linear system of the form $A\mathbf{x} = \mathbf{b}$:

$$
\underbrace{(M + \Delta t K)}_{\text{LHS Matrix } A} \cdot \underbrace{\mathbf{u}^{n+1}}_{\text{Unknown}} = \underbrace{M \mathbf{u}^{n} + \Delta t \mathbf{F}^{n+1}}_{\text{RHS Vector } \mathbf{b}}
$$

* **LHS ($A$):** This matrix is constant (if $\Delta t$ is constant) and symmetric positive definite. You can pre-factorize it (e.g., LU decomposition or Cholesky) for speed.
* **RHS ($b$):** This vector is updated every step based on the previous temperature profile.