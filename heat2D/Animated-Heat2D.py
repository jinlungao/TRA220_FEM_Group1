import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_mesh(nx, ny):
    # Grid points
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    
    # generate 2D grid.
    X, Y = np.meshgrid(x, y, indexing='ij')
    # flatten the 2D grid into a 1D list of (x,y) pairs.
    coords = np.column_stack([X.ravel(), Y.ravel()]) 

    # helper function that maps index of a node located at grid position (i,j)
    def node_id(i, j):
        return i * (ny + 1) + j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = node_id(i, j) # bottom left node (i,j)
            n1 = node_id(i + 1, j) # bottom right node (i+1,j)
            n2 = node_id(i, j + 1) # top left node (i, j+1)
            n3 = node_id(i + 1, j + 1) # top right node (i+1, j+1)

            # Two triangles per rectangle:
            # lower-left triangle (n0, n1, n3) ex:
            elements.append([n0, n1, n3])#      n2  n3
            #                                      / |
            #                                   n0--n1
            
            # upper-right triangle (n0, n3, n2) ex:
            elements.append([n0, n3, n2])#      n2--n3
            #                                   |  / 
            #                                   n0  n1

    elements = np.array(elements, dtype=int)

    # Boundary nodes: x=0,1 or y=0,1 
    # we find nodes that lie on the boundary.
    # we need small tolerance as 1.0 and 0.0 might not be exactly equal to 0, i.e. 1.0000000001
    tol = 1e-14 
    on_boundary = (
        (np.abs(coords[:, 0]) < tol) |
        (np.abs(coords[:, 0] - 1.0) < tol) |
        (np.abs(coords[:, 1]) < tol) |
        (np.abs(coords[:, 1] - 1.0) < tol)
    )
    boundary_nodes = np.where(on_boundary)[0]

    return coords, elements, boundary_nodes


 # We use P1 elements (linear triangular elements).
 # We have to change this if we fo with a different mesh.
def local_matrices(coords_elem, kappa): # these are allways 3x3 matricies.
    #  get the coordinates of the first node.
    x1, y1 = coords_elem[0]
    x2, y2 = coords_elem[1]
    x3, y3 = coords_elem[2]

    # Calculate the area of the triange, each is the same.
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # Gradients coefficients for P1 basis
    # grad(phi_i) = 1/(2*area) * [b_i, c_i]
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Local stiffness: K_ij = kappa/(4*area) * (b_i b_j + c_i c_j)
    Ke = kappa / (4.0 * area) * (np.outer(b, b) + np.outer(c, c))

    # Local mass matrix for P1: Me = (area/12) * [[2,1,1],[1,2,1],[1,1,2]]
    Me = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                   [1.0, 2.0, 1.0],
                                   [1.0, 1.0, 2.0]])

    return Ke, Me


def assemble_system(coords, elements, kappa):
    
    Nnodes = coords.shape[0]
    Ne = elements.shape[0]

    K = sp.lil_matrix((Nnodes, Nnodes), dtype=float)
    M = sp.lil_matrix((Nnodes, Nnodes), dtype=float)

    for e in range(Ne):
        nodes = elements[e]
        coords_elem = coords[nodes]
        Ke, Me = local_matrices(coords_elem, kappa)

        for i_local, i_global in enumerate(nodes):
            for j_local, j_global in enumerate(nodes):
                K[i_global, j_global] += Ke[i_local, j_local]
                M[i_global, j_global] += Me[i_local, j_local]

    return K.tocsr(), M.tocsr()



def solve_heat_equation(
    nx,
    ny,
    kappa,
    dt,
    T,
):        
    
    # U is the temperature field.
    # Default initial condition: a Gaussian bump in the center
    def u0_func(x, y):
        return np.exp(-50.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    
    # Heating function. In this case it is zero so we are neither heating or cooling the invironment.
    # This is the external heating source/sink
    #def f_func(x, y, t):
    #    return 0.0
    # 
    # ----------- A more difficut heat function:  ----------------------------------------
    
    #def f_func(x, y, t):
    #    if (x - 0.3)**2 + (y - 0.3)**2 < 0.1**2:
    #        return 100.0  # High intensity heating
    #    return 0.0
    
    # ------------ An even more complex function:  ------------------------------------------
    def f_func(x, y, t):
            if (x - 0.3)**2 + (y - 0.3)**2 < 0.1**2:
                return 50.0  # high intensity heating
            elif (0.7 < x < 0.8) and (0.6 < y < 0.8):
                return 100.0 * np.sin(10 * t * np.pi) # High intensity heating/cooling depending on time.
            return 0.0
    
    # The code follows the folloing ppeline:
    #   1. Builds mesh
    #   2. Builds local element matricies and Assembels global Mass and Stiffness matrix
    #   3. Sets initial conditions
    #   4. Loops over time using backwards euler method.
    
    # ---------------------------------------------- 1 ------------------------------------------
    coords, elements, boundary_nodes = generate_mesh(nx, ny)
    #Nnodes = coords.shape[0]

    # ---------------------------------------------- 2 ------------------------------------------
    # Assemble global matrices
    K, M = assemble_system(coords, elements, kappa)

    # Time-stepping matrices: (M + dt*kappa*K) u^{n+1} = M u^n + dt F^{n+1}
    # LHS:
    # since niether the mesh, nor time steps ever change then A remains constant. We can precompute this outside of the "simulation"
    A = M + dt * K
    A_fact = spla.factorized(A)  # LU factorization for repeated solves : Try changing to something else like Conjugate Gradient.

    
    # ---------------------------------------------- 3 ------------------------------------------
    # Initial condition vector
    # U is the temperature field.
    U = np.array([u0_func(x, y) for (x, y) in coords])

    # Enforce Dirichlet BC on initial condition (u=0 on boundary)
    U[boundary_nodes] = 0.0 # we can change later if we want to.
 

    # ---------------------------------------------- 4 ------------------------------------------
       
    # Time loop
    t = 0.0
    nsteps = int(np.round(T / dt))
    
    # for animation:
    frames=60
    # Calculate how many steps to skip between saves
    save_interval = max(1, nsteps // frames)
    print(f"Solving {nsteps} time steps. Saving every {save_interval} steps.")
    
    # Store history for animation
    u_history = [U.copy()]
    
    for n in range(nsteps):
        t_next = t + dt

        # Build RHS: M u^n + dt F^{n+1}
        # Here, F is the FEM load vector for f(x,y,t_next).
        # We evaluate f at nodes and mass-lump: F ≈ M * f_vec.
        f_vec = np.array([f_func(x, y, t_next) for (x, y) in coords])
        RHS = M.dot(U) + dt * M.dot(f_vec)

        # Apply Dirichlet BCs (u=0 on boundary) to the linear system
        # Easiest: set RHS[boundary_nodes] = 0 and then force U[boundary_nodes]=0 afterwards.
        RHS[boundary_nodes] = 0.0

        # Solve
        U_new = A_fact(RHS)
        
        # U[boundary_nodes]=0 afterwards. (Dirichlet BC)
        U_new[boundary_nodes] = 0.0

        U = U_new
        t = t_next
        
        # Save frame
        if (n + 1) % save_interval == 0:
            u_history.append(U.copy())

    return coords, u_history


if __name__ == "__main__":
    nx = 50
    ny = 50
    T_final = 1
    dt = 1e-2
    
    # Run solver
    coords, u_history = solve_heat_equation(
        nx=nx,
        ny=ny,
        kappa=1.0,
        dt=dt,
        T=T_final,
    )
    
    # Animated plot:
    
    print(f"Simulation complete. {len(u_history)} frames captured.")

    # --- Animation Setup ---
    # Reshape coordinates for plotting
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Reshape 1D node arrays to 2D grids for pcolormesh
    # We add 1 because node count is element count + 1
    X = x.reshape((nx + 1, ny + 1))
    Y = y.reshape((nx + 1, ny + 1))
    
    # Determine color limits based on the initial state (assuming heat decays or stays similar)
    # If you expect heat to grow significantly, calculate max from the whole u_history
    vmin = min(u_history[0].min(), 0.0) 
    vmax = u_history[0].max()

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create the initial plot using pcolormesh. 
    # shading='gouraud' allows for smooth interpolation between nodes, similar to contourf
    mesh_plot = ax.pcolormesh(
        X, Y, 
        u_history[0].reshape((nx + 1, ny + 1)), 
        cmap='inferno', 
        shading='gouraud',
        vmin=vmin, 
        vmax=vmax
    )
    
    fig.colorbar(mesh_plot, label='Temperature')
    ax.set_title("Heat Diffusion")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Update function for the animation
    def update(frame_idx):
        U_current = u_history[frame_idx].reshape((nx + 1, ny + 1))
        
        # Update the data in the mesh plot
        # For gouraud shading, we pass the 1D flattened array
        mesh_plot.set_array(U_current.ravel())
        
        ax.set_title(f"Heat Diffusion (Frame {frame_idx}/{len(u_history)})")
        return mesh_plot,

    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(u_history), 
        interval=50, # milliseconds per frame
        blit=False
    )

    plt.show()
    
    
