import numpy as np
import matplotlib.pyplot as plt

def generate_mesh(nx, ny):
    """
    Generate a uniform mesh on [0,1]x[0,1] with nx, ny elements in x,y directions,
    each rectangle is split into two triangles.
    """
    # Grid points
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel()])

    # Helper to map (i,j) -> node index
    def node_id(i, j):
        return i * (ny + 1) + j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)

            # Two triangles per rectangle:
            # lower-left triangle (n0, n1, n3)
            elements.append([n0, n1, n3])
            # upper-right triangle (n0, n3, n2)
            elements.append([n0, n3, n2])

    elements = np.array(elements, dtype=int)

    # Boundary nodes: x=0,1 or y=0,1
    tol = 1e-9
    on_boundary = (
        (np.abs(coords[:, 0]) < tol) |
        (np.abs(coords[:, 0] - 1.0) < tol) |
        (np.abs(coords[:, 1]) < tol) |
        (np.abs(coords[:, 1] - 1.0) < tol)
    )
    boundary_nodes = np.where(on_boundary)[0]

    return coords, elements, boundary_nodes

def visualize_mesh(coords, elements, boundary_nodes):
    """
    Plots the mesh nodes, elements, and highlights the boundary.
    """
    # Create the figure
    plt.figure(figsize=(8, 8))
    
    # 1. Plot the mesh edges (Triangulation)
    # coords[:,0] is X, coords[:,1] is Y
    plt.triplot(coords[:, 0], coords[:, 1], elements, 
                color='k', linewidth=1, alpha=0.5, label='Mesh Edges')

    # 2. Plot all nodes (Vertices)
    plt.scatter(coords[:, 0], coords[:, 1], 
                color='blue', s=10, zorder=2, label='Internal Nodes')

    # 3. Highlight boundary nodes
    # We select only the coordinates that correspond to boundary_nodes indices
    bx = coords[boundary_nodes, 0]
    by = coords[boundary_nodes, 1]
    plt.scatter(bx, by, color='red', s=40, zorder=3, label='Boundary Nodes')

    # Formatting
    plt.title(f"Mesh Visualization\nNodes: {len(coords)}, Elements: {len(elements)}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper right')
    plt.axis('equal') # Important so the square looks like a square
    plt.grid(True, linestyle=':', alpha=0.3)
    
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Parameters for the mesh
    nx_elements = 10
    ny_elements = 10

    # Generate data
    pts, tris, bnd_nodes = generate_mesh(nx_elements, ny_elements)

    # Visualize
    visualize_mesh(pts, tris, bnd_nodes)