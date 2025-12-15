# FEM Code Explanation: Mesh, Physics, and Assembly

This document explains a simple Python pipeline for solving physics problems (like heat transfer) on a 2D square domain. The process consists of three stages:
1. **`generate_mesh`**: Dividing the geometry into triangles.
2. **`local_matrices`**: Calculating physics for a single triangle.
3. **`assemble_system`**: Combining everything into a global linear system.

We will use a **Continuous Example** throughout this document: A simple **1x1 grid** (a single square split into two triangles).

---

## 1. `generate_mesh(nx, ny)`

### **Explanation**
This function creates the geometry. It defines a grid of points on a square $[0,1] \times [0,1]$ and connects them to form triangles.
* **Input:** `nx`, `ny` (Number of rectangles in x and y directions).
* **Output:** * `coords`: A list of $(x, y)$ positions for every node.
    * `elements`: A list of triangles, where each triangle is defined by the indices of its 3 corners.
    * `boundary_nodes`: Indices of nodes lying on the edges of the square.

### **Visual Logic**
The function moves through the grid rectangle by rectangle. It splits every rectangle into two triangles using a diagonal cut.

### **Example Trace (`nx=1, ny=1`)**
* **Nodes:** The function creates 4 nodes. 
    * Node 0: $(0,0)$
    * Node 1: $(0,1)$
    * Node 2: $(1,0)$
    * Node 3: $(1,1)$
* **Elements:** It connects them into 2 triangles.
    * Triangle A: Nodes $(0, 2, 3)$
    * Triangle B: Nodes $(0, 3, 1)$

### **Data Structure**
```python
# coords (4 nodes, x/y values)
[[0.0, 0.0],  # Node 0
 [0.0, 1.0],  # Node 1
 [1.0, 0.0],  # Node 2
 [1.0, 1.0]]  # Node 3

# elements (2 triangles, referencing Node IDs)
[[0, 2, 3],   # Triangle A
 [0, 3, 1]]   # Triangle B

 ```

## 2. `local_matrices(coords_elem, kappa)`

### **Explanation**
This function calculates the "local" physics for **one specific triangle**. 
* **Stiffness Matrix (`Ke`):** Represents how the element resists change (gradient energy). Calculated using the gradients of the basis functions.
* **Mass Matrix (`Me`):** Represents the volume/inertia of the element. Calculated using the element's area.

### **Input & Output**
* **Input:** Coordinates of just the 3 corners of the triangle (e.g., Triangle A).
* **Output:** Two $3 \times 3$ matrices (`Ke`, `Me`).

### **Example Trace (Triangle A)**
We process **Triangle A** (Nodes 0, 2, 3).
* Coordinates: $(0,0), (1,0), (1,1)$.
* **Area:** $0.5$
* **Stiffness (`Ke`):** The code calculates gradients vectors $b$ and $c$ based on coordinate differences and computes $\frac{\kappa}{4A} (b \otimes b + c \otimes c)$.

### **Data Structure**
```python
# Ke for Triangle A (Local indices 0, 1, 2 correspond to Global Nodes 0, 2, 3)
[[ 0.5, -0.5,  0.0],
 [-0.5,  1.0, -0.5],
 [ 0.0, -0.5,  0.5]]

= 0.5* [[ 1, -1,  0.0],
        [-1,  2, -1],
        [ 0.0, -1,  1]]

 ```