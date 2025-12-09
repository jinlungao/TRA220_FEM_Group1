import cupy as cp

def assemble_fem_1d(m, L, dtype=cp.float64):
    """
    m : 单元个数
    L : 总长度
    """
    dx = L / m

    # 单元质量矩阵和刚度矩阵（在 GPU 上）
    C_e = (dx / 3.0) * cp.array([[1.0, 0.5],
                                  [0.5, 1.0]], dtype=dtype)
    K_e = (1.0 / dx) * cp.array([[1.0, -1.0],
                                 [-1.0,  1.0]], dtype=dtype)

    # 全局矩阵：开在 GPU 上
    C = cp.zeros((m + 1, m + 1), dtype=dtype)
    K_mat = cp.zeros((m + 1, m + 1), dtype=dtype)

    # 组装：每个单元贡献到相邻两个节点
    for e in range(m):
        i = e
        j = e + 1
        # 用切片一次性加二维块
        C[i:j+1, i:j+1] += C_e
        K_mat[i:j+1, i:j+1] += K_e

    return C, K_mat