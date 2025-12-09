import matplotlib.pyplot as plt

#exact data from jason data
elements = [10,272,535,797,1060,1323,1585,1848,2111,2373,2636,2898,3161,3424,3686,3949,4212,4474,4737,5000]

matrix_time = [
    0.009186,0.132107,0.266557,0.405308,0.527336,0.655458,0.774238,0.898097,
    1.042744,1.170962,1.297459,1.418363,1.551028,1.708732,1.811022,1.946435,
    2.046211,2.199331,2.331858,2.458705
]

cholesky_time = [
    0.016626,0.001695,0.019037,0.004679,0.005635,0.008399,0.011795,0.015795,
    0.020042,0.023809,0.030238,0.037123,0.043564,0.052961,0.149079,0.162657,
    0.173212,0.255707,0.243570,0.270991
]

triangular_time = [
    0.027917,0.030546,0.036314,0.044146,0.056329,0.069864,0.085040,0.099907,
    0.126587,0.143928,0.171848,0.203369,0.215126,0.245928,0.306469,0.375445,
    0.390552,0.382496,0.425738,0.461994
]

total_time = [
    0.253971,0.226236,0.382006,0.517557,0.647591,0.792787,0.930879,1.074297,
    1.250544,1.402880,1.564593,1.724319,1.872204,2.070768,2.333520,2.549862,
    2.676094,2.905615,3.068522,3.260343
]

plt.figure(figsize=(10,6))

plt.plot(elements, matrix_time, marker='o', label="Matrix assembly time")
plt.plot(elements, cholesky_time, marker='o', label="Cholesky factor time")
plt.plot(elements, triangular_time, marker='o', label="Triangular solves time")
plt.plot(elements, total_time, marker='o', label="Total runtime")

plt.xlabel("Number of Elements (m)")
plt.ylabel("Runtime (seconds)")
plt.title(
    "GPU FEM Runtime Scaling (CuPy + Cholesky)\n"
    "GPU: NVIDIA <NVIDIA GeForce RTX 4070Ti>"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
