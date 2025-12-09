import matplotlib.pyplot as plt

# exact data from jason log
elements = [10,272,535,797,1060,1323,1585,1848,2111,2373,2636,2898,3161,3424,3686,3949,4212,4474,4737,5000]

matrix_time = [
    0.001095,0.024636,0.049358,0.073837,0.101553,0.126912,0.152062,0.173569,
    0.197417,0.224148,0.244697,0.272132,0.296892,0.320881,0.346666,0.370255,
    0.390877,0.420094,0.443223,0.464238
]

linear_solve_time = [
    0.010511,0.083487,0.240066,0.460324,1.025210,1.630504,2.321388,3.290919,
    5.158371,6.850395,8.494740,11.462414,14.729769,17.929461,22.572374,
    27.526383,29.730567,37.402651,44.159317,48.232485
]

total_time = [
    0.013293,0.112170,0.300491,0.554628,1.160799,1.809822,2.552226,3.587671,
    5.525681,7.300717,9.033534,12.117139,15.488158,18.842868,23.536257,
    28.590241,30.880464,38.698420,45.559339,49.739213
]

# start to picture
plt.figure(figsize=(10,6))

plt.plot(elements, matrix_time, marker='o', label="Matrix assembly time")
plt.plot(elements, linear_solve_time, marker='o', label="Linear solve time")
plt.plot(elements, total_time, marker='o', label="Total runtime")

plt.xlabel("Number of Elements (m)")
plt.ylabel("Runtime (seconds)")
plt.title(
    "FEM Runtime Scaling with Spatial Elements on CPU\n"
    "(AMD Ryzen 7 7800X3D, 8 cores / 8 threads, base clock 4.20 GHz)"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
