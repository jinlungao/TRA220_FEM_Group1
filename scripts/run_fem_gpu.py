import numpy as np

from src.EuropeanOption_1D_GPU import european_call_price_1D_gpu
from src.EuropeanOption_1D_GPU_newAssembler import european_call_price_1D_gpu_newAssembler
from src.EuropeanOption_1D_GPU_solver_cg import european_call_price_1D_gpu_cg
from src.utils.save_run_to_json import save_gpu_run_to_json


def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    else:
        return obj


if __name__ == "__main__":
    K = 10.0
    S0 = 10.0
    T = 1.0
    r = 0.10
    sigma = 0.20
    #The number of grids in the spatial direction
    #change this value to make increase the amount of computation
    m_values = np.linspace(10, 5000, 20, dtype=int)

    results = []

    for m0 in m_values:
        print(f"Running m = {m0} ...")
        price, metrics = european_call_price_1D_gpu_cg(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            S_max_factor=4.0,
            m=m0,
            n=200
        )
        clean_metrics = numpy_to_python(metrics)
        save_gpu_run_to_json(clean_metrics)

    print("All done!")

