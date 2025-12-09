import numpy as np

from src.EuropeanOption_1D_CPU import european_call_1D_price
from src.utils.save_run_to_json import save_cpu_run_to_json

if __name__ == "__main__":
    K = 10.0
    S0 = 10.0
    T = 1.0
    r = 0.10
    sigma = 0.20

    m_values = np.linspace(10, 5000, 20, dtype=int)

    results = []

    for m0 in m_values:
        print(f"Running m = {m0} ...")
        price, metrics = european_call_1D_price(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            S_max_factor=4.0,
            m=m0,
            n=200
        )
        save_cpu_run_to_json(metrics)

    print("All done!")