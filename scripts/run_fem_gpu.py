from src.EuropeanOption_1D_GPU import european_call_price_1D_gpu
from src.utils.save_run_to_json import save_gpu_run_to_json

if __name__ == "__main__":
    K = 10.0
    S0 = 10.0
    T = 1.0
    r = 0.10
    sigma = 0.20
    #The number of grids in the spatial direction
    #change this value to make increase the amount of computation
    m0 = 5000

    price,metrics = european_call_price_1D_gpu(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        S_max_factor=4.0,
        m=m0,
        n=200
    )
    save_gpu_run_to_json(metrics)