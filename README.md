```
TRA220_FEM_Group1/
│
├── benchmarks/                 # Benchmark datasets & performance evaluation resources
│
├── data/                       # Output directory for FEM run results (e.g., JSON metrics)
│   ├── picture/                # Pictures Showing the Experimental Results
│   └── fem_run.json
│
├── scripts/                    # Main execution scripts
│   ├── run_fem_cpu.py          # Entry point for CPU FEM solver
│   └── run_fem_gpu.py          # Entry point for GPU FEM solver
│
├── sources/                    # Additional supporting materials
│
├── src/
│   ├── utils/
│   │   └── save_run_to_json.py # JSON logging utilities
│   │
│   ├── EuropeanOption_CPU.py   # CPU implementation of the FEM solver
│   └── EuropeanOption_GPU.py   # GPU-accelerated solver (CUDA / CuPy)
│
└── README.md                   # Project documentation
```
