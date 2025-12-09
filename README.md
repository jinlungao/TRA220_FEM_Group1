TRA220_FEM_Group1/
│
├── benchmarks/               # Benchmark datasets & performance evaluation resources
│
├── data/                     # Output directory for run results (e.g., JSON metrics)
│   └── fem_run.json
│
├── scripts/                  # Main executables
│   ├── run_fem_cpu.py        # Entry point for CPU FEM solver
│   └── run_fem_gpu.py        # Entry point for GPU FEM solver
│
├── sources/                  # Supporting materials / additional resources
│
├── src/
│   ├── utils/
│   │   └── save_run_to_json.py   # JSON logging utilities
│   │
│   ├── EuropeanOption_CPU.py     # FEM solver (CPU implementation)
│   └── EuropeanOption_GPU.py     # FEM solver (GPU-accelerated, CUDA/CuPy)
│
└── README.md                 # Project documentation
