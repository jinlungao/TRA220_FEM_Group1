# TRA220_FEM_Group1
A project of TRA220 FEM, Group 1, Chalmers University of Technology
The main entry point of the project is located in the scripts/ directory.

TRA220_FEM_Group1/
│
├── benchmarks/              # Benchmark datasets or performance evaluation resources
│
├── data/                    # Output directory for run results (e.g., JSON files)
│   └── fem_run.json
│
├── scripts/
│   └── run_fem_cpu.py       # Main execution script for the CPU FEM solver
|   └── run_fem_gpu.py       # Main execution script for the GPU FEM solver 
│
├── sources/                 # Additional source files or supporting materials
│
├── src/
│   ├── utils/
│   │   └── save_run_to_json.py   # Utility for saving FEM run metrics to JSON
│   │
│   ├── EuropeanOption_CPU.py     # CPU implementation of the FEM solver
│   └── EuropeanOption_GPU.py     # GPU-accelerated FEM solver
│
└── README.md                # Project documentation
