# Motion-Token MAPF: Multi-Agent Highway Coordination

This project implements an advanced multi-agent coordination architecture based on **MotionLM** (an autoregressive Transformer-based motion model) and **BK-PBS** (Behavior-Knowledge Priority-Based Search).

The primary goal is to enable autonomous agents (CAVs) to navigate safely and efficiently in high-density traffic environments alongside human-driven vehicles (HDVs) by predicting their reactions to prevent collisions during critical maneuvers such as highway merging.

---

## Quick Start

The project is designed to be highly portable and capable of running in headless environments (servers without monitors).

### 1. Environment Setup
We have streamlined the installation into a single script that enforces Python 3.12 and manages all critical dependencies (e.g., PyArrow, Lightning).

```bash
source setup_env.sh
```
*This command creates the `.env` virtual environment, installs all packages, and activates it in your current terminal session.*

### 2. Run Full Pipeline
To execute the entire research cycle (Data Collection -> Training -> Benchmarking) automatically, use the **Prefect**-based orchestrator:

```bash
python src/scripts/orchestrate_pipeline.py
```

---

## Orchestrator Workflow

The `orchestrate_pipeline.py` script automates the following phases:

1.  **Phase 1: Data Collection** (`collect_data`)
    *   Executes high-density simulations (up to 20 agents).
    *   Uses `HFDatasetWriter` to save trajectories directly in Hugging Face Dataset format (memory-optimized).
2.  **Phase 2: Model Training** (`train_model`)
    *   Trains the **MotionLM** model using PyTorch Lightning.
    *   Hardened for small test datasets with dynamic validation interval adjustments.
3.  **Phase 3: Benchmarking** (`run_benchmark`)
    *   Evaluates the trained model against baselines (IDM-only, Static-PBS).
    *   Generates metrics for Success Rate, Collision Rate, and Latency.
    *   **Video Recording**: Automatically generates MP4 videos in `outputs/benchmarks_videos/` for visual verification.

---

## Key Features

*   **Total Portability**: Leverages `sys.executable` and relative paths. Clone the repo on any server and run the pipeline without changing a single line of code.
*   **Headless Ready**: Native support for `rgb_array` rendering and `imageio` recording, enabling video generation on server-side clusters.
*   **Dependency Hardening**: Pinned versions for `protobuf`, `pyarrow`, and `torch` to prevent common Deep Learning environment conflicts.
*   **Hydra Configuration**: The entire system is modular and configurable via YAML files in the `configs/` directory.

---

## Key Directories

*   `src/scripts/`: Orchestration and benchmarking scripts.
*   `experiments/`: Base training and data collection scripts.
*   `data/trajectories/`: Storage for generated datasets.
*   `checkpoints/`: Trained models (`.ckpt`).
*   `outputs/`: Evaluation results and video recordings.
