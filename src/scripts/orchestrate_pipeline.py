import os
import subprocess
import sys
from prefect import flow, task
from datetime import datetime
import glob

# Máxima portabilidad dinámica
PYTHON_EXEC = sys.executable
BASE_DIR = os.getcwd()


@task(log_prints=True)
def collect_data(num_episodes: int, tag: str):
    """Fase 1: Recolección masiva de trayectorias."""
    print(
        f"🎬 [1/3] Iniciando recolección de {num_episodes} episodios en data/trajectories/{tag}..."
    )
    cmd = [
        PYTHON_EXEC,
        "experiments/run_hdv_data_collection.py",
        f"num_episodes={num_episodes}",
        f"+tag={tag}",
        "++max_agents=20",
        "++render=false",
    ]
    subprocess.run(cmd, check=True)


@task(log_prints=True)
def train_model(tag: str, max_steps: int):
    """Fase 2: Entrenamiento del modelo MotionLM."""
    dataset_path = os.path.join("data", "trajectories", tag)
    print(f"🧠 [2/3] Entrenando modelo con dataset: {dataset_path}...")
    # Usamos ++ para sobrescribir los valores del config.yaml
    cmd = [
        PYTHON_EXEC,
        "experiments/train_pl.py",
        f"data.hf_path='{dataset_path}'",
        f"++trainer.max_steps={max_steps}",
    ]
    subprocess.run(cmd, check=True)


@task(log_prints=True)
def find_best_checkpoint():
    """Busca el checkpoint más reciente generado en la sesión."""
    search_paths = [
        os.path.join("checkpoints", "**", "*.ckpt"),
        os.path.join("lightning_logs", "version_*", "checkpoints", "*.ckpt"),
    ]

    checkpoints = []
    for path in search_paths:
        checkpoints.extend(glob.glob(path, recursive=True))

    if not checkpoints:
        raise FileNotFoundError(
            "❌ No se encontraron checkpoints. Revisa 'checkpoints/'"
        )

    # Seleccionamos el último modificado
    best_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"🏆 Mejor modelo detectado: {best_ckpt}")
    return best_ckpt


@task(log_prints=True)
def run_benchmark(checkpoint_path: str):
    """Fase 3: Evaluación comparativa de planners."""
    print(f"📊 [3/3] Ejecutando Benchmark sobre: {checkpoint_path}...")

    cmd = [
        PYTHON_EXEC,
        "src/scripts/benchmark_planners.py",
        f"++ckpt_path='{checkpoint_path}'",
        "++max_agents=20",
        "++num_rollouts=10",  # Valor estándar de benchmark
        "++test_render_mode=none",
    ]
    subprocess.run(cmd, check=True)


@flow(name="MotionLM Production Pipeline")
def motion_lm_pipeline(num_episodes: int = 1000, max_steps: int = 150000):
    print(f"🚀 Iniciando Pipeline de Producción en: {BASE_DIR}")

    # For this experiment with the 64-token tokenizer, we use the existing high-quality dataset
    tag = "run_20260423_015507"

    # 1. Collect Data (Commented out as we have the dataset ready)
    # collect_data(num_episodes, tag)

    # 2. Train Model
    # Since vocab size changed (169 -> 64), we start training from zero
    train_model(tag, max_steps)

    # 3. Benchmark
    best_model = find_best_checkpoint()
    run_benchmark(best_model)


if __name__ == "__main__":
    # Puedes sobrescribir estos valores aquí si lo deseas
    motion_lm_pipeline()
