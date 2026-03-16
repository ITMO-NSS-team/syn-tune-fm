import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.pipeline.runner import ExperimentRunner

def plot_experiment_results(results: dict, cfg: DictConfig, shared_dir: Path):
    """
    Создает и сохраняет столбчатые диаграммы для каждой метрики.
    """
    metrics = cfg.evaluation.metrics
    variants = list(results.keys())
    
    # Создаем папку plots в ОБЩЕЙ директории мультирана (с таймстемпом)
    plots_dir = shared_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем значение дисбаланса для имени файла
    minority_frac = cfg.get("minority_fraction", "null")
    
    for metric in metrics:
        means = []
        stds = []
        
        for var in variants:
            means.append(results[var].get(f"{metric}_mean", 0.0))
            stds.append(results[var].get(f"{metric}_std", 0.0))
            
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(variants))
        
        bars = plt.bar(
            x_pos, means, yerr=stds, capsize=8, 
            alpha=0.8, color='#4C72B0', edgecolor='black'
        )
        
        plt.xticks(x_pos, variants, rotation=45, ha="right")
        plt.ylabel(metric.replace('_', ' ').title())
        
        title_mf = minority_frac if minority_frac is not None else "None (50%)"
        plt.title(f"Dataset: {cfg.dataset.name} | Generator: {cfg.generator.name}\nMinority Fraction: {title_mf} | Metric: {metric}")
        
        for bar, mean_val, std_val in zip(bars, means, stds):
            yval = bar.get_height()
            offset = std_val if std_val > 0 else 0
            plt.text(
                bar.get_x() + bar.get_width()/2, yval + offset + 0.005, 
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Уникальное имя файла, чтобы графики мультирана не перезаписывали друг друга!
        plot_path = plots_dir / f"{metric}_mf_{minority_frac}.png"
        plt.savefig(str(plot_path), dpi=300)
        plt.close()
        print(f"      [Plot Saved] {plot_path}")


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig):
    # 1. Запуск эксперимента
    runner = ExperimentRunner(cfg)
    results = runner.run()

    # --- ОПРЕДЕЛЯЕМ ОБЩУЮ ПАПКУ ДЛЯ ТЕКУЩЕГО ЗАПУСКА (С ТАЙМСТЕМПОМ) ---
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    
    if hydra_cfg.mode == hydra.types.RunMode.MULTIRUN:
        shared_exp_dir = output_dir.parent
    else:
        shared_exp_dir = output_dir

    # 2. Генерируем графики в общую папку
    print("\n[5/5] Generating Visualizations...")
    plot_experiment_results(results, cfg, shared_exp_dir)

    # 3. Собираем параметры текущего запуска
    row = {
        "dataset": cfg.dataset.name,
        "generator": cfg.generator.name,
        "model": cfg.model.name,
        "minority_fraction": cfg.get("minority_fraction", "None"), 
    }
    
    # 4. Распаковываем результаты по вариантам
    if isinstance(results, dict):
        for variant, metrics in results.items(): 
            for metric_name, val in metrics.items():
                row[f"{variant}_{metric_name}"] = round(val, 4)

    df = pd.DataFrame([row])

    # 5. Сохраняем локальный CSV чисто для этого запуска (в папку с таймстемпом)
    local_csv = shared_exp_dir / "run_results.csv"
    df.to_csv(str(local_csv), mode='a', header=not local_csv.exists(), index=False)

    # 6. Сохраняем глобальный CSV для истории (в корень проекта)
    orig_cwd = Path(get_original_cwd())
    global_csv = orig_cwd / "all_experiments_results.csv"
    df.to_csv(str(global_csv), mode='a', header=not global_csv.exists(), index=False)
    
    print("\n" + "="*80)
    print(f"EXPERIMENTS SUMMARY TABLE")
    print(f" -> Local record: {local_csv}")
    print(f" -> Global record: {global_csv}")
    print("="*80)
    print(df.T.to_string(header=False))

if __name__ == "__main__":
    main()