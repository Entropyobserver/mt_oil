from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualizer:
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        plt.style.use("default")
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    def plot_learning_curve(
        self,
        data: pd.DataFrame,
        output_path: Path,
        x_col: str = "data_size",
        y_col: str = "bleu",
        title: str = "Learning Curve"
    ) -> None:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(
            data[x_col],
            data[y_col],
            "o-",
            linewidth=2,
            markersize=8,
            color=self.colors[0]
        )
        
        ax.set_xlabel("Training Data Size", fontsize=12)
        ax.set_ylabel("BLEU Score", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_parameter_sensitivity(
        self,
        data: pd.DataFrame,
        output_path: Path,
        params: List[str] = None,
        metric: str = "bleu"
    ) -> None:
        
        if params is None:
            params = ["r", "alpha", "dropout"]
        
        fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5))
        
        if len(params) == 1:
            axes = [axes]
        
        for idx, param in enumerate(params):
            grouped = data.groupby(param)[metric].agg(["mean", "std"])
            
            axes[idx].errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                fmt="o-",
                linewidth=2,
                markersize=8,
                capsize=5,
                color=self.colors[idx]
            )
            
            axes[idx].set_xlabel(param.upper(), fontsize=12)
            axes[idx].set_ylabel(f"{metric.upper()} Score", fontsize=12)
            axes[idx].set_title(f"Effect of {param.upper()}", fontsize=14)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_heatmap(
        self,
        data: pd.DataFrame,
        output_path: Path,
        x_param: str = "r",
        y_param: str = "alpha",
        metric: str = "bleu"
    ) -> None:
        
        pivot_df = data.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc="mean"
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": f"{metric.upper()} Score"}
        )
        
        ax.set_title(f"{metric.upper()} Heatmap: {x_param} vs {y_param}", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_comparison(
        self,
        baseline_metrics: Dict,
        finetuned_metrics: Dict,
        output_path: Path,
        metrics: List[str] = None
    ) -> None:
        
        if metrics is None:
            metrics = ["bleu", "chrf"]
        
        baseline_values = [baseline_metrics.get(m, 0) for m in metrics]
        finetuned_values = [finetuned_metrics.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(
            x - width / 2,
            baseline_values,
            width,
            label="Baseline",
            color=self.colors[0]
        )
        bars2 = ax.bar(
            x + width / 2,
            finetuned_values,
            width,
            label="Fine-tuned",
            color=self.colors[1]
        )
        
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom"
                )
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()