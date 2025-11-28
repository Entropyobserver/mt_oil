from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualizer:
    def __init__(self, style: str = 'default'):
        plt.style.use(style)
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    def plot_learning_curve(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        save_path: Path,
        title: str = "Learning Curve",
        xlabel: str = "Training Data Size",
        figsize: tuple = (12, 6)
    ):
        
        fig, axes = plt.subplots(1, len(y_cols), figsize=figsize)
        if len(y_cols) == 1:
            axes = [axes]
        
        for idx, (ax, y_col) in enumerate(zip(axes, y_cols)):
            ax.plot(data[x_col], data[y_col], 'o-', linewidth=2, 
                   markersize=8, color=self.colors[idx])
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(y_col.upper(), fontsize=12)
            ax.set_title(f'{title} - {y_col.upper()}', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_heatmap(
        self,
        data: pd.DataFrame,
        index_col: str,
        columns_col: str,
        values_col: str,
        save_path: Path,
        title: str = "Heatmap",
        figsize: tuple = (10, 8),
        cmap: str = 'YlOrRd'
    ):
        
        pivot = data.pivot_table(values=values_col, index=index_col, columns=columns_col)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap=cmap, ax=ax, 
                   cbar_kws={'label': values_col.upper()})
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(
        self,
        data: Dict[str, float],
        save_path: Path,
        title: str = "Comparison",
        ylabel: str = "Score",
        figsize: tuple = (10, 6)
    ):
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = list(data.keys())
        values = list(data.values())
        x = np.arange(len(labels))
        
        bars = ax.bar(x, values, color=self.colors[:len(labels)], 
                     edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pareto_front(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        save_path: Path,
        title: str = "Pareto Front",
        highlight_best: bool = True,
        figsize: tuple = (10, 6)
    ):
        
        def is_pareto_efficient(costs):
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                    is_efficient[i] = True
            return is_efficient
        
        costs = np.column_stack([data[x_col], data[y_col]])
        pareto_mask = is_pareto_efficient(costs)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(data[~pareto_mask][x_col], data[~pareto_mask][y_col],
                  c='lightgray', s=50, alpha=0.6, label='Other')
        ax.scatter(data[pareto_mask][x_col], data[pareto_mask][y_col],
                  c='red', s=100, alpha=0.8, label='Pareto Front', edgecolors='darkred')
        
        if highlight_best:
            best_idx = (data[x_col] + data[y_col]).idxmax()
            ax.scatter(data.loc[best_idx, x_col], data.loc[best_idx, y_col],
                      c='gold', s=300, marker='*', label='Best Config',
                      edgecolors='black', linewidth=2, zorder=5)
        
        ax.set_xlabel(x_col.upper(), fontsize=12)
        ax.set_ylabel(y_col.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()