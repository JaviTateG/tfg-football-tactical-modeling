"""
Módulo de visualizaciones temporales de métricas de grafos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Optional, Dict


class TemporalPlots:
    """
    Clase para visualizaciones de la evolución temporal
    de métricas de grafos de pases.
    """

    COLORS = {
        'primary':    '#A50044',
        'secondary':  '#004D98',
        'accent':     '#EDBB00',
        'neutral':    '#6B7280',
        'background': '#1a1a2e'
    }

    def __init__(self, figsize: tuple = (16, 6)):
        self.figsize = figsize

    def _get_time_col(self, temporal_metrics: pd.DataFrame):
        """Obtiene el array de tiempo correcto."""
        if 'window_start' in temporal_metrics.columns:
            return temporal_metrics['window_start'].values
        return temporal_metrics.index.values

    def plot_metric_evolution(
        self,
        temporal_metrics: pd.DataFrame,
        metrics: List[str],
        title: str = 'Evolución Temporal de Métricas',
        mark_halftime: bool = True
    ):
        """
        Visualiza la evolución temporal de métricas de red.

        Args:
            temporal_metrics: DataFrame con métricas por ventana temporal
            metrics: Lista de métricas a visualizar
            title: Título del gráfico
            mark_halftime: Marcar el descanso (minuto 45)
        """
        metrics = [m for m in metrics if m in temporal_metrics.columns]
        if not metrics:
            print("  No se encontraron las métricas especificadas")
            return

        time_col = self._get_time_col(temporal_metrics)

        n         = len(metrics)
        fig, axes = plt.subplots(n, 1, figsize=(self.figsize[0], 5 * n))
        if n == 1:
            axes = [axes]

        colors = [
            self.COLORS['primary'],
            self.COLORS['secondary'],
            self.COLORS['accent'],
            self.COLORS['neutral']
        ]

        for idx, metric in enumerate(metrics):
            ax    = axes[idx]
            color = colors[idx % len(colors)]
            vals  = temporal_metrics[metric].values

            ax.fill_between(
                time_col, vals,
                alpha=0.3, color=color
            )
            ax.plot(
                time_col, vals,
                color=color, linewidth=2, label=metric
            )

            mean_val = vals.mean()
            ax.axhline(
                y=mean_val, color=color,
                linestyle='--', alpha=0.6, linewidth=1,
                label=f'Media: {mean_val:.4f}'
            )

            if mark_halftime:
                ax.axvline(
                    x=45, color='gray', linestyle='-',
                    alpha=0.4, linewidth=1.5
                )
                y_max = vals.max()
                ax.text(
                    45.5, y_max * 0.95,
                    'HT', fontsize=9, color='gray'
                )

            ax.set_xlabel('Minuto', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(
                metric.replace('_', ' ').title(),
                fontsize=12, fontweight='bold'
            )
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_halftime_comparison(
        self,
        temporal_metrics: pd.DataFrame,
        metrics: List[str] = None,
        title: str = 'Comparación Primera vs Segunda Mitad'
    ):
        """
        Compara métricas entre primera y segunda mitad.

        Args:
            temporal_metrics: DataFrame con métricas por ventana
            metrics: Métricas a comparar
            title: Título
        """
        if metrics is None:
            metrics = ['density', 'avg_clustering', 'num_edges']

        metrics = [m for m in metrics if m in temporal_metrics.columns]
        if not metrics:
            print("  No se encontraron las métricas especificadas")
            return

        if 'window_start' in temporal_metrics.columns:
            first_half  = temporal_metrics[
                temporal_metrics['window_start'] <= 45
            ]
            second_half = temporal_metrics[
                temporal_metrics['window_start'] > 45
            ]
        else:
            first_half  = temporal_metrics[temporal_metrics.index <= 45]
            second_half = temporal_metrics[temporal_metrics.index > 45]

        fig, axes = plt.subplots(1, len(metrics), figsize=self.figsize)
        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            val_1h = first_half[metric].mean()
            val_2h = second_half[metric].mean()

            bars = axes[idx].bar(
                ['1ª Mitad', '2ª Mitad'],
                [val_1h, val_2h],
                color=[self.COLORS['primary'], self.COLORS['secondary']],
                edgecolor='black', linewidth=0.8, alpha=0.85
            )

            for bar, val in zip(bars, [val_1h, val_2h]):
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold'
                )

            if val_1h > 0:
                pct = ((val_2h - val_1h) / val_1h) * 100
                axes[idx].text(
                    0.5, 0.02,
                    f'Variación: {pct:+.1f}%',
                    ha='center', transform=axes[idx].transAxes,
                    fontsize=10, color='gray'
                )

            axes[idx].set_title(
                metric.replace('_', ' ').title(),
                fontsize=12, fontweight='bold'
            )
            axes[idx].set_ylabel('Valor medio', fontsize=11)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_player_evolution(
        self,
        temporal_metrics: pd.DataFrame,
        players: List[str],
        metric: str = 'pagerank',
        title: str = 'Evolución de Importancia de Jugadores'
    ):
        """
        Visualiza la evolución de jugadores a lo largo del partido.

        Args:
            temporal_metrics: DataFrame con métricas temporales
            players: Lista de jugadores a visualizar
            metric: Métrica de centralidad a mostrar
            title: Título
        """
        time_col = self._get_time_col(temporal_metrics)

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = [
            self.COLORS['primary'],
            self.COLORS['secondary'],
            self.COLORS['accent'],
            self.COLORS['neutral']
        ]

        for idx, player in enumerate(players):
            col = f'{metric}_{player}'
            if col not in temporal_metrics.columns:
                matching = [
                    c for c in temporal_metrics.columns
                    if player.split()[-1].lower() in c.lower()
                ]
                if not matching:
                    print(f"  No se encontró {player} en las métricas")
                    continue
                col = matching[0]

            color = colors[idx % len(colors)]
            ax.plot(
                time_col,
                temporal_metrics[col].values,
                color=color, linewidth=2,
                label=player, marker='o', markersize=3
            )

        ax.axvline(x=45, color='gray', linestyle='--', alpha=0.5)
        y_lim = ax.get_ylim()
        ax.text(
            45.5, y_lim[1] * 0.95,
            'HT', fontsize=9, color='gray'
        )
        ax.set_xlabel('Minuto', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_tactical_changes(
        self,
        changes: Dict,
        metric: str = 'density',
        title: str = 'Cambios Tácticos Detectados'
    ):
        """
        Visualiza los cambios tácticos detectados en el partido.

        Args:
            changes: Diccionario de cambios detectados
            metric: Métrica analizada
            title: Título
        """
        if not changes:
            print("  No se detectaron cambios significativos")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        minutes = list(changes.keys())
        values  = list(changes.values())

        colors = [
            self.COLORS['primary'] if v > 0
            else self.COLORS['secondary']
            for v in values
        ]

        ax.bar(
            minutes, values,
            color=colors, edgecolor='black',
            linewidth=0.8, alpha=0.85, width=1.5
        )

        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(
            x=45, color='gray', linestyle='--',
            alpha=0.5, linewidth=1.5
        )

        if values:
            ax.text(
                45.5, max(values) * 0.9,
                'HT', fontsize=9, color='gray'
            )

        patch_up   = mpatches.Patch(
            color=self.COLORS['primary'], label='Aumento'
        )
        patch_down = mpatches.Patch(
            color=self.COLORS['secondary'], label='Descenso'
        )
        ax.legend(handles=[patch_up, patch_down], fontsize=10)

        ax.set_xlabel('Minuto', fontsize=12)
        ax.set_ylabel(f'Variación en {metric}', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\n CAMBIOS TÁCTICOS DETECTADOS ({metric}):")
        print(f"   Total cambios: {len(changes)}")
        if changes:
            max_min = max(changes, key=lambda k: abs(changes[k]))
            print(f"   Cambio mayor: minuto {max_min} "
                  f"({changes[max_min]:+.4f})")