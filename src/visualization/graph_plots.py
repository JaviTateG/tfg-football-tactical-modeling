"""
Módulo de visualizaciones de grafos de pases.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List, Dict


class GraphPlots:
    """
    Clase para visualizaciones avanzadas de grafos de pases
    sobre el campo de fútbol.
    """

    # Dimensiones estándar campo StatsBomb (en metros)
    FIELD_LENGTH = 120
    FIELD_WIDTH  = 80

    def __init__(self, figsize: tuple = (14, 9)):
        self.figsize = figsize

    def draw_pitch(self, ax: plt.Axes, color: str = '#4a7c4e'):
        """
        Dibuja un campo de fútbol en el eje dado.

        Args:
            ax: Eje de matplotlib
            color: Color del césped
        """
        # Fondo
        ax.set_facecolor(color)
        ax.set_xlim(0, self.FIELD_LENGTH)
        ax.set_ylim(0, self.FIELD_WIDTH)

        line_kw = dict(color='white', linewidth=1.5, alpha=0.8)

        # Líneas del campo
        ax.plot([0, 0, self.FIELD_LENGTH, self.FIELD_LENGTH, 0],
                [0, self.FIELD_WIDTH, self.FIELD_WIDTH, 0, 0], **line_kw)
        ax.plot([self.FIELD_LENGTH/2, self.FIELD_LENGTH/2],
                [0, self.FIELD_WIDTH], **line_kw)

        # Círculo central
        circle = plt.Circle(
            (self.FIELD_LENGTH/2, self.FIELD_WIDTH/2),
            9.15, fill=False, **line_kw
        )
        ax.add_patch(circle)
        ax.plot(self.FIELD_LENGTH/2, self.FIELD_WIDTH/2,
                'o', color='white', markersize=3)

        # Área grande local (izquierda)
        ax.plot([0, 16.5, 16.5, 0],
                [13.85, 13.85, 66.15, 66.15], **line_kw)
        # Área pequeña local
        ax.plot([0, 5.5, 5.5, 0],
                [24.85, 24.85, 55.15, 55.15], **line_kw)

        # Área grande visitante (derecha)
        ax.plot([self.FIELD_LENGTH, self.FIELD_LENGTH-16.5,
                 self.FIELD_LENGTH-16.5, self.FIELD_LENGTH],
                [13.85, 13.85, 66.15, 66.15], **line_kw)
        # Área pequeña visitante
        ax.plot([self.FIELD_LENGTH, self.FIELD_LENGTH-5.5,
                 self.FIELD_LENGTH-5.5, self.FIELD_LENGTH],
                [24.85, 24.85, 55.15, 55.15], **line_kw)

        # Puntos de penalti
        ax.plot(11, self.FIELD_WIDTH/2, 'o', color='white', markersize=3)
        ax.plot(self.FIELD_LENGTH-11, self.FIELD_WIDTH/2,
                'o', color='white', markersize=3)

        ax.axis('off')

    def plot_pass_network_on_pitch(
        self,
        graph: nx.DiGraph,
        passes: pd.DataFrame,
        team: str,
        title: Optional[str] = None,
        min_passes: int = 3
    ):
        """
        Visualiza la red de pases de un equipo sobre el campo.

        Args:
            graph: Grafo dirigido de pases
            passes: DataFrame con pases preprocesados
            team: Nombre del equipo
            title: Título del gráfico
            min_passes: Mínimo de pases para mostrar una conexión
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        self.draw_pitch(ax)

        # Calcular posición media de cada jugador
        team_passes = passes[passes['team'] == team].copy()

        if 'location' not in team_passes.columns:
            print("  No hay datos de ubicación disponibles")
            plt.close()
            return

        def extract_coords(loc):
            try:
                if isinstance(loc, list):
                    return loc[0], loc[1]
                import ast
                loc = ast.literal_eval(loc)
                return loc[0], loc[1]
            except:
                return None, None

        team_passes['x'] = team_passes['location'].apply(
            lambda l: extract_coords(l)[0]
        )
        team_passes['y'] = team_passes['location'].apply(
            lambda l: extract_coords(l)[1]
        )

        player_pos = team_passes.groupby('player')[['x', 'y']].mean()

        # PageRank para tamaño de nodos
        pagerank = nx.pagerank(graph, weight='weight')
        max_pr   = max(pagerank.values()) if pagerank else 1

        # Dibujar aristas
        edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                 if d.get('weight', 0) >= min_passes]

        max_weight = max([d['weight'] for _, _, d in edges]) if edges else 1

        for u, v, d in edges:
            if u not in player_pos.index or v not in player_pos.index:
                continue
            x1, y1 = player_pos.loc[u, 'x'], player_pos.loc[u, 'y']
            x2, y2 = player_pos.loc[v, 'x'], player_pos.loc[v, 'y']
            width   = 0.5 + 4 * (d['weight'] / max_weight)
            alpha   = 0.3 + 0.5 * (d['weight'] / max_weight)

            ax.annotate(
                '', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color='white',
                    lw=width, alpha=alpha
                )
            )

        # Dibujar nodos
        for player, row in player_pos.iterrows():
            if player not in graph.nodes:
                continue
            pr   = pagerank.get(player, 0)
            size = 100 + 800 * (pr / max_pr)

            ax.scatter(
                row['x'], row['y'],
                s=size, c='#A50044',
                edgecolors='white', linewidths=1.5,
                zorder=5
            )

            # Nombre abreviado
            short_name = player.split()[-1] if player else ''
            ax.annotate(
                short_name,
                (row['x'], row['y'] + 2.5),
                ha='center', va='bottom',
                fontsize=7, color='white',
                fontweight='bold', zorder=6
            )

        title = title or f"Red de Pases — {team}"
        ax.set_title(title, fontsize=14, fontweight='bold',
                     color='white', pad=10)
        fig.patch.set_facecolor('#1a1a2e')

        plt.tight_layout()
        plt.show()

    def plot_centrality_comparison(
        self,
        metrics_df: pd.DataFrame,
        metrics: List[str] = None,
        top_n: int = 10,
        title: str = 'Comparación de Métricas de Centralidad'
    ):
        """
        Visualiza comparación de métricas de centralidad por jugador.

        Args:
            metrics_df: DataFrame con métricas por jugador
            metrics: Lista de métricas a comparar
            top_n: Número de jugadores a mostrar
            title: Título del gráfico
        """
        if metrics is None:
            metrics = ['pagerank', 'betweenness', 'out_degree_centrality']

        metrics = [m for m in metrics if m in metrics_df.columns]
        if not metrics:
            print("  No se encontraron las métricas especificadas")
            return

        df_top = metrics_df.head(top_n)

        fig, axes = plt.subplots(
            1, len(metrics),
            figsize=(6 * len(metrics), 7)
        )
        if len(metrics) == 1:
            axes = [axes]

        colors = ['#A50044', '#004D98', '#EDBB00']

        for idx, metric in enumerate(metrics):
            df_plot = df_top.sort_values(metric, ascending=True)
            axes[idx].barh(
                df_plot['player'], df_plot[metric],
                color=colors[idx % len(colors)],
                edgecolor='black', linewidth=0.5,
                alpha=0.85
            )
            axes[idx].set_xlabel(
                metric.replace('_', ' ').title(), fontsize=11
            )
            axes[idx].set_title(
                metric.replace('_', ' ').title(),
                fontsize=12, fontweight='bold'
            )
            axes[idx].grid(axis='x', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_graph_summary(
        self,
        graph: nx.DiGraph,
        title: str = 'Resumen del Grafo de Pases'
    ):
        """
        Visualiza un resumen estadístico del grafo.

        Args:
            graph: Grafo dirigido
            title: Título
        """
        in_degrees  = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        weights     = [d['weight'] for _, _, d in graph.edges(data=True)]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # In-degree
        axes[0].hist(
            list(in_degrees.values()), bins=15,
            color='#A50044', edgecolor='black', alpha=0.8
        )
        axes[0].set_xlabel('In-Degree', fontsize=11)
        axes[0].set_ylabel('Frecuencia', fontsize=11)
        axes[0].set_title('Distribución In-Degree\n(Pases Recibidos)',
                          fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Out-degree
        axes[1].hist(
            list(out_degrees.values()), bins=15,
            color='#004D98', edgecolor='black', alpha=0.8
        )
        axes[1].set_xlabel('Out-Degree', fontsize=11)
        axes[1].set_ylabel('Frecuencia', fontsize=11)
        axes[1].set_title('Distribución Out-Degree\n(Pases Realizados)',
                          fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # Pesos
        axes[2].hist(
            weights, bins=20,
            color='#EDBB00', edgecolor='black', alpha=0.8
        )
        axes[2].set_xlabel('Peso (Frecuencia de Pases)', fontsize=11)
        axes[2].set_ylabel('Frecuencia', fontsize=11)
        axes[2].set_title('Distribución de Pesos\nde Aristas',
                          fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Estadísticas
        print(f"\n RESUMEN DEL GRAFO:")
        print(f"   Nodos:            {graph.number_of_nodes()}")
        print(f"   Aristas:          {graph.number_of_edges()}")
        print(f"   Densidad:         {nx.density(graph):.4f}")
        print(f"   In-degree medio:  {np.mean(list(in_degrees.values())):.2f}")
        print(f"   Out-degree medio: {np.mean(list(out_degrees.values())):.2f}")
        print(f"   Peso medio:       {np.mean(weights):.2f}")