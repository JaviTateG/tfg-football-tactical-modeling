"""
Módulo para simulación de estrategias tácticas.
Evalúa el impacto de cambios tácticos en las métricas del grafo.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import copy

from src.simulation.graph_modifications import GraphModifications
from src.graphs.network_metrics import NetworkMetrics


class StrategySimulator:
    """
    Clase para simular escenarios tácticos alternativos
    y evaluar su impacto en las métricas del grafo.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Args:
            graph: Grafo dirigido original de pases
        """
        self.graph      = graph
        self.modifier   = GraphModifications(graph)
        self.scenarios  = {}

    def simulate_player_removal(
        self,
        players: List[str]
    ) -> pd.DataFrame:
        """
        Simula el impacto de eliminar cada jugador del grafo.
        Útil para identificar jugadores críticos.

        Args:
            players: Lista de jugadores a analizar

        Returns:
            DataFrame con impacto por jugador
        """
        print("\n SIMULANDO IMPACTO DE ELIMINACIÓN DE JUGADORES...")
        

        # Métricas originales
        orig_density   = nx.density(self.graph)
        orig_pagerank  = np.mean(
            list(nx.pagerank(self.graph, weight='weight').values())
        )
        orig_between   = np.mean(
            list(nx.betweenness_centrality(
                self.graph, weight='weight'
            ).values())
        )

        results = []

        for player in players:
            if player not in self.graph.nodes:
                continue

            # Simular eliminación
            self.modifier.reset()
            G_mod = self.modifier.remove_player(player)

            if G_mod.number_of_nodes() == 0:
                continue

            # Calcular métricas del grafo modificado
            new_density  = nx.density(G_mod)
            new_pagerank = np.mean(
                list(nx.pagerank(G_mod, weight='weight').values())
            ) if G_mod.number_of_nodes() > 1 else 0
            new_between  = np.mean(
                list(nx.betweenness_centrality(
                    G_mod, weight='weight'
                ).values())
            ) if G_mod.number_of_nodes() > 1 else 0

            results.append({
                'player':           player,
                'density_change':   new_density  - orig_density,
                'pagerank_change':  new_pagerank - orig_pagerank,
                'between_change':   new_between  - orig_between,
                'density_pct':      (new_density - orig_density) / orig_density * 100,
                'impact_score':     abs(new_density - orig_density) +
                                    abs(new_pagerank - orig_pagerank)
            })

        self.modifier.reset()

        df = pd.DataFrame(results).sort_values(
            'impact_score', ascending=False
        ).reset_index(drop=True)

        print(f"\n IMPACTO POR JUGADOR (ordenado por importancia):")
        print(df[['player', 'density_change', 'density_pct', 'impact_score']
               ].to_string(index=False))

        return df

    def simulate_pass_redistribution(
        self,
        scenarios: List[Dict]
    ) -> pd.DataFrame:
        """
        Simula diferentes redistribuciones de pases y evalúa su impacto.

        Args:
            scenarios: Lista de escenarios con formato:
                [{'from': 'Jugador A', 'to': 'Jugador B', 'pct': 0.3}]

        Returns:
            DataFrame con resultados por escenario
        """
        print("\n SIMULANDO REDISTRIBUCIÓN DE PASES...")
        

        orig_density = nx.density(self.graph)
        results      = []

        for i, scenario in enumerate(scenarios):
            self.modifier.reset()
            self.modifier.redistribute_passes(
                from_player=scenario['from'],
                to_player=scenario['to'],
                pct=scenario.get('pct', 0.3)
            )

            G_mod       = self.modifier.modified_graph
            new_density = nx.density(G_mod)
            new_cluster = np.mean(
                list(nx.clustering(
                    G_mod.to_undirected(), weight='weight'
                ).values())
            )

            results.append({
                'scenario':      f"Escenario {i+1}",
                'from':          scenario['from'],
                'to':            scenario['to'],
                'pct':           scenario.get('pct', 0.3),
                'new_density':   new_density,
                'density_change':new_density - orig_density,
                'new_clustering':new_cluster,
            })

        self.modifier.reset()

        df = pd.DataFrame(results)
        print(f"\n RESULTADOS POR ESCENARIO:")
        print(df.to_string(index=False))

        return df

    def plot_impact_analysis(
        self,
        impact_df: pd.DataFrame,
        title: str = 'Impacto de Eliminación de Jugadores'
    ):
        """
        Visualiza el análisis de impacto de eliminación de jugadores.

        Args:
            impact_df: DataFrame con resultados de simulate_player_removal
            title: Título del gráfico
        """
        if impact_df.empty:
            print("  No hay datos para visualizar")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico 1: Cambio en densidad
        colors = [
            '#A50044' if v < 0 else '#004D98'
            for v in impact_df['density_change']
        ]
        bars = axes[0].barh(
            impact_df['player'].apply(lambda x: x.split()[-1]),
            impact_df['density_change'],
            color=colors, edgecolor='black',
            linewidth=0.5, alpha=0.85
        )
        axes[0].axvline(x=0, color='black', linewidth=1)
        axes[0].set_xlabel('Cambio en Densidad', fontsize=11)
        axes[0].set_title(
            'Impacto en Densidad al Eliminar Jugador',
            fontsize=12, fontweight='bold'
        )
        axes[0].grid(axis='x', alpha=0.3)

        # Gráfico 2: Impact score
        axes[1].barh(
            impact_df['player'].apply(lambda x: x.split()[-1]),
            impact_df['impact_score'],
            color='#EDBB00', edgecolor='black',
            linewidth=0.5, alpha=0.85
        )
        axes[1].set_xlabel('Impact Score', fontsize=11)
        axes[1].set_title(
            'Importancia Táctica del Jugador',
            fontsize=12, fontweight='bold'
        )
        axes[1].grid(axis='x', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def full_tactical_report(
        self,
        top_n: int = 5
    ) -> Dict:
        """
        Genera un informe táctico completo con simulaciones.

        Args:
            top_n: Número de jugadores a analizar

        Returns:
            Diccionario con resultados completos
        """
        
        print(" INFORME TÁCTICO COMPLETO")
        

        # Identificar jugadores más importantes
        pagerank = nx.pagerank(self.graph, weight='weight')
        top_players = sorted(
            pagerank, key=pagerank.get, reverse=True
        )[:top_n]

        print(f"\n Analizando top {top_n} jugadores por PageRank:")
        for i, p in enumerate(top_players, 1):
            print(f"   {i}. {p}: {pagerank[p]:.4f}")

        # Simular impacto de eliminación
        impact_df = self.simulate_player_removal(top_players)

        # Visualizar
        self.plot_impact_analysis(
            impact_df,
            title='Análisis de Impacto Táctico — Top Jugadores'
        )

        # Escenario de redistribución más relevante
        if len(top_players) >= 2:
            scenarios = [{
                'from': top_players[0],
                'to':   top_players[1],
                'pct':  0.3
            }]
            redist_df = self.simulate_pass_redistribution(scenarios)
        else:
            redist_df = pd.DataFrame()

        
        print(f" INFORME COMPLETADO")
        

        return {
            'top_players': top_players,
            'impact_df':   impact_df,
            'redist_df':   redist_df
        }