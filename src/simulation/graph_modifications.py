"""
Módulo para modificación de grafos de pases.
Permite simular cambios tácticos alterando la estructura del grafo.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import copy


class GraphModifications:
    """
    Clase para modificar grafos de pases y simular
    escenarios tácticos alternativos.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Args:
            graph: Grafo dirigido original de pases
        """
        self.original_graph = graph
        self.modified_graph = copy.deepcopy(graph)

    def reset(self):
        """Restaura el grafo a su estado original."""
        self.modified_graph = copy.deepcopy(self.original_graph)
        print(" Grafo restaurado al estado original")

    def remove_player(self, player: str) -> nx.DiGraph:
        """
        Elimina un jugador del grafo (simula lesión o sustitución).

        Args:
            player: Nombre del jugador a eliminar

        Returns:
            Grafo modificado
        """
        if player not in self.modified_graph.nodes:
            print(f"  {player} no encontrado en el grafo")
            return self.modified_graph

        self.modified_graph.remove_node(player)
        print(f" Jugador eliminado: {player}")
        print(f"   Nodos restantes: {self.modified_graph.number_of_nodes()}")
        print(f"   Aristas restantes: {self.modified_graph.number_of_edges()}")

        return self.modified_graph

    def boost_player(
        self,
        player: str,
        factor: float = 1.5
    ) -> nx.DiGraph:
        """
        Aumenta el peso de todas las conexiones de un jugador.
        Simula un jugador en mejor forma o con más protagonismo.

        Args:
            player: Nombre del jugador
            factor: Factor multiplicador del peso

        Returns:
            Grafo modificado
        """
        if player not in self.modified_graph.nodes:
            print(f"  {player} no encontrado en el grafo")
            return self.modified_graph

        modified = 0
        for u, v, data in list(self.modified_graph.edges(data=True)):
            if u == player or v == player:
                self.modified_graph[u][v]['weight'] = (
                    data['weight'] * factor
                )
                modified += 1

        print(f" Jugador potenciado: {player} (factor {factor}x)")
        print(f"   Conexiones modificadas: {modified}")

        return self.modified_graph

    def reduce_player(
        self,
        player: str,
        factor: float = 0.5
    ) -> nx.DiGraph:
        """
        Reduce el peso de todas las conexiones de un jugador.
        Simula un jugador con menos protagonismo o marcado.

        Args:
            player: Nombre del jugador
            factor: Factor reductor del peso

        Returns:
            Grafo modificado
        """
        return self.boost_player(player, factor)

    def redistribute_passes(
        self,
        from_player: str,
        to_player: str,
        pct: float = 0.3
    ) -> nx.DiGraph:
        """
        Redistribuye un porcentaje de pases de un jugador a otro.
        Simula un cambio en la circulación del balón.

        Args:
            from_player: Jugador que cede protagonismo
            to_player: Jugador que recibe protagonismo
            pct: Porcentaje de pases a redistribuir (0-1)

        Returns:
            Grafo modificado
        """
        if from_player not in self.modified_graph.nodes:
            print(f"  {from_player} no encontrado")
            return self.modified_graph

        if to_player not in self.modified_graph.nodes:
            print(f"  {to_player} no encontrado")
            return self.modified_graph

        modified = 0
        for u, v, data in list(self.modified_graph.edges(data=True)):
            if u == from_player:
                transfer = data['weight'] * pct
                self.modified_graph[u][v]['weight'] -= transfer

                if self.modified_graph.has_edge(to_player, v):
                    self.modified_graph[to_player][v]['weight'] += transfer
                else:
                    self.modified_graph.add_edge(
                        to_player, v, weight=transfer
                    )
                modified += 1

        print(f" Redistribución de pases:")
        print(f"   {from_player} → {to_player} ({pct*100:.0f}% de pases)")
        print(f"   Conexiones afectadas: {modified}")

        return self.modified_graph

    def add_connection(
        self,
        player1: str,
        player2: str,
        weight: float = 5.0
    ) -> nx.DiGraph:
        """
        Añade o refuerza una conexión entre dos jugadores.

        Args:
            player1: Jugador origen
            player2: Jugador destino
            weight: Peso de la nueva conexión

        Returns:
            Grafo modificado
        """
        if self.modified_graph.has_edge(player1, player2):
            self.modified_graph[player1][player2]['weight'] += weight
            print(f"Conexión reforzada: {player1} → {player2}")
        else:
            self.modified_graph.add_edge(player1, player2, weight=weight)
            print(f" Nueva conexión añadida: {player1} → {player2}")

        return self.modified_graph

    def compare_graphs(self) -> Dict:
        """
        Compara las métricas del grafo original con el modificado.

        Returns:
            Diccionario con comparativa de métricas
        """
        def get_metrics(G):
            return {
                'nodes':       G.number_of_nodes(),
                'edges':       G.number_of_edges(),
                'density':     nx.density(G),
                'avg_pagerank': np.mean(
                    list(nx.pagerank(G, weight='weight').values())
                ),
                'avg_betweenness': np.mean(
                    list(nx.betweenness_centrality(
                        G, weight='weight'
                    ).values())
                ),
                'avg_clustering': np.mean(
                    list(nx.clustering(
                        G.to_undirected(), weight='weight'
                    ).values())
                ),
            }

        orig = get_metrics(self.original_graph)
        mod  = get_metrics(self.modified_graph)

        comparison = {}
        for key in orig:
            comparison[key] = {
                'original':  orig[key],
                'modified':  mod[key],
                'diff':      mod[key] - orig[key],
                'pct_change': (
                    (mod[key] - orig[key]) / orig[key] * 100
                ) if orig[key] != 0 else 0
            }

        print("\n COMPARATIVA ORIGINAL vs MODIFICADO:")
        
        for key, vals in comparison.items():
            print(f"\n{key}:")
            print(f"   Original:  {vals['original']:.4f}")
            print(f"   Modificado:{vals['modified']:.4f}")
            print(f"   Cambio:    {vals['diff']:+.4f} "
                  f"({vals['pct_change']:+.1f}%)")
        

        return comparison