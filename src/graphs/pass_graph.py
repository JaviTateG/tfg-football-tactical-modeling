"""
Módulo para construcción de grafos de pases.
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple


class PassGraph:
    """
    Clase para construir y gestionar grafos de pases entre jugadores.
    """
    
    def __init__(self, weight_type: str = 'frequency'):
        """
        Inicializa el constructor de grafos.
        
        Args:
            weight_type: Tipo de peso para las aristas
                - 'frequency': número de pases
                - 'success_rate': tasa de éxito (0-1)
                - 'distance': distancia promedio
        """
        self.weight_type = weight_type
        self.graph = None
        
    def build_graph(self, passes: pd.DataFrame, team: Optional[str] = None) -> nx.DiGraph:
        """
        Construye un grafo dirigido de pases.
        
        Args:
            passes: DataFrame con pases preprocesados
            team: Filtrar por equipo específico (opcional)
            
        Returns:
            Grafo dirigido de NetworkX
        """
        print(f"\n🔨 Construyendo grafo de pases (peso: {self.weight_type})...")
        
        # Filtrar por equipo si se especifica
        if team:
            passes = passes[passes['team'] == team].copy()
            print(f"   Filtrando equipo: {team}")
        
        # Verificar columnas necesarias
        required_cols = ['player', 'pass_recipient']
        if not all(col in passes.columns for col in required_cols):
            raise ValueError(f"Faltan columnas requeridas: {required_cols}")
        
        # Crear grafo dirigido
        G = nx.DiGraph()
        
        # Agrupar pases por (jugador, receptor)
        if self.weight_type == 'frequency':
            edge_data = self._calculate_frequency_weights(passes)
        elif self.weight_type == 'success_rate':
            edge_data = self._calculate_success_weights(passes)
        elif self.weight_type == 'distance':
            edge_data = self._calculate_distance_weights(passes)
        else:
            raise ValueError(f"Tipo de peso inválido: {self.weight_type}")
        
        # Añadir aristas al grafo
        for (player, recipient), weight in edge_data.items():
            G.add_edge(player, recipient, weight=weight)
        
        # Añadir atributos a nodos (equipo, posición si está disponible)
        self._add_node_attributes(G, passes)
        
        self.graph = G
        
        print(f"✅ Grafo construido:")
        print(f"   - Nodos (jugadores): {G.number_of_nodes()}")
        print(f"   - Aristas (conexiones): {G.number_of_edges()}")
        
        return G
    
    def _calculate_frequency_weights(self, passes: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calcula pesos basados en frecuencia de pases.
        """
        connections = passes.groupby(['player', 'pass_recipient']).size()
        return {(player, recipient): count 
                for (player, recipient), count in connections.items()}
    
    def _calculate_success_weights(self, passes: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calcula pesos basados en tasa de éxito.
        """
        if 'pass_success' not in passes.columns:
            print("⚠️  'pass_success' no disponible, usando frecuencia")
            return self._calculate_frequency_weights(passes)
        
        grouped = passes.groupby(['player', 'pass_recipient'])['pass_success'].agg(['sum', 'count'])
        success_rate = (grouped['sum'] / grouped['count']).to_dict()
        
        return {(player, recipient): rate 
                for (player, recipient), rate in success_rate.items()}
    
    def _calculate_distance_weights(self, passes: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """
        Calcula pesos basados en distancia promedio.
        """
        if 'pass_distance' not in passes.columns:
            print("⚠️  'pass_distance' no disponible, usando frecuencia")
            return self._calculate_frequency_weights(passes)
        
        avg_distance = passes.groupby(['player', 'pass_recipient'])['pass_distance'].mean()
        return {(player, recipient): dist 
                for (player, recipient), dist in avg_distance.items()}
    
    def _add_node_attributes(self, G: nx.DiGraph, passes: pd.DataFrame):
        """
        Añade atributos a los nodos del grafo.
        """
        # Equipo de cada jugador
        player_team = passes.groupby('player')['team'].first().to_dict()
        nx.set_node_attributes(G, player_team, 'team')
        
        # Número total de pases realizados
        passes_made = passes.groupby('player').size().to_dict()
        nx.set_node_attributes(G, passes_made, 'passes_made')
        
        # Número total de pases recibidos
        passes_received = passes.groupby('pass_recipient').size().to_dict()
        nx.set_node_attributes(G, passes_received, 'passes_received')
    
    def get_top_connections(self, n: int = 10) -> pd.DataFrame:
        """
        Obtiene las n conexiones más fuertes.
        
        Args:
            n: Número de conexiones a retornar
            
        Returns:
            DataFrame con las top conexiones
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo")
        
        edges = [(u, v, data['weight']) 
                 for u, v, data in self.graph.edges(data=True)]
        
        edges_df = pd.DataFrame(edges, columns=['from', 'to', 'weight'])
        edges_df = edges_df.sort_values('weight', ascending=False).head(n)
        
        return edges_df
    
    def get_isolated_players(self) -> List[str]:
        """
        Encuentra jugadores sin conexiones (nodos aislados).
        
        Returns:
            Lista de jugadores aislados
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo")
        
        isolated = list(nx.isolates(self.graph))
        return isolated
    
    def visualize(self, 
                  figsize: Tuple[int, int] = (12, 8),
                  node_size: int = 1000,
                  with_labels: bool = True,
                  title: Optional[str] = None):
        """
        Visualiza el grafo de pases.
        
        Args:
            figsize: Tamaño de la figura
            node_size: Tamaño de los nodos
            with_labels: Mostrar nombres de jugadores
            title: Título del gráfico
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo")
        
        plt.figure(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)
        
        # Obtener pesos para ajustar grosor de aristas
        edges = self.graph.edges()
        weights = [self.graph[u][v]['weight'] for u, v in edges]
        
        # Normalizar pesos para visualización
        max_weight = max(weights) if weights else 1
        widths = [3 * (w / max_weight) for w in weights]
        
        # Dibujar grafo
        nx.draw_networkx_nodes(self.graph, pos, 
                               node_color='lightblue', 
                               node_size=node_size,
                               alpha=0.9)
        
        nx.draw_networkx_edges(self.graph, pos, 
                               width=widths,
                               alpha=0.6,
                               edge_color='gray',
                               arrows=True,
                               arrowsize=20,
                               arrowstyle='->')
        
        if with_labels:
            nx.draw_networkx_labels(self.graph, pos, 
                                    font_size=8,
                                    font_weight='bold')
        
        plt.title(title or f"Red de Pases (peso: {self.weight_type})", 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_graph(self, filename: str):
        """
        Guarda el grafo en formato GraphML.
        
        Args:
            filename: Ruta del archivo de salida
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo")
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        nx.write_graphml(self.graph, filename)
        print(f"💾 Grafo guardado en: {filename}")
    
    def load_graph(self, filename: str):
        """
        Carga un grafo desde archivo GraphML.
        
        Args:
            filename: Ruta del archivo
        """
        self.graph = nx.read_graphml(filename)
        print(f"📂 Grafo cargado desde: {filename}")
        return self.graph
    
    def get_graph_summary(self) -> Dict:
        """
        Obtiene resumen estadístico del grafo.
        
        Returns:
            Diccionario con estadísticas
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo")
        
        summary = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
        }
        
        return summary
    
    def print_summary(self):
        """
        Imprime resumen del grafo.
        """
        summary = self.get_graph_summary()
        
        print("\n📊 RESUMEN DEL GRAFO")
        print("="*60)
        print(f"Nodos (jugadores): {summary['num_nodes']}")
        print(f"Aristas (conexiones): {summary['num_edges']}")
        print(f"Densidad: {summary['density']:.4f}")
        print(f"Grado promedio: {summary['avg_degree']:.2f}")
        print(f"Conectado: {'Sí' if summary['is_connected'] else 'No'}")
        print(f"Componentes: {summary['num_components']}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando PassGraph...\n")
    
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data.loader import StatsBombLoader
    from src.data.preprocessing import PassPreprocessor
    
    # Cargar y preprocesar datos
    loader = StatsBombLoader()
    matches = loader.get_matches(competition_id=11, season_id=90)
    
    barcelona_matches = matches[
        (matches['home_team'] == 'Barcelona') | 
        (matches['away_team'] == 'Barcelona')
    ]
    
    match_id = barcelona_matches.iloc[0]['match_id']
    match_info = barcelona_matches.iloc[0]
    
    print(f"🎯 Partido: {match_info['home_team']} vs {match_info['away_team']}\n")
    
    events, _ = loader.load_match_data(match_id)
    
    preprocessor = PassPreprocessor()
    passes_clean = preprocessor.preprocess_full_pipeline(events)
    
    # Construir grafo
    pass_graph = PassGraph(weight_type='frequency')
    G = pass_graph.build_graph(passes_clean, team='Barcelona')
    
    # Resumen
    pass_graph.print_summary()
    
    # Top conexiones
    print("🔗 Top 10 conexiones más fuertes:")
    print(pass_graph.get_top_connections(10))
    
    # Guardar
    pass_graph.save_graph('data/processed/pass_graph.graphml')
    
    # Visualizar
    pass_graph.visualize(title=f"Red de Pases - Barcelona")