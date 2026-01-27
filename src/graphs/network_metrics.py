"""
Módulo para cálculo de métricas de red en grafos de pases.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class NetworkMetrics:
    """
    Clase para calcular y analizar métricas de redes de pases.
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el calculador de métricas.
        
        Args:
            graph: Grafo dirigido de NetworkX
        """
        self.graph = graph
        self.metrics = {}
    
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula múltiples métricas de centralidad.
        
        Returns:
            Diccionario con métricas de centralidad por jugador
        """
        print("\n📊 Calculando métricas de centralidad...")
        
        # Degree Centrality (in/out)
        in_degree = nx.in_degree_centrality(self.graph)
        out_degree = nx.out_degree_centrality(self.graph)
        
        # Betweenness Centrality (intermediación)
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        
        # Closeness Centrality (cercanía)
        try:
            closeness = nx.closeness_centrality(self.graph, distance='weight')
        except:
            closeness = nx.closeness_centrality(self.graph)
        
        # PageRank (importancia en la red)
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Eigenvector Centrality (influencia)
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, weight='weight', max_iter=1000)
        except:
            print("⚠️  Eigenvector centrality no convergió, usando alternativa")
            eigenvector = {node: 0 for node in self.graph.nodes()}
        
        self.metrics['centrality'] = {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'betweenness': betweenness,
            'closeness': closeness,
            'pagerank': pagerank,
            'eigenvector': eigenvector
        }
        
        print("✅ Métricas de centralidad calculadas")
        return self.metrics['centrality']
    
    def calculate_clustering_coefficient(self) -> Dict[str, float]:
        """
        Calcula coeficiente de clustering (agrupamiento).
        
        Returns:
            Diccionario con coeficiente de clustering por jugador
        """
        print("📊 Calculando coeficiente de clustering...")
        
        # Convertir a grafo no dirigido para clustering
        G_undirected = self.graph.to_undirected()
        clustering = nx.clustering(G_undirected, weight='weight')
        
        self.metrics['clustering'] = clustering
        
        avg_clustering = sum(clustering.values()) / len(clustering)
        print(f"✅ Clustering promedio: {avg_clustering:.4f}")
        
        return clustering
    
    def calculate_degree_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Calcula distribución de grados (in/out).
        
        Returns:
            Diccionario con grados in/out por jugador
        """
        print("📊 Calculando distribución de grados...")
        
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        total_degree = {node: in_degree[node] + out_degree[node] 
                       for node in self.graph.nodes()}
        
        self.metrics['degree'] = {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': total_degree
        }
        
        print(f"✅ Grado promedio: {sum(total_degree.values()) / len(total_degree):.2f}")
        
        return self.metrics['degree']
    
    def calculate_graph_density(self) -> float:
        """
        Calcula densidad del grafo.
        
        Returns:
            Densidad (0-1)
        """
        density = nx.density(self.graph)
        self.metrics['density'] = density
        
        print(f"📊 Densidad del grafo: {density:.4f}")
        return density
    
    def identify_hubs(self, n: int = 5, metric: str = 'pagerank') -> List[str]:
        """
        Identifica los n jugadores más importantes (hubs).
        
        Args:
            n: Número de hubs a identificar
            metric: Métrica a usar ('pagerank', 'betweenness', 'out_degree')
            
        Returns:
            Lista de jugadores hub
        """
        if 'centrality' not in self.metrics:
            self.calculate_centrality_metrics()
        
        if metric == 'out_degree':
            if 'degree' not in self.metrics:
                self.calculate_degree_distribution()
            scores = self.metrics['degree']['out_degree']
        else:
            scores = self.metrics['centrality'][metric]
        
        hubs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        print(f"\n🎯 Top {n} hubs (por {metric}):")
        for i, (player, score) in enumerate(hubs, 1):
            print(f"   {i}. {player}: {score:.4f}")
        
        return [player for player, _ in hubs]
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calcula todas las métricas disponibles.
        
        Returns:
            Diccionario completo con todas las métricas
        """
        print("\n" + "="*60)
        print("🔬 CALCULANDO TODAS LAS MÉTRICAS DE RED")
        print("="*60)
        
        self.calculate_centrality_metrics()
        self.calculate_clustering_coefficient()
        self.calculate_degree_distribution()
        self.calculate_graph_density()
        
        print("\n" + "="*60)
        print("✅ TODAS LAS MÉTRICAS CALCULADAS")
        print("="*60 + "\n")
        
        return self.metrics
    
    def get_player_metrics(self, player: str) -> Dict:
        """
        Obtiene todas las métricas de un jugador específico.
        
        Args:
            player: Nombre del jugador
            
        Returns:
            Diccionario con métricas del jugador
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        player_metrics = {
            'in_degree': self.metrics['degree']['in_degree'].get(player, 0),
            'out_degree': self.metrics['degree']['out_degree'].get(player, 0),
            'total_degree': self.metrics['degree']['total_degree'].get(player, 0),
            'in_degree_centrality': self.metrics['centrality']['in_degree'].get(player, 0),
            'out_degree_centrality': self.metrics['centrality']['out_degree'].get(player, 0),
            'betweenness': self.metrics['centrality']['betweenness'].get(player, 0),
            'closeness': self.metrics['centrality']['closeness'].get(player, 0),
            'pagerank': self.metrics['centrality']['pagerank'].get(player, 0),
            'eigenvector': self.metrics['centrality']['eigenvector'].get(player, 0),
            'clustering': self.metrics['clustering'].get(player, 0),
        }
        
        return player_metrics
    
    def create_metrics_dataframe(self) -> pd.DataFrame:
        """
        Crea un DataFrame con todas las métricas por jugador.
        
        Returns:
            DataFrame con métricas
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        players = list(self.graph.nodes())
        
        data = []
        for player in players:
            player_data = {'player': player}
            player_data.update(self.get_player_metrics(player))
            data.append(player_data)
        
        df = pd.DataFrame(data)
        df = df.sort_values('pagerank', ascending=False).reset_index(drop=True)
        
        return df
    
    def visualize_centrality_comparison(self, metrics: List[str] = None, top_n: int = 10):
        """
        Visualiza comparación de métricas de centralidad.
        
        Args:
            metrics: Lista de métricas a comparar
            top_n: Número de jugadores top a mostrar
        """
        if metrics is None:
            metrics = ['pagerank', 'betweenness', 'out_degree_centrality']
        
        if not self.metrics:
            self.calculate_all_metrics()
        
        df = self.create_metrics_dataframe()
        df_top = df.head(top_n)
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 8))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in df_top.columns:
                df_plot = df_top.sort_values(metric, ascending=True)
                axes[idx].barh(df_plot['player'], df_plot[metric], color=f'C{idx}')
                axes[idx].set_xlabel(metric.replace('_', ' ').title())
                axes[idx].set_title(f'Top {top_n} - {metric.replace("_", " ").title()}')
                axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_degree_distribution(self):
        """
        Visualiza distribución de grados.
        """
        if 'degree' not in self.metrics:
            self.calculate_degree_distribution()
        
        in_degrees = list(self.metrics['degree']['in_degree'].values())
        out_degrees = list(self.metrics['degree']['out_degree'].values())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # In-degree
        axes[0].hist(in_degrees, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('In-Degree')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de In-Degree (Pases Recibidos)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Out-degree
        axes[1].hist(out_degrees, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Out-Degree')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de Out-Degree (Pases Realizados)')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_correlation_matrix(self):
        """
        Visualiza matriz de correlación entre métricas.
        """
        df = self.create_metrics_dataframe()
        
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlación entre Métricas de Red', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def export_metrics(self, filename: str = 'data/processed/network_metrics.csv'):
        """
        Exporta métricas a CSV.
        
        Args:
            filename: Ruta del archivo de salida
        """
        df = self.create_metrics_dataframe()
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        print(f"💾 Métricas exportadas a: {filename}")
    
    def print_summary(self):
        """
        Imprime resumen completo de métricas.
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        df = self.create_metrics_dataframe()
        
        print("\n📊 RESUMEN DE MÉTRICAS DE RED")
        print("="*60)
        print(f"Número de jugadores: {len(df)}")
        print(f"Densidad del grafo: {self.metrics['density']:.4f}")
        print(f"\nGrados promedio:")
        print(f"  - In-degree: {df['in_degree'].mean():.2f}")
        print(f"  - Out-degree: {df['out_degree'].mean():.2f}")
        print(f"\nClustering promedio: {df['clustering'].mean():.4f}")
        print(f"\nTop 5 jugadores (PageRank):")
        for i, row in df.head(5).iterrows():
            print(f"  {i+1}. {row['player']}: {row['pagerank']:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando NetworkMetrics...\n")
    
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data.loader import StatsBombLoader
    from src.data.preprocessing import PassPreprocessor
    from src.graphs.pass_graph import PassGraph
    
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
    
    # Calcular métricas
    metrics_calculator = NetworkMetrics(G)
    metrics_calculator.calculate_all_metrics()
    
    # Resumen
    metrics_calculator.print_summary()
    
    # Identificar hubs
    metrics_calculator.identify_hubs(n=5, metric='pagerank')
    metrics_calculator.identify_hubs(n=5, metric='betweenness')
    
    # DataFrame de métricas
    print("\n📋 DataFrame de métricas (top 10):")
    df_metrics = metrics_calculator.create_metrics_dataframe()
    print(df_metrics.head(10))
    
    # Exportar
    metrics_calculator.export_metrics()
    
    # Visualizaciones
    print("\n📊 Generando visualizaciones...")
    metrics_calculator.visualize_centrality_comparison()
    metrics_calculator.visualize_degree_distribution()
    metrics_calculator.visualize_correlation_matrix()