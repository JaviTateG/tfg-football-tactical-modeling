"""
Módulo para construcción y análisis de grafos dinámicos (temporales).
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


class DynamicPassGraph:
    """
    Clase para construir y analizar grafos de pases dinámicos (por ventanas temporales).
    """
    
    def __init__(self, window_size: int = 5, step_size: int = 1, weight_type: str = 'frequency'):
        """
        Inicializa el constructor de grafos dinámicos.
        
        Args:
            window_size: Tamaño de la ventana temporal en minutos
            step_size: Paso entre ventanas en minutos (1 = ventanas deslizantes)
            weight_type: Tipo de peso para las aristas
        """
        self.window_size = window_size
        self.step_size = step_size
        self.weight_type = weight_type
        self.graphs = []  # Lista de (tiempo_inicio, tiempo_fin, grafo)
        self.metrics_timeline = None
    
    def create_temporal_windows(self, passes: pd.DataFrame) -> List[Tuple[float, float, pd.DataFrame]]:
        """
        Divide los pases en ventanas temporales.
        
        Args:
            passes: DataFrame con pases preprocesados (debe tener 'timestamp')
            
        Returns:
            Lista de tuplas (inicio, fin, pases_en_ventana)
        """
        if 'timestamp' not in passes.columns:
            raise ValueError("DataFrame debe contener columna 'timestamp'")
        
        min_time = passes['timestamp'].min()
        max_time = passes['timestamp'].max()
        
        windows = []
        current_start = min_time
        
        while current_start < max_time:
            window_start = current_start
            window_end = current_start + (self.window_size * 60)  # convertir a segundos
            
            # Filtrar pases en esta ventana
            window_passes = passes[
                (passes['timestamp'] >= window_start) & 
                (passes['timestamp'] < window_end)
            ].copy()
            
            if len(window_passes) > 0:
                windows.append((window_start, window_end, window_passes))
            
            current_start += (self.step_size * 60)
        
        print(f"✅ Creadas {len(windows)} ventanas temporales")
        print(f"   - Tamaño de ventana: {self.window_size} min")
        print(f"   - Paso: {self.step_size} min")
        
        return windows
    
    def build_dynamic_graphs(self, passes: pd.DataFrame, team: Optional[str] = None) -> List[Tuple[float, float, nx.DiGraph]]:
        """
        Construye grafos para cada ventana temporal.
        
        Args:
            passes: DataFrame con pases preprocesados
            team: Filtrar por equipo específico (opcional)
            
        Returns:
            Lista de tuplas (inicio, fin, grafo)
        """
        print(f"\n🔨 Construyendo grafos dinámicos...")
        
        if team:
            passes = passes[passes['team'] == team].copy()
            print(f"   Filtrando equipo: {team}")
        
        # Crear ventanas temporales
        windows = self.create_temporal_windows(passes)
        
        # Construir grafo para cada ventana
        from src.graphs.pass_graph import PassGraph
        
        self.graphs = []
        
        for window_start, window_end, window_passes in windows:
            if len(window_passes) >= 3:  # Mínimo de pases para construir grafo
                pg = PassGraph(weight_type=self.weight_type)
                G = pg.build_graph(window_passes, team=None)  # Ya filtrado
                
                self.graphs.append((window_start, window_end, G))
        
        print(f"✅ {len(self.graphs)} grafos construidos")
        
        return self.graphs
    
    def calculate_temporal_metrics(self) -> pd.DataFrame:
        """
        Calcula métricas para cada ventana temporal.
        
        Returns:
            DataFrame con métricas temporales
        """
        if not self.graphs:
            raise ValueError("Primero debe construir los grafos dinámicos")
        
        print("\n📊 Calculando métricas temporales...")
        
        from src.graphs.network_metrics import NetworkMetrics
        
        temporal_data = []
        
        for window_start, window_end, G in self.graphs:
            window_center = (window_start + window_end) / 2 / 60  # en minutos
            
            # Calcular métricas básicas
            metrics_calc = NetworkMetrics(G)
            
            window_metrics = {
                'window_start': window_start / 60,
                'window_end': window_end / 60,
                'window_center': window_center,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            }
            
            # Calcular métricas de centralidad
            try:
                centrality = metrics_calc.calculate_centrality_metrics()
                
                # Promedios de centralidad
                window_metrics['avg_betweenness'] = np.mean(list(centrality['betweenness'].values()))
                window_metrics['avg_closeness'] = np.mean(list(centrality['closeness'].values()))
                window_metrics['avg_pagerank'] = np.mean(list(centrality['pagerank'].values()))
                
                # Clustering
                clustering = metrics_calc.calculate_clustering_coefficient()
                window_metrics['avg_clustering'] = np.mean(list(clustering.values()))
                
            except Exception as e:
                print(f"⚠️  Error calculando métricas para ventana {window_center:.1f}min: {e}")
                window_metrics['avg_betweenness'] = 0
                window_metrics['avg_closeness'] = 0
                window_metrics['avg_pagerank'] = 0
                window_metrics['avg_clustering'] = 0
            
            temporal_data.append(window_metrics)
        
        self.metrics_timeline = pd.DataFrame(temporal_data)
        
        print(f"✅ Métricas calculadas para {len(self.metrics_timeline)} ventanas")
        
        return self.metrics_timeline
    
    def detect_tactical_changes(self, metric: str = 'density', threshold: float = 0.15) -> List[float]:
        """
        Detecta cambios tácticos significativos basados en variaciones de métricas.
        
        Args:
            metric: Métrica a analizar
            threshold: Umbral de cambio relativo para considerar significativo
            
        Returns:
            Lista de momentos (en minutos) donde se detectaron cambios
        """
        if self.metrics_timeline is None:
            self.calculate_temporal_metrics()
        
        if metric not in self.metrics_timeline.columns:
            raise ValueError(f"Métrica '{metric}' no disponible")
        
        values = self.metrics_timeline[metric].values
        times = self.metrics_timeline['window_center'].values
        
        # Calcular cambios relativos
        changes = np.abs(np.diff(values) / (values[:-1] + 1e-10))
        
        # Identificar cambios significativos
        significant_changes = np.where(changes > threshold)[0]
        change_times = times[significant_changes + 1]
        
        print(f"\n🔍 Cambios tácticos detectados (métrica: {metric}, umbral: {threshold}):")
        print(f"   Total: {len(change_times)} cambios")
        
        for i, time in enumerate(change_times, 1):
            idx = np.where(times == time)[0][0]
            old_val = values[idx - 1]
            new_val = values[idx]
            change_pct = ((new_val - old_val) / old_val) * 100
            print(f"   {i}. Minuto {time:.1f}: {old_val:.4f} → {new_val:.4f} ({change_pct:+.1f}%)")
        
        return change_times.tolist()
    
    def get_player_temporal_importance(self, player: str, metric: str = 'pagerank') -> pd.DataFrame:
        """
        Analiza la evolución de la importancia de un jugador a lo largo del tiempo.
        
        Args:
            player: Nombre del jugador
            metric: Métrica de centralidad a usar
            
        Returns:
            DataFrame con importancia temporal del jugador
        """
        from src.graphs.network_metrics import NetworkMetrics
        
        player_timeline = []
        
        for window_start, window_end, G in self.graphs:
            if player not in G.nodes():
                continue
            
            window_center = (window_start + window_end) / 2 / 60
            
            metrics_calc = NetworkMetrics(G)
            centrality = metrics_calc.calculate_centrality_metrics()
            
            player_data = {
                'time': window_center,
                'metric_value': centrality[metric].get(player, 0)
            }
            
            player_timeline.append(player_data)
        
        df = pd.DataFrame(player_timeline)
        return df
    
    def visualize_temporal_evolution(self, metrics: List[str] = None):
        """
        Visualiza evolución temporal de métricas.
        
        Args:
            metrics: Lista de métricas a visualizar
        """
        if self.metrics_timeline is None:
            self.calculate_temporal_metrics()
        
        if metrics is None:
            metrics = ['density', 'avg_clustering', 'num_edges']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4*len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric not in self.metrics_timeline.columns:
                print(f"⚠️  Métrica '{metric}' no disponible")
                continue
            
            axes[idx].plot(self.metrics_timeline['window_center'], 
                          self.metrics_timeline[metric],
                          marker='o', linewidth=2, markersize=5, color=f'C{idx}')
            
            axes[idx].set_xlabel('Tiempo (minutos)')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'Evolución Temporal - {metric.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axvline(x=45, color='red', linestyle='--', alpha=0.5, label='Medio Tiempo')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_player_importance_evolution(self, players: List[str], metric: str = 'pagerank'):
        """
        Visualiza evolución de importancia de múltiples jugadores.
        
        Args:
            players: Lista de nombres de jugadores
            metric: Métrica de centralidad
        """
        plt.figure(figsize=(14, 6))
        
        for player in players:
            df = self.get_player_temporal_importance(player, metric)
            if len(df) > 0:
                plt.plot(df['time'], df['metric_value'], 
                        marker='o', linewidth=2, label=player)
        
        plt.xlabel('Tiempo (minutos)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Evolución de {metric.replace("_", " ").title()} por Jugador')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=45, color='red', linestyle='--', alpha=0.5, label='Medio Tiempo')
        plt.tight_layout()
        plt.show()
    
    def compare_halves(self) -> Dict:
        """
        Compara métricas entre primera y segunda mitad.
        
        Returns:
            Diccionario con comparación de mitades
        """
        if self.metrics_timeline is None:
            self.calculate_temporal_metrics()
        
        first_half = self.metrics_timeline[self.metrics_timeline['window_center'] <= 45]
        second_half = self.metrics_timeline[self.metrics_timeline['window_center'] > 45]
        
        comparison = {}
        
        metrics_to_compare = ['density', 'avg_clustering', 'num_edges', 'avg_betweenness']
        
        for metric in metrics_to_compare:
            if metric in self.metrics_timeline.columns:
                comparison[metric] = {
                    'first_half_avg': first_half[metric].mean(),
                    'second_half_avg': second_half[metric].mean(),
                    'difference': second_half[metric].mean() - first_half[metric].mean(),
                    'pct_change': ((second_half[metric].mean() - first_half[metric].mean()) / 
                                  first_half[metric].mean() * 100) if first_half[metric].mean() != 0 else 0
                }
        
        print("\n📊 COMPARACIÓN ENTRE MITADES")
        print("="*60)
        for metric, values in comparison.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Primera mitad: {values['first_half_avg']:.4f}")
            print(f"  Segunda mitad: {values['second_half_avg']:.4f}")
            print(f"  Diferencia: {values['difference']:+.4f} ({values['pct_change']:+.2f}%)")
        print("="*60 + "\n")
        
        return comparison
    
    def export_temporal_metrics(self, filename: str = 'data/processed/temporal_metrics.csv'):
        """
        Exporta métricas temporales a CSV.
        
        Args:
            filename: Ruta del archivo de salida
        """
        if self.metrics_timeline is None:
            self.calculate_temporal_metrics()
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.metrics_timeline.to_csv(filename, index=False)
        
        print(f"💾 Métricas temporales exportadas a: {filename}")
    
    def print_summary(self):
        """
        Imprime resumen de grafos dinámicos.
        """
        if not self.graphs:
            print("⚠️  No hay grafos construidos")
            return
        
        print("\n📊 RESUMEN DE GRAFOS DINÁMICOS")
        print("="*60)
        print(f"Ventanas temporales: {len(self.graphs)}")
        print(f"Tamaño de ventana: {self.window_size} minutos")
        print(f"Paso entre ventanas: {self.step_size} minutos")
        print(f"Tipo de peso: {self.weight_type}")
        
        if self.metrics_timeline is not None:
            print(f"\nMétricas disponibles:")
            for col in self.metrics_timeline.columns:
                if col not in ['window_start', 'window_end', 'window_center']:
                    avg = self.metrics_timeline[col].mean()
                    print(f"  - {col}: promedio = {avg:.4f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando DynamicPassGraph...\n")
    
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
    
    # Construir grafos dinámicos
    dynamic_graph = DynamicPassGraph(window_size=5, step_size=2, weight_type='frequency')
    graphs = dynamic_graph.build_dynamic_graphs(passes_clean, team='Barcelona')
    
    # Resumen
    dynamic_graph.print_summary()
    
    # Calcular métricas temporales
    temporal_metrics = dynamic_graph.calculate_temporal_metrics()
    print("\n📋 Primeras ventanas temporales:")
    print(temporal_metrics.head(10))
    
    # Detectar cambios tácticos
    changes = dynamic_graph.detect_tactical_changes(metric='density', threshold=0.15)
    
    # Comparar mitades
    comparison = dynamic_graph.compare_halves()
    
    # Exportar
    dynamic_graph.export_temporal_metrics()
    
    # Visualizaciones
    print("\n📊 Generando visualizaciones...")
    dynamic_graph.visualize_temporal_evolution(['density', 'avg_clustering', 'num_edges'])