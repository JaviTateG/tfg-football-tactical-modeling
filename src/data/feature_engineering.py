"""
Módulo para extracción de características (features) para Machine Learning.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from src.graphs.pass_graph import PassGraph
from src.graphs.network_metrics import NetworkMetrics
from src.graphs.dynamic_graph import DynamicPassGraph


class FeatureExtractor:
    """
    Clase para extraer características de grafos de pases y datos temporales.
    """
    
    def __init__(self):
        """
        Inicializa el extractor de features.
        """
        self.features = None
        self.feature_names = []
    
    def extract_graph_features(self, G: nx.DiGraph, prefix: str = '') -> Dict[str, float]:
        """
        Extrae características de un grafo de pases.
        
        Args:
            G: Grafo de NetworkX
            prefix: Prefijo para nombres de features
            
        Returns:
            Diccionario con features del grafo
        """
        metrics = NetworkMetrics(G)
        
        features = {}
        
        # Métricas básicas del grafo
        features[f'{prefix}num_nodes'] = G.number_of_nodes()
        features[f'{prefix}num_edges'] = G.number_of_edges()
        features[f'{prefix}density'] = nx.density(G)
        
        # Grados
        degree_dict = dict(G.degree())
        if degree_dict:
            features[f'{prefix}avg_degree'] = np.mean(list(degree_dict.values()))
            features[f'{prefix}max_degree'] = np.max(list(degree_dict.values()))
            features[f'{prefix}min_degree'] = np.min(list(degree_dict.values()))
            features[f'{prefix}std_degree'] = np.std(list(degree_dict.values()))
        else:
            features[f'{prefix}avg_degree'] = 0
            features[f'{prefix}max_degree'] = 0
            features[f'{prefix}min_degree'] = 0
            features[f'{prefix}std_degree'] = 0
        
        # In-degree y Out-degree
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        if in_degree:
            features[f'{prefix}avg_in_degree'] = np.mean(list(in_degree.values()))
            features[f'{prefix}max_in_degree'] = np.max(list(in_degree.values()))
        else:
            features[f'{prefix}avg_in_degree'] = 0
            features[f'{prefix}max_in_degree'] = 0
        
        if out_degree:
            features[f'{prefix}avg_out_degree'] = np.mean(list(out_degree.values()))
            features[f'{prefix}max_out_degree'] = np.max(list(out_degree.values()))
        else:
            features[f'{prefix}avg_out_degree'] = 0
            features[f'{prefix}max_out_degree'] = 0
        
        # Clustering
        clustering = metrics.calculate_clustering_coefficient()
        if clustering:
            features[f'{prefix}avg_clustering'] = np.mean(list(clustering.values()))
            features[f'{prefix}max_clustering'] = np.max(list(clustering.values()))
            features[f'{prefix}std_clustering'] = np.std(list(clustering.values()))
        else:
            features[f'{prefix}avg_clustering'] = 0
            features[f'{prefix}max_clustering'] = 0
            features[f'{prefix}std_clustering'] = 0
        
        # Centralidad
        try:
            centrality = metrics.calculate_centrality_metrics()
            
            # Betweenness
            betweenness = list(centrality['betweenness'].values())
            features[f'{prefix}avg_betweenness'] = np.mean(betweenness)
            features[f'{prefix}max_betweenness'] = np.max(betweenness)
            features[f'{prefix}std_betweenness'] = np.std(betweenness)
            
            # PageRank
            pagerank = list(centrality['pagerank'].values())
            features[f'{prefix}avg_pagerank'] = np.mean(pagerank)
            features[f'{prefix}max_pagerank'] = np.max(pagerank)
            features[f'{prefix}std_pagerank'] = np.std(pagerank)
            
            # Closeness
            closeness = list(centrality['closeness'].values())
            features[f'{prefix}avg_closeness'] = np.mean(closeness)
            features[f'{prefix}max_closeness'] = np.max(closeness)
            
        except Exception as e:
            print(f"⚠️  Error calculando centralidad: {e}")
            features[f'{prefix}avg_betweenness'] = 0
            features[f'{prefix}max_betweenness'] = 0
            features[f'{prefix}std_betweenness'] = 0
            features[f'{prefix}avg_pagerank'] = 0
            features[f'{prefix}max_pagerank'] = 0
            features[f'{prefix}std_pagerank'] = 0
            features[f'{prefix}avg_closeness'] = 0
            features[f'{prefix}max_closeness'] = 0
        
        # Conectividad
        features[f'{prefix}is_connected'] = int(nx.is_weakly_connected(G))
        features[f'{prefix}num_components'] = nx.number_weakly_connected_components(G)
        
        # Pesos de aristas
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        if weights:
            features[f'{prefix}avg_edge_weight'] = np.mean(weights)
            features[f'{prefix}max_edge_weight'] = np.max(weights)
            features[f'{prefix}std_edge_weight'] = np.std(weights)
            features[f'{prefix}total_edge_weight'] = np.sum(weights)
        else:
            features[f'{prefix}avg_edge_weight'] = 0
            features[f'{prefix}max_edge_weight'] = 0
            features[f'{prefix}std_edge_weight'] = 0
            features[f'{prefix}total_edge_weight'] = 0
        
        return features
    
    def extract_pass_statistics(self, passes: pd.DataFrame, prefix: str = '') -> Dict[str, float]:
        """
        Extrae estadísticas de pases.
        
        Args:
            passes: DataFrame con pases
            prefix: Prefijo para nombres de features
            
        Returns:
            Diccionario con features de pases
        """
        features = {}
        
        # Conteos básicos
        features[f'{prefix}total_passes'] = len(passes)
        features[f'{prefix}unique_players'] = passes['player'].nunique()
        
        if 'pass_recipient' in passes.columns:
            features[f'{prefix}unique_recipients'] = passes['pass_recipient'].nunique()
        
        # Éxito de pases
        if 'pass_success' in passes.columns:
            features[f'{prefix}successful_passes'] = passes['pass_success'].sum()
            features[f'{prefix}pass_accuracy'] = passes['pass_success'].mean()
        
        # Distancias de pases
        if 'pass_distance' in passes.columns:
            features[f'{prefix}avg_pass_distance'] = passes['pass_distance'].mean()
            features[f'{prefix}max_pass_distance'] = passes['pass_distance'].max()
            features[f'{prefix}min_pass_distance'] = passes['pass_distance'].min()
            features[f'{prefix}std_pass_distance'] = passes['pass_distance'].std()
            features[f'{prefix}total_pass_distance'] = passes['pass_distance'].sum()
        
        # Distribución temporal
        if 'minute' in passes.columns:
            features[f'{prefix}passes_first_half'] = len(passes[passes['minute'] <= 45])
            features[f'{prefix}passes_second_half'] = len(passes[passes['minute'] > 45])
            
            if features[f'{prefix}passes_first_half'] > 0:
                features[f'{prefix}passes_ratio_halves'] = (
                    features[f'{prefix}passes_second_half'] / 
                    features[f'{prefix}passes_first_half']
                )
            else:
                features[f'{prefix}passes_ratio_halves'] = 0
        
        # Pases por jugador
        passes_per_player = passes.groupby('player').size()
        features[f'{prefix}avg_passes_per_player'] = passes_per_player.mean()
        features[f'{prefix}max_passes_per_player'] = passes_per_player.max()
        features[f'{prefix}std_passes_per_player'] = passes_per_player.std()
        
        return features
    
    def extract_temporal_features(self, dynamic_graph: DynamicPassGraph, prefix: str = '') -> Dict[str, float]:
        """
        Extrae características de grafos dinámicos.
        
        Args:
            dynamic_graph: Objeto DynamicPassGraph con grafos temporales
            prefix: Prefijo para nombres de features
            
        Returns:
            Diccionario con features temporales
        """
        features = {}
        
        if dynamic_graph.metrics_timeline is None:
            dynamic_graph.calculate_temporal_metrics()
        
        metrics_df = dynamic_graph.metrics_timeline
        
        # Estadísticas de densidad a lo largo del tiempo
        features[f'{prefix}density_mean'] = metrics_df['density'].mean()
        features[f'{prefix}density_std'] = metrics_df['density'].std()
        features[f'{prefix}density_max'] = metrics_df['density'].max()
        features[f'{prefix}density_min'] = metrics_df['density'].min()
        features[f'{prefix}density_range'] = metrics_df['density'].max() - metrics_df['density'].min()
        
        # Variabilidad temporal (detectar cambios)
        features[f'{prefix}density_variability'] = metrics_df['density'].diff().abs().mean()
        
        # Clustering temporal
        features[f'{prefix}clustering_mean'] = metrics_df['avg_clustering'].mean()
        features[f'{prefix}clustering_std'] = metrics_df['avg_clustering'].std()
        
        # Número de conexiones temporal
        features[f'{prefix}edges_mean'] = metrics_df['num_edges'].mean()
        features[f'{prefix}edges_std'] = metrics_df['num_edges'].std()
        features[f'{prefix}edges_max'] = metrics_df['num_edges'].max()
        
        # Comparación entre mitades
        first_half = metrics_df[metrics_df['window_center'] <= 45]
        second_half = metrics_df[metrics_df['window_center'] > 45]
        
        if len(first_half) > 0 and len(second_half) > 0:
            features[f'{prefix}density_first_half'] = first_half['density'].mean()
            features[f'{prefix}density_second_half'] = second_half['density'].mean()
            features[f'{prefix}density_change_halves'] = (
                second_half['density'].mean() - first_half['density'].mean()
            )
            
            features[f'{prefix}clustering_first_half'] = first_half['avg_clustering'].mean()
            features[f'{prefix}clustering_second_half'] = second_half['avg_clustering'].mean()
        
        # Tendencia temporal (regresión lineal simple)
        time = metrics_df['window_center'].values
        density = metrics_df['density'].values
        
        if len(time) > 1:
            # Pendiente de densidad (aumenta o disminuye con el tiempo)
            slope = np.polyfit(time, density, 1)[0]
            features[f'{prefix}density_trend'] = slope
        else:
            features[f'{prefix}density_trend'] = 0
        
        return features
    
    def create_match_features(
        self, 
        passes: pd.DataFrame, 
        team: str,
        include_temporal: bool = True,
        window_size: int = 5,
        step_size: int = 2
    ) -> Dict[str, float]:
        """
        Crea vector completo de features para un partido.
        
        Args:
            passes: DataFrame con pases preprocesados
            team: Nombre del equipo
            include_temporal: Incluir features de grafos dinámicos
            window_size: Tamaño de ventana para grafos dinámicos
            step_size: Paso para grafos dinámicos
            
        Returns:
            Diccionario con todas las features
        """
        print(f"\n🔧 Extrayendo features para {team}...")
        
        # Filtrar por equipo
        team_passes = passes[passes['team'] == team].copy()
        
        all_features = {}
        
        # 1. Características de pases
        print("   - Estadísticas de pases...")
        pass_features = self.extract_pass_statistics(team_passes, prefix='')
        all_features.update(pass_features)
        
        # 2. Características del grafo estático
        print("   - Grafo estático...")
        pass_graph = PassGraph(weight_type='frequency')
        G = pass_graph.build_graph(team_passes, team=None)
        graph_features = self.extract_graph_features(G, prefix='static_')
        all_features.update(graph_features)
        
        # 3. Características temporales (opcional)
        if include_temporal and len(team_passes) >= 10:
            print("   - Grafos dinámicos...")
            try:
                dynamic_graph = DynamicPassGraph(
                    window_size=window_size, 
                    step_size=step_size, 
                    weight_type='frequency'
                )
                dynamic_graph.build_dynamic_graphs(team_passes, team=None)
                temporal_features = self.extract_temporal_features(dynamic_graph, prefix='temporal_')
                all_features.update(temporal_features)
            except Exception as e:
                print(f"   ⚠️  Error en features temporales: {e}")
        
        print(f"✅ Total de features extraídas: {len(all_features)}")
        
        return all_features
    
    def create_dataset_from_matches(
        self,
        matches_data: List[Tuple[str, pd.DataFrame, str]],
        include_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Crea dataset completo desde múltiples partidos.
        
        Args:
            matches_data: Lista de tuplas (match_id, passes, team_name)
            include_temporal: Incluir features temporales
            
        Returns:
            DataFrame con features de todos los partidos
        """
        print("\n" + "="*60)
        print("🏗️  CONSTRUYENDO DATASET DE FEATURES")
        print("="*60)
        
        all_features = []
        
        for idx, (match_id, passes, team) in enumerate(matches_data, 1):
            print(f"\n[{idx}/{len(matches_data)}] Match ID: {match_id} - Team: {team}")
            
            try:
                features = self.create_match_features(
                    passes, 
                    team, 
                    include_temporal=include_temporal
                )
                features['match_id'] = match_id
                features['team'] = team
                
                all_features.append(features)
                
            except Exception as e:
                print(f"   ❌ Error procesando partido: {e}")
                continue
        
        # Crear DataFrame
        df = pd.DataFrame(all_features)
        
        # Reorganizar columnas (match_id y team primero)
        cols = ['match_id', 'team'] + [col for col in df.columns if col not in ['match_id', 'team']]
        df = df[cols]
        
        print("\n" + "="*60)
        print(f"✅ DATASET CREADO: {len(df)} instancias, {len(df.columns)} features")
        print("="*60 + "\n")
        
        self.features = df
        self.feature_names = [col for col in df.columns if col not in ['match_id', 'team']]
        
        return df
    
    def add_labels(
        self, 
        df: pd.DataFrame, 
        labels_dict: Dict[str, any],
        label_column: str = 'label'
    ) -> pd.DataFrame:
        """
        Añade etiquetas al dataset (para clasificación supervisada).
        
        Args:
            df: DataFrame con features
            labels_dict: Diccionario {match_id: label}
            label_column: Nombre de la columna de etiqueta
            
        Returns:
            DataFrame con columna de etiquetas
        """
        df[label_column] = df['match_id'].map(labels_dict)
        
        missing = df[label_column].isna().sum()
        if missing > 0:
            print(f"⚠️  {missing} instancias sin etiqueta")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normaliza features (z-score normalization).
        
        Args:
            df: DataFrame con features
            exclude_cols: Columnas a excluir de normalización
            
        Returns:
            Tupla (DataFrame normalizado, diccionario con parámetros de normalización)
        """
        if exclude_cols is None:
            exclude_cols = ['match_id', 'team', 'label']
        
        df_normalized = df.copy()
        normalization_params = {}
        
        for col in df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    df_normalized[col] = (df[col] - mean) / std
                    normalization_params[col] = {'mean': mean, 'std': std}
        
        print(f"✅ {len(normalization_params)} features normalizadas")
        
        return df_normalized, normalization_params
    
    def get_feature_importance_ready_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'label'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos en formato X, y para modelos de ML.
        
        Args:
            df: DataFrame con features y etiquetas
            target_col: Nombre de la columna objetivo
            
        Returns:
            Tupla (X, y) donde X son las features e y las etiquetas
        """
        exclude_cols = ['match_id', 'team', target_col]
        
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target_col] if target_col in df.columns else None
        
        print(f"✅ Datos preparados: X shape = {X.shape}")
        if y is not None:
            print(f"   Distribución de clases: {y.value_counts().to_dict()}")
        
        return X, y
    
    def export_features(self, df: pd.DataFrame, filename: str = 'data/processed/features_dataset.csv'):
        """
        Exporta dataset de features a CSV.
        
        Args:
            df: DataFrame con features
            filename: Ruta del archivo de salida
        """
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        print(f"💾 Dataset exportado a: {filename}")
    
    def print_feature_summary(self, df: pd.DataFrame):
        """
        Imprime resumen del dataset de features.
        
        Args:
            df: DataFrame con features
        """
        print("\n📊 RESUMEN DEL DATASET DE FEATURES")
        print("="*60)
        print(f"Número de instancias: {len(df)}")
        print(f"Número de features: {len([col for col in df.columns if col not in ['match_id', 'team', 'label']])}")
        print(f"\nColumnas: {df.columns.tolist()}")
        print(f"\nTipos de datos:")
        print(df.dtypes.value_counts())
        print(f"\nValores nulos por columna:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        print("="*60 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando FeatureExtractor...\n")
    
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data.loader import StatsBombLoader
    from src.data.preprocessing import PassPreprocessor
    
    # Cargar datos
    loader = StatsBombLoader()
    matches = loader.get_matches(competition_id=11, season_id=90)
    
    # Seleccionar algunos partidos de Barcelona
    barcelona_matches = matches[
        (matches['home_team'] == 'Barcelona') | 
        (matches['away_team'] == 'Barcelona')
    ].head(3)  # Primeros 3 partidos
    
    preprocessor = PassPreprocessor()
    extractor = FeatureExtractor()
    
    # Preparar datos para múltiples partidos
    matches_data = []
    
    for idx, match_row in barcelona_matches.iterrows():
        match_id = match_row['match_id']
        print(f"\n📥 Cargando partido {match_id}: {match_row['home_team']} vs {match_row['away_team']}")
        
        events, _ = loader.load_match_data(match_id)
        passes_clean = preprocessor.preprocess_full_pipeline(events)
        
        # Añadir ambos equipos
        matches_data.append((match_id, passes_clean, match_row['home_team']))
        matches_data.append((match_id, passes_clean, match_row['away_team']))
    
    # Crear dataset
    df_features = extractor.create_dataset_from_matches(matches_data, include_temporal=True)
    
    # Resumen
    extractor.print_feature_summary(df_features)
    
    # Mostrar primeras filas
    print("📋 Primeras instancias del dataset:")
    print(df_features.head())
    
    # Exportar
    extractor.export_features(df_features, 'data/processed/features_example.csv')