"""
Módulo para preprocesamiento y limpieza de datos de eventos de fútbol.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class PassPreprocessor:
    """
    Clase para preprocesar datos de pases de StatsBomb.
    """
    
    def __init__(self, min_pass_length: float = 0.0, max_pass_length: float = 120.0):
        """
        Inicializa el preprocesador.
        
        Args:
            min_pass_length: Longitud mínima del pase en metros (filtro)
            max_pass_length: Longitud máxima del pase en metros (filtro)
        """
        self.min_pass_length = min_pass_length
        self.max_pass_length = max_pass_length
    
    def clean_passes(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra y limpia eventos de pases.
        
        Args:
            events: DataFrame con todos los eventos del partido
            
        Returns:
            DataFrame con pases limpios
        """
        print(f"📊 Eventos totales: {len(events)}")
        
        # Filtrar solo pases
        passes = events[events['type'] == 'Pass'].copy()
        print(f"✅ Pases encontrados: {len(passes)}")
        
        # Eliminar filas con datos críticos faltantes
        initial_count = len(passes)
        passes = passes.dropna(subset=['player', 'team'])
        print(f"🧹 Pases después de eliminar nulos en player/team: {len(passes)}")
        
        return passes
    
    def add_pass_outcome_flag(self, passes: pd.DataFrame) -> pd.DataFrame:
        """
        Añade columna binaria de éxito del pase.
        
        Args:
            passes: DataFrame con pases
            
        Returns:
            DataFrame con columna 'pass_success'
        """
        # En StatsBomb, si 'pass_outcome' es NaN, el pase fue exitoso
        passes['pass_success'] = passes['pass_outcome'].isna().astype(int)
        
        success_rate = passes['pass_success'].mean() * 100
        print(f"✅ Precisión de pase: {success_rate:.2f}%")
        
        return passes
    
    def calculate_pass_distance(self, passes: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la distancia euclidiana de cada pase.
        
        Args:
            passes: DataFrame con pases (debe tener 'location' y 'pass_end_location')
            
        Returns:
            DataFrame con columna 'pass_distance'
        """
        if 'location' not in passes.columns or 'pass_end_location' not in passes.columns:
            print("⚠️  No se pueden calcular distancias (faltan columnas de ubicación)")
            passes['pass_distance'] = np.nan
            return passes
        
        def euclidean_distance(row):
            try:
                start = np.array(row['location'])
                end = np.array(row['pass_end_location'])
                return np.linalg.norm(end - start)
            except:
                return np.nan
        
        passes['pass_distance'] = passes.apply(euclidean_distance, axis=1)
        
        # Filtrar por longitud
        before = len(passes)
        passes = passes[
            (passes['pass_distance'] >= self.min_pass_length) & 
            (passes['pass_distance'] <= self.max_pass_length)
        ]
        print(f"🧹 Pases después de filtrar por distancia: {len(passes)} (eliminados: {before - len(passes)})")
        
        return passes
    
    def normalize_time(self, passes: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza información temporal.
        
        Args:
            passes: DataFrame con pases
            
        Returns:
            DataFrame con columnas temporales normalizadas
        """
        # Crear timestamp único (minuto + segundo)
        if 'minute' in passes.columns and 'second' in passes.columns:
            passes['timestamp'] = passes['minute'] * 60 + passes['second']
        elif 'minute' in passes.columns:
            passes['timestamp'] = passes['minute'] * 60
        else:
            print("⚠️  No se puede normalizar tiempo (faltan columnas)")
            passes['timestamp'] = np.arange(len(passes))
        
        # Identificar periodo del partido
        if 'period' in passes.columns:
            passes['half'] = passes['period'].apply(lambda x: '1H' if x == 1 else '2H' if x == 2 else 'Extra')
        
        print(f"✅ Normalización temporal completada")
        
        return passes
    
    def add_recipient_info(self, passes: pd.DataFrame) -> pd.DataFrame:
        """
        Asegura que la información del receptor esté presente.
        
        Args:
            passes: DataFrame con pases
            
        Returns:
            DataFrame con información de receptor limpia
        """
        if 'pass_recipient' not in passes.columns:
            print("⚠️  Columna 'pass_recipient' no encontrada")
            return passes
        
        # Eliminar pases sin receptor (no podemos construir grafo)
        initial = len(passes)
        passes = passes.dropna(subset=['pass_recipient'])
        print(f"🧹 Pases con receptor válido: {len(passes)} (eliminados: {initial - len(passes)})")
        
        return passes
    
    def filter_by_team(self, passes: pd.DataFrame, team_name: Optional[str] = None) -> pd.DataFrame:
        """
        Filtra pases de un equipo específico.
        
        Args:
            passes: DataFrame con pases
            team_name: Nombre del equipo a filtrar (None = todos)
            
        Returns:
            DataFrame filtrado
        """
        if team_name is None:
            return passes
        
        filtered = passes[passes['team'] == team_name].copy()
        print(f"🎯 Pases de {team_name}: {len(filtered)}")
        
        return filtered
    
    def preprocess_full_pipeline(
        self, 
        events: pd.DataFrame, 
        team_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            events: DataFrame con todos los eventos
            team_name: Filtrar por equipo específico (opcional)
            
        Returns:
            DataFrame con pases preprocesados y listos para análisis
        """
        print("\n" + "="*60)
        print("🧹 INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*60 + "\n")
        
        # 1. Filtrar y limpiar pases
        passes = self.clean_passes(events)
        
        # 2. Añadir flag de éxito
        passes = self.add_pass_outcome_flag(passes)
        
        # 3. Calcular distancias
        passes = self.calculate_pass_distance(passes)
        
        # 4. Normalizar tiempo
        passes = self.normalize_time(passes)
        
        # 5. Asegurar receptor válido
        passes = self.add_recipient_info(passes)
        
        # 6. Filtrar por equipo (opcional)
        if team_name:
            passes = self.filter_by_team(passes, team_name)
        
        # Resetear índice
        passes = passes.reset_index(drop=True)
        
        print("\n" + "="*60)
        print(f"✅ PREPROCESAMIENTO COMPLETADO: {len(passes)} pases válidos")
        print("="*60 + "\n")
        
        return passes
    
    def get_summary_stats(self, passes: pd.DataFrame) -> dict:
        """
        Obtiene estadísticas resumen de los pases preprocesados.
        
        Args:
            passes: DataFrame con pases preprocesados
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'total_passes': len(passes),
            'successful_passes': passes['pass_success'].sum() if 'pass_success' in passes.columns else None,
            'pass_accuracy': passes['pass_success'].mean() * 100 if 'pass_success' in passes.columns else None,
            'unique_players': passes['player'].nunique(),
            'unique_recipients': passes['pass_recipient'].nunique() if 'pass_recipient' in passes.columns else None,
            'teams': passes['team'].unique().tolist() if 'team' in passes.columns else None,
            'avg_pass_distance': passes['pass_distance'].mean() if 'pass_distance' in passes.columns else None,
            'min_pass_distance': passes['pass_distance'].min() if 'pass_distance' in passes.columns else None,
            'max_pass_distance': passes['pass_distance'].max() if 'pass_distance' in passes.columns else None,
        }
        
        return stats
    
    def print_summary(self, passes: pd.DataFrame):
        """
        Imprime resumen de estadísticas.
        
        Args:
            passes: DataFrame con pases preprocesados
        """
        stats = self.get_summary_stats(passes)
        
        print("\n📊 RESUMEN DE DATOS PREPROCESADOS")
        print("="*60)
        print(f"Total de pases: {stats['total_passes']}")
        print(f"Pases exitosos: {stats['successful_passes']}")
        print(f"Precisión: {stats['pass_accuracy']:.2f}%")
        print(f"Jugadores únicos: {stats['unique_players']}")
        print(f"Receptores únicos: {stats['unique_recipients']}")
        print(f"Equipos: {', '.join(stats['teams']) if stats['teams'] else 'N/A'}")
        
        if stats['avg_pass_distance']:
            print(f"\nDistancia de pases:")
            print(f"  - Promedio: {stats['avg_pass_distance']:.2f}m")
            print(f"  - Mínima: {stats['min_pass_distance']:.2f}m")
            print(f"  - Máxima: {stats['max_pass_distance']:.2f}m")
        
        print("="*60 + "\n")


def save_processed_passes(passes: pd.DataFrame, filename: str = 'data/processed/passes_clean.csv'):
    """
    Guarda pases preprocesados a disco.
    
    Args:
        passes: DataFrame con pases preprocesados
        filename: Ruta del archivo de salida
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    passes.to_csv(filename, index=False)
    print(f"💾 Datos guardados en: {filename}")

if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando PassPreprocessor...\n")
    
    import sys
    import os
    
    # Añadir la raíz del proyecto al path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data.loader import StatsBombLoader
    
    # Cargar datos
    loader = StatsBombLoader()
    matches = loader.get_matches(competition_id=11, season_id=90)
    
    # Seleccionar primer partido de Barcelona
    barcelona_matches = matches[
        (matches['home_team'] == 'Barcelona') | 
        (matches['away_team'] == 'Barcelona')
    ]
    
    match_id = barcelona_matches.iloc[0]['match_id']
    match_info = barcelona_matches.iloc[0]
    
    print(f"🎯 Partido: {match_info['home_team']} vs {match_info['away_team']}\n")
    
    # Cargar eventos
    events, _ = loader.load_match_data(match_id)
    
    # Preprocesar
    preprocessor = PassPreprocessor(min_pass_length=1.0, max_pass_length=100.0)
    passes_clean = preprocessor.preprocess_full_pipeline(events)
    
    # Mostrar resumen
    preprocessor.print_summary(passes_clean)
    
    # Guardar
    save_processed_passes(passes_clean)
    
    # Mostrar muestra de datos
    print("📋 Muestra de datos preprocesados:")
    columns_to_show = ['player', 'pass_recipient', 'team', 'minute', 
                       'pass_success', 'pass_distance', 'timestamp']
    available_cols = [col for col in columns_to_show if col in passes_clean.columns]
    print(passes_clean[available_cols].head(10))

    