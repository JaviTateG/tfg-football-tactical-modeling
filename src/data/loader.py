"""
Módulo para la carga de datos desde StatsBomb.
"""

from statsbombpy import sb
import pandas as pd
from typing import Tuple, Optional
import os


class StatsBombLoader:
    """
    Clase para cargar y gestionar datos de StatsBomb.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Inicializa el loader.
        
        Args:
            data_dir: Directorio donde guardar datos raw (None = auto-detectar)
        """
        if data_dir is None:
            # Detectar raíz del proyecto automáticamente
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            self.data_dir = os.path.join(project_root, 'data', 'raw')
        else:
            self.data_dir = data_dir
        
        os.makedirs(self.data_dir, exist_ok=True)
        print(f" Data directory: {self.data_dir}")
    
    def get_competitions(self) -> pd.DataFrame:
        """
        Obtiene todas las competiciones disponibles.
        
        Returns:
            DataFrame con competiciones
        """
        return sb.competitions()
    
    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """
        Obtiene partidos de una competición y temporada específica.
        
        Args:
            competition_id: ID de la competición (ej: 11 para La Liga)
            season_id: ID de la temporada (ej: 90 para 2020/2021)
            
        Returns:
            DataFrame con los partidos
        """
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        
        # Guardar en raw
        filename = os.path.join(self.data_dir, f"matches_{competition_id}_{season_id}.csv")
        matches.to_csv(filename, index=False)
        print(f" Partidos guardados en: {filename}")
        
        return matches
    
    def get_events(self, match_id: int) -> pd.DataFrame:
        """
        Obtiene todos los eventos de un partido.
        
        Args:
            match_id: ID del partido
            
        Returns:
            DataFrame con eventos del partido
        """
        events = sb.events(match_id=match_id)
        
        # Guardar en raw
        filename = os.path.join(self.data_dir, f"events_{match_id}.csv")
        events.to_csv(filename, index=False)
        print(f" Eventos guardados en: {filename}")
        
        return events
    
    def get_passes(self, match_id: int) -> pd.DataFrame:
        """
        Obtiene solo los pases de un partido.
        
        Args:
            match_id: ID del partido
            
        Returns:
            DataFrame con pases del partido
        """
        events = self.get_events(match_id)
        passes = events[events['type'] == 'Pass'].copy()
        
        print(f" Total de pases encontrados: {len(passes)}")
        
        return passes
    
    def load_match_data(self, match_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga eventos completos y pases de un partido.
        
        Args:
            match_id: ID del partido
            
        Returns:
            Tupla con (eventos_completos, pases)
        """
        events = self.get_events(match_id)
        passes = events[events['type'] == 'Pass'].copy()
        
        print(f" Partido {match_id}:")
        print(f"   - Total eventos: {len(events)}")
        print(f"   - Total pases: {len(passes)}")
        
        return events, passes


# Función auxiliar para uso rápido
def quick_load_passes(match_id: int) -> pd.DataFrame:
    """
    Carga rápida de pases de un partido.
    
    Args:
        match_id: ID del partido
        
    Returns:
        DataFrame con pases
    """
    loader = StatsBombLoader()
    return loader.get_passes(match_id)


if __name__ == "__main__":
    # Ejemplo de uso
    print(" Probando StatsBombLoader...\n")
    
    loader = StatsBombLoader()
    
    # Obtener competiciones
    print("1 Competiciones disponibles:")
    comps = loader.get_competitions()
    print(comps[['competition_name', 'season_name']].head())
    
    # Obtener partidos de La Liga 2020/2021
    print("\n 2 Partidos de La Liga 2020/2021:")
    matches = loader.get_matches(competition_id=11, season_id=90)
    print(matches[['match_date', 'home_team', 'away_team']].head())
    
    # Obtener pases de un partido
    print("\n 3 Pases del primer partido:")
    match_id = matches.iloc[0]['match_id']
    passes = loader.get_passes(match_id)
    print(passes[['team', 'player', 'pass_recipient', 'minute']].head())