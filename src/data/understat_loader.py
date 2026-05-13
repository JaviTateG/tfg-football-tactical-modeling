"""
Módulo para carga de datos históricos de equipos desde Understat.
Usado para el módulo de predicción pre-partido.
"""

import asyncio
import aiohttp
import understat
import nest_asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class UnderstatLoader:
    """
    Clase para cargar datos históricos de equipos desde Understat.
    Proporciona estadísticas de xG, resultados y forma reciente
    para cualquier equipo de las 5 grandes ligas.
    """

    TEAM_NAME_MAP = {
        'Barcelona':       'Barcelona',
        'Real Madrid':     'Real Madrid',
        'Atletico Madrid': 'Atletico Madrid',
        'Sevilla':         'Sevilla',
        'Valencia':        'Valencia',
        'Villarreal':      'Villarreal',
        'Athletic Club':   'Athletic Club',
        'Real Sociedad':   'Real Sociedad',
        'Real Betis':      'Real Betis',
        'Getafe':          'Getafe',
        'Celta Vigo':      'Celta Vigo',
        'Osasuna':         'Osasuna',
        'Granada':         'Granada',
        'Levante UD':      'Levante',
        'Cadiz':           'Cadiz',
        'Huesca':          'Huesca',
        'Alaves':          'Alaves',
        'Valladolid':      'Valladolid',
    }

    LEAGUE_MAP = {
        'La Liga':        'La_liga',
        'Premier League': 'EPL',
        'Bundesliga':     'Bundesliga',
        'Serie A':        'Serie_A',
        'Ligue 1':        'Ligue_1',
    }

    def __init__(self):
        nest_asyncio.apply()

    def _run_async(self, coro):
        """Ejecuta una corrutina async de forma síncrona."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    async def _get_team_matches_async(
        self,
        team_name: str,
        season: str = '2020'
    ) -> List[Dict]:
        """
        Obtiene los partidos de un equipo desde Understat de forma async.
        """
        async with aiohttp.ClientSession() as session:
            u = understat.Understat(session)
            matches = await u.get_team_results(team_name, season)
            return matches

    async def _get_league_matches_async(
        self,
        league: str = 'La_liga',
        season: str = '2020'
    ) -> List[Dict]:
        """
        Obtiene todos los partidos de una liga en una temporada.
        """
        async with aiohttp.ClientSession() as session:
            u = understat.Understat(session)
            matches = await u.get_league_results(
                league_name=league,
                season=season
            )
            return matches

    def get_team_matches(
        self,
        team_name: str,
        league: str = 'La Liga',
        season: str = '2020'
    ) -> pd.DataFrame:
        """
        Obtiene todos los partidos de un equipo en una temporada.

        Args:
            team_name: Nombre del equipo
            league: Liga
            season: Temporada (ej: '2023' para 2023/24)

        Returns:
            DataFrame con partidos y estadísticas
        """
        understat_name = self.TEAM_NAME_MAP.get(team_name, team_name)

        print(f" Cargando partidos de {team_name} ({season})...")

        try:
            matches = self._run_async(
                self._get_team_matches_async(understat_name, season=season)
            )

            if not matches:
                print(f"  No se encontraron datos para {team_name}")
                return pd.DataFrame()

            df = pd.DataFrame(matches)

            # Parsear columnas relevantes
            df['xG_home'] = df['xG'].apply(
                lambda x: float(x['h']) if isinstance(x, dict) else 0.0
            )
            df['xG_away'] = df['xG'].apply(
                lambda x: float(x['a']) if isinstance(x, dict) else 0.0
            )
            df['goals_home'] = df['goals'].apply(
                lambda x: int(x['h']) if isinstance(x, dict) else 0
            )
            df['goals_away'] = df['goals'].apply(
                lambda x: int(x['a']) if isinstance(x, dict) else 0
            )

            # Determinar si es local o visitante
            df['is_home'] = df['side'] == 'h'

            # xG del equipo y del rival
            df['xg_team']  = np.where(df['is_home'], df['xG_home'], df['xG_away'])
            df['xg_rival'] = np.where(df['is_home'], df['xG_away'], df['xG_home'])

            # Goles del equipo y del rival
            df['goals_team']  = np.where(df['is_home'], df['goals_home'], df['goals_away'])
            df['goals_rival'] = np.where(df['is_home'], df['goals_away'], df['goals_home'])

            # Resultado
            df['result'] = df.apply(
                lambda row: 'W' if row['goals_team'] > row['goals_rival']
                else ('L' if row['goals_team'] < row['goals_rival'] else 'D'),
                axis=1
            )

            # Fecha
            df['date'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('date').reset_index(drop=True)

            print(f" {len(df)} partidos cargados para {team_name}")
            return df

        except Exception as e:
            print(f" Error cargando {team_name}: {e}")
            return pd.DataFrame()

    def get_league_matches(
        self,
        league: str = 'La Liga',
        season: str = '2023'
    ) -> pd.DataFrame:
        """
        Obtiene todos los partidos de una liga en una temporada.

        Args:
            league: Liga
            season: Temporada

        Returns:
            DataFrame con todos los partidos
        """
        league_key = self.LEAGUE_MAP.get(league, 'La_liga')

        print(f" Cargando partidos de {league} ({season})...")

        try:
            matches = self._run_async(
                self._get_league_matches_async(league_key, season=season)
            )

            if not matches:
                print(f"  No se encontraron partidos")
                return pd.DataFrame()

            df = pd.DataFrame(matches)

            # Parsear columnas
            df['xG_home'] = df['xG'].apply(
                lambda x: float(x['h']) if isinstance(x, dict) else 0.0
            )
            df['xG_away'] = df['xG'].apply(
                lambda x: float(x['a']) if isinstance(x, dict) else 0.0
            )
            df['goals_home'] = df['goals'].apply(
                lambda x: int(x['h']) if isinstance(x, dict) else 0
            )
            df['goals_away'] = df['goals'].apply(
                lambda x: int(x['a']) if isinstance(x, dict) else 0
            )

            # Resultado
            df['result'] = df.apply(
                lambda row: 'H' if row['goals_home'] > row['goals_away']
                else ('A' if row['goals_home'] < row['goals_away'] else 'D'),
                axis=1
            )

            # Nombres de equipos
            df['home_team'] = df['h'].apply(
                lambda x: x.get('title', '') if isinstance(x, dict) else ''
            )
            df['away_team'] = df['a'].apply(
                lambda x: x.get('title', '') if isinstance(x, dict) else ''
            )

            print(f" {len(df)} partidos cargados")
            return df

        except Exception as e:
            print(f" Error cargando liga: {e}")
            return pd.DataFrame()

    def get_team_features(
        self,
        team_name: str,
        season: str = '2023',
        last_n: int = 5
    ) -> Dict:
        """
        Calcula features históricas de un equipo para predicción pre-partido.

        Args:
            team_name: Nombre del equipo
            season: Temporada
            last_n: Últimos N partidos para forma reciente

        Returns:
            Diccionario con features del equipo
        """
        df = self.get_team_matches(team_name, season=season)

        if df.empty:
            return {}

        features = {
            'team':   team_name,
            'season': season,

            # xG
            'xg_mean':            df['xg_team'].mean(),
            'xg_std':             df['xg_team'].std(),
            'xg_against_mean':    df['xg_rival'].mean(),
            'xg_against_std':     df['xg_rival'].std(),
            'xg_diff_mean':       (df['xg_team'] - df['xg_rival']).mean(),

            # Goles
            'goals_mean':         df['goals_team'].mean(),
            'goals_against_mean': df['goals_rival'].mean(),
            'goals_diff_mean':    (df['goals_team'] - df['goals_rival']).mean(),

            # Resultados
            'win_rate':           (df['result'] == 'W').mean(),
            'draw_rate':          (df['result'] == 'D').mean(),
            'loss_rate':          (df['result'] == 'L').mean(),
            'total_matches':      len(df),

            # Local vs visitante
            'home_win_rate': (df[df['is_home']]['result'] == 'W').mean()
                              if df['is_home'].sum() > 0 else 0.0,
            'away_win_rate': (df[~df['is_home']]['result'] == 'W').mean()
                              if (~df['is_home']).sum() > 0 else 0.0,
        }

        # Forma reciente
        recent = df.tail(last_n)
        features.update({
            'recent_xg_mean':         recent['xg_team'].mean(),
            'recent_xg_against_mean': recent['xg_rival'].mean(),
            'recent_goals_mean':      recent['goals_team'].mean(),
            'recent_win_rate':        (recent['result'] == 'W').mean(),
            'recent_draw_rate':       (recent['result'] == 'D').mean(),
            'recent_loss_rate':       (recent['result'] == 'L').mean(),
            'recent_points':          (
                (recent['result'] == 'W').sum() * 3 +
                (recent['result'] == 'D').sum()
            ),
        })

        return features

    def build_match_prediction_features(
        self,
        home_team: str,
        away_team: str,
        season: str = '2023',
        last_n: int = 5
    ) -> Dict:
        """
        Construye el vector de features completo para predecir
        el resultado de un partido antes de que se juegue.

        Args:
            home_team: Equipo local
            away_team: Equipo visitante
            season: Temporada
            last_n: Últimos N partidos para forma reciente

        Returns:
            Diccionario con todas las features del enfrentamiento
        """
        print(f"\n Construyendo features: {home_team} vs {away_team}")
        

        home_features = self.get_team_features(home_team, season, last_n)
        away_features = self.get_team_features(away_team, season, last_n)

        if not home_features or not away_features:
            print(" No se pudieron obtener features de uno o ambos equipos")
            return {}

        match_features = {'home_advantage': 1.0}

        for key, val in home_features.items():
            if key not in ['team', 'season']:
                match_features[f'home_{key}'] = val

        for key, val in away_features.items():
            if key not in ['team', 'season']:
                match_features[f'away_{key}'] = val

        numeric_keys = [
            'xg_mean', 'xg_against_mean', 'xg_diff_mean',
            'goals_mean', 'goals_against_mean',
            'win_rate', 'recent_xg_mean', 'recent_win_rate',
            'recent_points'
        ]
        for key in numeric_keys:
            if f'home_{key}' in match_features and f'away_{key}' in match_features:
                match_features[f'diff_{key}'] = (
                    match_features[f'home_{key}'] -
                    match_features[f'away_{key}']
                )

        match_features['home_advantage'] = 1.0

        print(f" Features construidas: {len(match_features)}")
        return match_features