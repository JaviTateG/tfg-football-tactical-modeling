"""
Script para descargar datos de Understat y guardarlos en CSV.
Ejecutar desde terminal, NO desde Jupyter.
"""

import asyncio
import aiohttp
import understat
import pandas as pd
import numpy as np
import os

EQUIPOS_LALIGA = [
    'Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla',
    'Valencia', 'Villarreal', 'Athletic Club', 'Real Sociedad',
    'Real Betis', 'Getafe', 'Celta Vigo', 'Osasuna'
]

TEMPORADAS = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']

async def download_all():

    all_features  = []
    all_matches   = []

    async with aiohttp.ClientSession() as session:
        u = understat.Understat(session)

        #  Features por equipo 
        for season in TEMPORADAS:
            print(f"\n Temporada {season}/{int(season)+1}")

            for team in EQUIPOS_LALIGA:
                try:
                    matches = await u.get_team_results(team, season)

                    if not matches:
                        continue

                    df = pd.DataFrame(matches)

                    df['xG_home']    = df['xG'].apply(lambda x: float(x['h']) if isinstance(x, dict) else 0.0)
                    df['xG_away']    = df['xG'].apply(lambda x: float(x['a']) if isinstance(x, dict) else 0.0)
                    df['goals_home'] = df['goals'].apply(lambda x: int(x['h']) if isinstance(x, dict) else 0)
                    df['goals_away'] = df['goals'].apply(lambda x: int(x['a']) if isinstance(x, dict) else 0)
                    df['is_home']    = df['side'] == 'h'

                    df['xg_team']     = np.where(df['is_home'], df['xG_home'], df['xG_away'])
                    df['xg_rival']    = np.where(df['is_home'], df['xG_away'], df['xG_home'])
                    df['goals_team']  = np.where(df['is_home'], df['goals_home'], df['goals_away'])
                    df['goals_rival'] = np.where(df['is_home'], df['goals_away'], df['goals_home'])

                    df['result'] = df.apply(
                        lambda row: 'W' if row['goals_team'] > row['goals_rival']
                        else ('L' if row['goals_team'] < row['goals_rival'] else 'D'),
                        axis=1
                    )
                    df['date'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('date').reset_index(drop=True)

                    recent = df.tail(5)
                    features = {
                        'team':   team,
                        'season': season,

                        'xg_mean':             df['xg_team'].mean(),
                        'xg_std':              df['xg_team'].std(),
                        'xg_against_mean':     df['xg_rival'].mean(),
                        'xg_against_std':      df['xg_rival'].std(),
                        'xg_diff_mean':        (df['xg_team'] - df['xg_rival']).mean(),

                        'goals_mean':          df['goals_team'].mean(),
                        'goals_against_mean':  df['goals_rival'].mean(),
                        'goals_diff_mean':     (df['goals_team'] - df['goals_rival']).mean(),

                        'win_rate':            (df['result'] == 'W').mean(),
                        'draw_rate':           (df['result'] == 'D').mean(),
                        'loss_rate':           (df['result'] == 'L').mean(),
                        'total_matches':       len(df),

                        'home_win_rate': (df[df['is_home']]['result'] == 'W').mean()
                                          if df['is_home'].sum() > 0 else 0.0,
                        'away_win_rate': (df[~df['is_home']]['result'] == 'W').mean()
                                          if (~df['is_home']).sum() > 0 else 0.0,

                        'recent_xg_mean':         recent['xg_team'].mean(),
                        'recent_xg_against_mean': recent['xg_rival'].mean(),
                        'recent_goals_mean':      recent['goals_team'].mean(),
                        'recent_win_rate':        (recent['result'] == 'W').mean(),
                        'recent_draw_rate':       (recent['result'] == 'D').mean(),
                        'recent_loss_rate':       (recent['result'] == 'L').mean(),
                        'recent_points': (
                            (recent['result'] == 'W').sum() * 3 +
                            (recent['result'] == 'D').sum()
                        ),
                    }

                    all_features.append(features)
                    print(f"    {team} ({season}): {len(df)} partidos")

                except Exception as e:
                    print(f"    {team} ({season}): {e}")

        #  Partidos de la liga 
        for season in TEMPORADAS:
            print(f"\n Cargando partidos liga {season}/{int(season)+1}...")
            try:
                matches = await u.get_league_results(
                    league_name='La_liga',
                    season=season
                )
                df = pd.DataFrame(matches)
                df['season'] = season
                all_matches.append(df)
                print(f"    {len(df)} partidos")
            except Exception as e:
                print(f"    Error: {e}")

    return all_features, all_matches


if __name__ == '__main__':
    # Crear carpeta de salida
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data', 'processed'
    )
    os.makedirs(output_dir, exist_ok=True)

    print(" Descargando datos de Understat...")
    all_features, all_matches = asyncio.run(download_all())

    # Guardar features de equipos
    df_features = pd.DataFrame(all_features)
    path_features = os.path.join(output_dir, 'understat_team_features.csv')
    df_features.to_csv(path_features, index=False)
    print(f"\n Features guardadas: {path_features}")
    print(f"   Registros: {len(df_features)}")

    # Guardar partidos
    df_matches = pd.concat(all_matches, ignore_index=True)
    path_matches = os.path.join(output_dir, 'understat_matches.csv')
    df_matches.to_csv(path_matches, index=False)
    print(f"\n Partidos guardados: {path_matches}")
    print(f"   Partidos: {len(df_matches)}")

    print("\n Descarga completada")