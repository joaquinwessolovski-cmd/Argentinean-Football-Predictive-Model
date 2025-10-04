"""
Utilidades para preprocesar, limpiar y analizar datos de fútbol
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# LIMPIEZA Y VALIDACIÓN DE DATOS
# ============================================================================

def validate_and_clean_data(matches_df):
    """
    Valida y limpia el dataset de partidos
    """
    print("🔍 Validando y limpiando datos...")
    
    original_len = len(matches_df)
    issues = []
    
    # 1. Verificar columnas requeridas
    required_cols = ['date', 'home_team', 'away_team', 'home_score', 
                     'away_score', 'xg_home', 'xg_away']
    missing_cols = [col for col in required_cols if col not in matches_df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ Columnas faltantes: {missing_cols}")
    
    # 2. Convertir fecha a datetime
    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
    
    # 3. Eliminar filas con fechas inválidas
    invalid_dates = matches_df['date'].isna().sum()
    if invalid_dates > 0:
        issues.append(f"Fechas inválidas: {invalid_dates}")
        matches_df = matches_df.dropna(subset=['date'])
    
    # 4. Verificar valores numéricos
    for col in ['home_score', 'away_score', 'xg_home', 'xg_away']:
        matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce')
        invalid = matches_df[col].isna().sum()
        if invalid > 0:
            issues.append(f"{col} con valores inválidos: {invalid}")
    
    matches_df = matches_df.dropna(subset=['home_score', 'away_score', 'xg_home', 'xg_away'])
    
    # 5. Validar rangos razonables
    # Goles
    unrealistic_goals = matches_df[
        (matches_df['home_score'] < 0) | 
        (matches_df['away_score'] < 0) |
        (matches_df['home_score'] > 15) | 
        (matches_df['away_score'] > 15)
    ]
    if len(unrealistic_goals) > 0:
        issues.append(f"Goles fuera de rango: {len(unrealistic_goals)}")
        matches_df = matches_df[
            (matches_df['home_score'] >= 0) & 
            (matches_df['away_score'] >= 0) &
            (matches_df['home_score'] <= 15) & 
            (matches_df['away_score'] <= 15)
        ]
    
    # xG
    unrealistic_xg = matches_df[
        (matches_df['xg_home'] < 0) | 
        (matches_df['xg_away'] < 0) |
        (matches_df['xg_home'] > 8) | 
        (matches_df['xg_away'] > 8)
    ]
    if len(unrealistic_xg) > 0:
        issues.append(f"xG fuera de rango: {len(unrealistic_xg)}")
        matches_df = matches_df[
            (matches_df['xg_home'] >= 0) & 
            (matches_df['xg_away'] >= 0) &
            (matches_df['xg_home'] <= 8) & 
            (matches_df['xg_away'] <= 8)
        ]
    
    # 6. Normalizar nombres de equipos
    matches_df['home_team'] = matches_df['home_team'].str.strip()
    matches_df['away_team'] = matches_df['away_team'].str.strip()
    
    # 7. Eliminar duplicados exactos
    duplicates = matches_df.duplicated(subset=['date', 'home_team', 'away_team']).sum()
    if duplicates > 0:
        issues.append(f"Partidos duplicados: {duplicates}")
        matches_df = matches_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    
    # 8. Verificar que equipos no jueguen contra sí mismos
    self_matches = matches_df[matches_df['home_team'] == matches_df['away_team']]
    if len(self_matches) > 0:
        issues.append(f"Equipos jugando contra sí mismos: {len(self_matches)}")
        matches_df = matches_df[matches_df['home_team'] != matches_df['away_team']]
    
    # Resumen
    rows_removed = original_len - len(matches_df)
    
    print(f"\n✅ Validación completada:")
    print(f"   - Filas originales: {original_len}")
    print(f"   - Filas después de limpieza: {len(matches_df)}")
    print(f"   - Filas eliminadas: {rows_removed} ({rows_removed/original_len*100:.1f}%)")
    
    if issues:
        print(f"\n⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"   - {issue}")
    
    return matches_df.sort_values('date').reset_index(drop=True)


# ============================================================================
# FEATURE ENGINEERING AVANZADO
# ============================================================================

def create_advanced_features(matches_df):
    """
    Crea features avanzadas para mejorar las predicciones
    """
    print("\n🔧 Creando features avanzadas...")
    
    df = matches_df.copy()
    
    # 1. Resultado del partido
    df['result'] = df.apply(
        lambda x: 'H' if x['home_score'] > x['away_score'] 
        else ('A' if x['home_score'] < x['away_score'] else 'D'),
        axis=1
    )
    
    # 2. Total de goles
    df['total_goals'] = df['home_score'] + df['away_score']
    df['total_xg'] = df['xg_home'] + df['xg_away']
    
    # 3. Diferencia de goles
    df['goal_diff'] = df['home_score'] - df['away_score']
    df['xg_diff'] = df['xg_home'] - df['xg_away']
    
    # 4. Over/Under markets
    df['over_1.5'] = (df['total_goals'] > 1.5).astype(int)
    df['over_2.5'] = (df['total_goals'] > 2.5).astype(int)
    df['over_3.5'] = (df['total_goals'] > 3.5).astype(int)
    
    # 5. Both Teams To Score (BTTS)
    df['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)
    
    # 6. Eficiencia (goles vs xG)
    df['home_efficiency'] = df['home_score'] / (df['xg_home'] + 0.01)  # +0.01 para evitar división por 0
    df['away_efficiency'] = df['away_score'] / (df['xg_away'] + 0.01)
    
    # 7. Suerte/Fortuna (desviación de xG)
    df['home_luck'] = df['home_score'] - df['xg_home']
    df['away_luck'] = df['away_score'] - df['xg_away']
    
    # 8. Features temporales
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 9. Calcular días desde último partido por equipo
    df = calculate_rest_days(df)
    
    print(f"✅ {len(df.columns) - len(matches_df.columns)} nuevas features creadas")
    
    return df


def calculate_rest_days(df):
    """
    Calcula días de descanso entre partidos para cada equipo
    """
    df = df.sort_values('date').copy()
    
    team_last_match = {}
    rest_days_home = []
    rest_days_away = []
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        match_date = row['date']
        
        # Días de descanso para local
        if home_team in team_last_match:
            days_rest = (match_date - team_last_match[home_team]).days
            rest_days_home.append(days_rest)
        else:
            rest_days_home.append(None)  # Primer partido
        
        # Días de descanso para visitante
        if away_team in team_last_match:
            days_rest = (match_date - team_last_match[away_team]).days
            rest_days_away.append(days_rest)
        else:
            rest_days_away.append(None)  # Primer partido
        
        # Actualizar última fecha
        team_last_match[home_team] = match_date
        team_last_match[away_team] = match_date
    
    df['home_rest_days'] = rest_days_home
    df['away_rest_days'] = rest_days_away
    
    # Rellenar None con mediana
    median_rest = df[['home_rest_days', 'away_rest_days']].median().median()
    df['home_rest_days'].fillna(median_rest, inplace=True)
    df['away_rest_days'].fillna(median_rest, inplace=True)
    
    return df


# ============================================================================
# ANÁLISIS EXPLORATORIO
# ============================================================================

def exploratory_data_analysis(matches_df):
    """
    Genera un reporte completo del dataset
    """
    print("="*70)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*70)
    
    # 1. Información general
    print(f"\n📊 INFORMACIÓN GENERAL")
    print(f"   Total de partidos: {len(matches_df)}")
    print(f"   Rango de fechas: {matches_df['date'].min()} a {matches_df['date'].max()}")
    print(f"   Días cubiertos: {(matches_df['date'].max() - matches_df['date'].min()).days}")
    
    teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())
    print(f"   Equipos únicos: {len(teams)}")
    
    if 'league' in matches_df.columns:
        print(f"   Ligas: {matches_df['league'].nunique()}")
        print(f"   Ligas: {', '.join(matches_df['league'].unique())}")
    
    # 2. Estadísticas de goles
    print(f"\n⚽ ESTADÍSTICAS DE GOLES")
    print(f"   Goles totales: {matches_df['home_score'].sum() + matches_df['away_score'].sum():.0f}")
    print(f"   Promedio por partido: {(matches_df['home_score'] + matches_df['away_score']).mean():.2f}")
    print(f"   Goles local promedio: {matches_df['home_score'].mean():.2f}")
    print(f"   Goles visitante promedio: {matches_df['away_score'].mean():.2f}")
    print(f"   Partido con más goles: {(matches_df['home_score'] + matches_df['away_score']).max():.0f}")
    
    # 3. Estadísticas de xG
    print(f"\n📈 ESTADÍSTICAS DE xG")
    print(f"   xG total: {matches_df['xg_home'].sum() + matches_df['xg_away'].sum():.1f}")
    print(f"   xG promedio por partido: {(matches_df['xg_home'] + matches_df['xg_away']).mean():.2f}")
    print(f"   xG local promedio: {matches_df['xg_home'].mean():.2f}")
    print(f"   xG visitante promedio: {matches_df['xg_away'].mean():.2f}")
    
    # Correlación goles vs xG
    corr_home = matches_df['home_score'].corr(matches_df['xg_home'])
    corr_away = matches_df['away_score'].corr(matches_df['xg_away'])
    print(f"   Correlación goles-xG (local): {corr_home:.3f}")
    print(f"   Correlación goles-xG (visitante): {corr_away:.3f}")
    
    # 4. Distribución de resultados
    home_wins = len(matches_df[matches_df['home_score'] > matches_df['away_score']])
    draws = len(matches_df[matches_df['home_score'] == matches_df['away_score']])
    away_wins = len(matches_df[matches_df['home_score'] < matches_df['away_score']])
    
    print(f"\n🏆 DISTRIBUCIÓN DE RESULTADOS")
    print(f"   Victorias locales: {home_wins} ({home_wins/len(matches_df)*100:.1f}%)")
    print(f"   Empates: {draws} ({draws/len(matches_df)*100:.1f}%)")
    print(f"   Victorias visitantes: {away_wins} ({away_wins/len(matches_df)*100:.1f}%)")
    print(f"   Factor localía: {home_wins/(home_wins+away_wins)*100:.1f}%")
    
    # 5. Mercados Over/Under
    over_05 = len(matches_df[(matches_df['home_score'] + matches_df['away_score']) > 0.5])
    over_15 = len(matches_df[(matches_df['home_score'] + matches_df['away_score']) > 1.5])
    over_25 = len(matches_df[(matches_df['home_score'] + matches_df['away_score']) > 2.5])
    over_35 = len(matches_df[(matches_df['home_score'] + matches_df['away_score']) > 3.5])
    
    print(f"\n📊 MERCADOS OVER/UNDER")
    print(f"   Over 0.5: {over_05/len(matches_df)*100:.1f}%")
    print(f"   Over 1.5: {over_15/len(matches_df)*100:.1f}%")
    print(f"   Over 2.5: {over_25/len(matches_df)*100:.1f}%")
    print(f"   Over 3.5: {over_35/len(matches_df)*100:.1f}%")
    
    # 6. BTTS
    btts = len(matches_df[(matches_df['home_score'] > 0) & (matches_df['away_score'] > 0)])
    print(f"   Both Teams To Score: {btts/len(matches_df)*100:.1f}%")
    
    # 7. Partidos con más goles
    print(f"\n🔥 TOP 5 PARTIDOS CON MÁS GOLES")
    top_goals = matches_df.copy()
    top_goals['total_goals'] = top_goals['home_score'] + top_goals['away_score']
    top_goals = top_goals.nlargest(5, 'total_goals')
    
    for _, row in top_goals.iterrows():
        print(f"   {row['home_team']} {row['home_score']:.0f}-{row['away_score']:.0f} {row['away_team']} "
              f"({row['date'].strftime('%d/%m/%Y')})")
    
    # 8. Equipos más prolíficos
    print(f"\n⚽ TOP 10 EQUIPOS CON MÁS GOLES")
    team_goals = {}
    for _, row in matches_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        if home not in team_goals:
            team_goals[home] = 0
        if away not in team_goals:
            team_goals[away] = 0
        
        team_goals[home] += row['home_score']
        team_goals[away] += row['away_score']
    
    sorted_goals = sorted(team_goals.items(), key=lambda x: x[1], reverse=True)
    for i, (team, goals) in enumerate(sorted_goals[:10], 1):
        matches_played = len(matches_df[
            (matches_df['home_team'] == team) | (matches_df['away_team'] == team)
        ])
        avg = goals / matches_played if matches_played > 0 else 0
        print(f"   {i:2}. {team:<30} {goals:.0f} goles ({avg:.2f} por partido)")


# ============================================================================
# DETECCIÓN DE ANOMALÍAS
# ============================================================================

def detect_anomalies(matches_df):
    """
    Detecta partidos o valores anómalos
    """
    print("\n" + "="*70)
    print("DETECCIÓN DE ANOMALÍAS")
    print("="*70)
    
    anomalies = []
    
    # 1. Goles muy altos
    high_scoring = matches_df[
        (matches_df['home_score'] + matches_df['away_score']) > 8
    ]
    if len(high_scoring) > 0:
        print(f"\n⚠️  Partidos con más de 8 goles ({len(high_scoring)}):")
        for _, row in high_scoring.iterrows():
            print(f"   {row['home_team']} {row['home_score']:.0f}-{row['away_score']:.0f} "
                  f"{row['away_team']} ({row['date'].strftime('%d/%m/%Y')})")
            anomalies.append(('high_scoring', row))
    
    # 2. xG muy alto
    high_xg = matches_df[
        (matches_df['xg_home'] + matches_df['xg_away']) > 6
    ]
    if len(high_xg) > 0:
        print(f"\n⚠️  Partidos con xG total > 6 ({len(high_xg)}):")
        for _, row in high_xg.iterrows():
            print(f"   {row['home_team']} vs {row['away_team']} - "
                  f"xG: {row['xg_home']:.2f} - {row['xg_away']:.2f}")
            anomalies.append(('high_xg', row))
    
    # 3. Gran desviación entre goles y xG
    df_temp = matches_df.copy()
    df_temp['home_diff'] = abs(df_temp['home_score'] - df_temp['xg_home'])
    df_temp['away_diff'] = abs(df_temp['away_score'] - df_temp['xg_away'])
    df_temp['total_diff'] = df_temp['home_diff'] + df_temp['away_diff']
    
    large_deviation = df_temp[df_temp['total_diff'] > 4]
    if len(large_deviation) > 0:
        print(f"\n⚠️  Gran desviación goles vs xG (>4) - ({len(large_deviation)}):")
        for _, row in large_deviation.head(5).iterrows():
            print(f"   {row['home_team']} {row['home_score']:.0f} (xG: {row['xg_home']:.2f}) - "
                  f"{row['away_score']:.0f} (xG: {row['xg_away']:.2f}) {row['away_team']}")
            anomalies.append(('xg_deviation', row))
    
    # 4. Partidos sin goles pero alto xG
    no_goals_high_xg = matches_df[
        ((matches_df['home_score'] == 0) & (matches_df['away_score'] == 0)) &
        ((matches_df['xg_home'] + matches_df['xg_away']) > 2.5)
    ]
    if len(no_goals_high_xg) > 0:
        print(f"\n⚠️  0-0 pero con alto xG ({len(no_goals_high_xg)}):")
        for _, row in no_goals_high_xg.iterrows():
            print(f"   {row['home_team']} vs {row['away_team']} - "
                  f"xG: {row['xg_home']:.2f} - {row['xg_away']:.2f}")
            anomalies.append(('high_xg_no_goals', row))
    
    if len(anomalies) == 0:
        print("\n✅ No se detectaron anomalías significativas")
    
    return anomalies


# ============================================================================
# EXPORTAR DATOS PROCESADOS
# ============================================================================

def export_processed_data(matches_df, output_path='processed_matches.xlsx'):
    """
    Exporta datos procesados y limpios
    """
    print(f"\n💾 Exportando datos procesados a {output_path}...")
    
    # Agregar features avanzadas
    df_export = create_advanced_features(matches_df)
    
    # Exportar
    df_export.to_excel(output_path, index=False)
    
    print(f"✅ Datos exportados: {len(df_export)} filas, {len(df_export.columns)} columnas")
    
    return df_export


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("🔧 UTILIDADES DE DATOS - Sistema de Predicción")
    print("="*70)
    
    # Cargar datos
    matches_path = input("\n📁 Path del archivo de partidos: ").strip() or "matches.xlsx"
    
    print(f"\n📂 Cargando {matches_path}...")
    df = pd.read_excel(matches_path)
    
    print(f"✅ Cargados {len(df)} partidos\n")
    
    # Menú
    while True:
        print("\n" + "="*70)
        print("MENÚ DE UTILIDADES")
        print("="*70)
        print("\n1. Validar y limpiar datos")
        print("2. Crear features avanzadas")
        print("3. Análisis exploratorio completo")
        print("4. Detectar anomalías")
        print("5. Exportar datos procesados")
        print("6. Salir")
        
        option = input("\nSeleccioná una opción (1-6): ").strip()
        
        if option == "1":
            df = validate_and_clean_data(df)
        elif option == "2":
            df = create_advanced_features(df)
        elif option == "3":
            exploratory_data_analysis(df)
        elif option == "4":
            detect_anomalies(df)
        elif option == "5":
            output = input("Nombre del archivo de salida (default: processed_matches.xlsx): ").strip()
            output = output if output else "processed_matches.xlsx"
            export_processed_data(df, output)
        elif option == "6":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción inválida")
        
        input("\nPresioná ENTER para continuar...")
