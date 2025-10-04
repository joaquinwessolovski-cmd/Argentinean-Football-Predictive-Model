"""
Script CLI para entrenar y usar el predictor de fútbol argentino
Uso: python cli_predictor.py
"""

from football_predictor import FootballPredictor, format_prediction
import sys

def main():
    print("="*60)
    print("⚽ PREDICTOR DE FÚTBOL ARGENTINO")
    print("="*60)
    
    # Paths de archivos
    matches_path = input("\n📁 Path del archivo de partidos (matches.xlsx): ").strip() or "matches.xlsx"
    stats_path = input("📁 Path del archivo de estadísticas (stats.xlsx): ").strip() or "stats.xlsx"
    
    # Inicializar predictor
    print("\n🔄 Cargando datos...")
    predictor = FootballPredictor()
    
    try:
        matches_df, stats_df = predictor.load_data(matches_path, stats_path)
    except Exception as e:
        print(f"\n❌ Error al cargar archivos: {e}")
        print("\nAsegurate de que los archivos existan y tengan el formato correcto.")
        return
    
    # Entrenar modelos
    print("\n" + "="*60)
    print("ENTRENANDO MODELOS")
    print("="*60)
    
    print("\n1️⃣ Entrenando Gradient Boosting...")
    predictor.train_gradient_boosting(train_split=0.85)
    
    print("\n2️⃣ Entrenando Sistema ELO...")
    predictor.train_elo_system()
    
    print("\n✅ Modelos entrenados exitosamente!")
    
    # Menú principal
    while True:
        print("\n" + "="*60)
        print("MENÚ PRINCIPAL")
        print("="*60)
        print("\n1. Predecir resultado de un partido")
        print("2. Ver ranking ELO")
        print("3. Listar equipos disponibles")
        print("4. Análisis de equipo")
        print("5. Salir")
        
        option = input("\nSeleccioná una opción (1-5): ").strip()
        
        if option == "1":
            predict_match(predictor)
        elif option == "2":
            show_elo_ranking(predictor)
        elif option == "3":
            show_teams(predictor)
        elif option == "4":
            analyze_team(predictor, matches_df)
        elif option == "5":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción inválida")

def predict_match(predictor):
    """Predecir resultado de un partido"""
    print("\n" + "="*60)
    print("PREDICCIÓN DE PARTIDO")
    print("="*60)
    
    teams = predictor.get_team_list()
    
    print(f"\nEquipos disponibles: {len(teams)}")
    print("(Escribí 'lista' para ver todos los equipos)")
    
    home_team = input("\n🏠 Equipo local: ").strip()
    
    if home_team.lower() == 'lista':
        for i, team in enumerate(teams, 1):
            print(f"  {i}. {team}")
        home_team = input("\n🏠 Equipo local: ").strip()
    
    away_team = input("✈️  Equipo visitante: ").strip()
    
    if home_team not in teams:
        print(f"\n❌ '{home_team}' no encontrado en la base de datos")
        return
    
    if away_team not in teams:
        print(f"\n❌ '{away_team}' no encontrado en la base de datos")
        return
    
    if home_team == away_team:
        print("\n❌ Los equipos deben ser diferentes")
        return
    
    print("\n🔮 Calculando predicción...")
    predictions = predictor.predict_match(home_team, away_team, method='all')
    
    print("\n" + "="*60)
    print(f"PREDICCIÓN: {home_team} vs {away_team}")
    print("="*60)
    
    print(format_prediction(predictions))
    
    # Resumen
    if 'ensemble' in predictions:
        main_pred = predictions['ensemble']
        print("\n" + "="*60)
        print("RECOMENDACIÓN FINAL (Ensemble)")
        print("="*60)
    elif 'gradient_boosting' in predictions:
        main_pred = predictions['gradient_boosting']
        print("\n" + "="*60)
        print("RECOMENDACIÓN FINAL (Gradient Boosting)")
        print("="*60)
    else:
        main_pred = predictions['elo']
        print("\n" + "="*60)
        print("RECOMENDACIÓN FINAL (ELO)")
        print("="*60)
    
    probs = [
        ('Victoria Local', main_pred['prob_home']),
        ('Empate', main_pred['prob_draw']),
        ('Victoria Visitante', main_pred['prob_away'])
    ]
    probs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Resultado más probable: {probs[0][0]} ({probs[0][1]*100:.1f}%)")
    
    confidence = probs[0][1] - probs[1][1]
    if confidence > 0.2:
        print("📊 Confianza: ALTA")
    elif confidence > 0.1:
        print("📊 Confianza: MEDIA")
    else:
        print("📊 Confianza: BAJA")

def show_elo_ranking(predictor):
    """Mostrar ranking ELO"""
    print("\n" + "="*60)
    print("RANKING ELO")
    print("="*60)
    
    sorted_elo = sorted(predictor.elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    top = input("\n¿Cuántos equipos querés ver? (default: 20): ").strip()
    top = int(top) if top.isdigit() else 20
    
    print(f"\n{'Pos':<5} {'Equipo':<35} {'Rating ELO':<10}")
    print("-" * 60)
    
    for i, (team, rating) in enumerate(sorted_elo[:top], 1):
        print(f"{i:<5} {team:<35} {rating:>10.0f}")

def show_teams(predictor):
    """Listar todos los equipos"""
    print("\n" + "="*60)
    print("EQUIPOS DISPONIBLES")
    print("="*60)
    
    teams = predictor.get_team_list()
    
    for i, team in enumerate(teams, 1):
        print(f"  {i:>2}. {team}")
    
    print(f"\nTotal: {len(teams)} equipos")

def analyze_team(predictor, matches_df):
    """Analizar rendimiento de un equipo"""
    print("\n" + "="*60)
    print("ANÁLISIS DE EQUIPO")
    print("="*60)
    
    teams = predictor.get_team_list()
    team = input("\n📊 ¿Qué equipo querés analizar?: ").strip()
    
    if team not in teams:
        print(f"\n❌ '{team}' no encontrado en la base de datos")
        return
    
    # Filtrar partidos
    team_matches = matches_df[
        (matches_df['home_team'] == team) | 
        (matches_df['away_team'] == team)
    ].copy()
    
    # Calcular estadísticas
    total = len(team_matches)
    
    wins = len(team_matches[
        ((team_matches['home_team'] == team) & (team_matches['home_score'] > team_matches['away_score'])) |
        ((team_matches['away_team'] == team) & (team_matches['away_score'] > team_matches['home_score']))
    ])
    
    draws = len(team_matches[team_matches['home_score'] == team_matches['away_score']])
    losses = total - wins - draws
    
    # Calcular goles y xG
    goals_for = 0
    goals_against = 0
    xg_for = 0
    xg_against = 0
    
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            goals_for += row['home_score']
            goals_against += row['away_score']
            xg_for += row['xg_home']
            xg_against += row['xg_away']
        else:
            goals_for += row['away_score']
            goals_against += row['home_score']
            xg_for += row['xg_away']
            xg_against += row['xg_home']
    
    # Mostrar análisis
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE {team.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📊 RÉCORD")
    print(f"  Partidos jugados:  {total}")
    print(f"  Victorias:         {wins} ({wins/total*100:.1f}%)")
    print(f"  Empates:           {draws} ({draws/total*100:.1f}%)")
    print(f"  Derrotas:          {losses} ({losses/total*100:.1f}%)")
    
    print(f"\n⚽ GOLES")
    print(f"  Goles a favor:     {goals_for} (promedio: {goals_for/total:.2f})")
    print(f"  Goles en contra:   {goals_against} (promedio: {goals_against/total:.2f})")
    print(f"  Diferencia:        {goals_for - goals_against:+d}")
    
    print(f"\n📈 EXPECTED GOALS (xG)")
    print(f"  xG a favor:        {xg_for:.1f} (promedio: {xg_for/total:.2f})")
    print(f"  xG en contra:      {xg_against:.1f} (promedio: {xg_against/total:.2f})")
    print(f"  Diferencia xG:     {xg_for - xg_against:+.1f}")
    
    print(f"\n🎯 EFICIENCIA")
    efficiency = (goals_for / xg_for * 100) if xg_for > 0 else 0
    print(f"  Conversión:        {efficiency:.1f}% (goles vs xG esperado)")
    
    if efficiency > 110:
        print("  → El equipo está siendo muy eficiente (sobrerindiendo)")
    elif efficiency < 90:
        print("  → El equipo está siendo poco eficiente (subrendiendo)")
    else:
        print("  → El equipo está convirtiendo según lo esperado")
    
    print(f"\n🏆 RATING ELO ACTUAL")
    elo_rating = predictor.elo_ratings.get(team, 1500)
    print(f"  {elo_rating:.0f} puntos")
    
    # Posición en el ranking
    sorted_elo = sorted(predictor.elo_ratings.items(), key=lambda x: x[1], reverse=True)
    position = next((i for i, (t, _) in enumerate(sorted_elo, 1) if t == team), None)
    print(f"  Posición: #{position} de {len(sorted_elo)}")
    
    # Últimos 5 partidos
    print(f"\n📅 ÚLTIMOS 5 PARTIDOS")
    recent = team_matches.sort_values('date', ascending=False).head(5)
    
    for _, row in recent.iterrows():
        date = row['date'].strftime('%d/%m/%Y')
        if row['home_team'] == team:
            opponent = row['away_team']
            score = f"{row['home_score']}-{row['away_score']}"
            xg_display = f"xG: {row['xg_home']:.1f}-{row['xg_away']:.1f}"
            location = "Local"
            if row['home_score'] > row['away_score']:
                result = "✅ Victoria"
            elif row['home_score'] < row['away_score']:
                result = "❌ Derrota"
            else:
                result = "➖ Empate"
        else:
            opponent = row['home_team']
            score = f"{row['away_score']}-{row['home_score']}"
            xg_display = f"xG: {row['xg_away']:.1f}-{row['xg_home']:.1f}"
            location = "Visit"
            if row['away_score'] > row['home_score']:
                result = "✅ Victoria"
            elif row['away_score'] < row['home_score']:
                result = "❌ Derrota"
            else:
                result = "➖ Empate"
        
        print(f"  {date} | {location} vs {opponent:<25} | {score} | {xg_display} | {result}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)