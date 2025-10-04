"""
Ejemplos avanzados de uso del predictor de fútbol argentino
"""

from football_predictor import FootballPredictor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# EJEMPLO 1: Backtesting - Evaluar performance histórica
# ============================================================================

def backtest_model(matches_path, stats_path, test_start_date='2024-06-01'):
    """
    Evalúa la performance del modelo en datos históricos
    """
    print("="*70)
    print("BACKTESTING - Evaluación de Performance Histórica")
    print("="*70)
    
    # Cargar datos completos
    predictor_full = FootballPredictor()
    full_df, _ = predictor_full.load_data(matches_path, stats_path)
    
    # Split temporal
    train_df = full_df[full_df['date'] < test_start_date].copy()
    test_df = full_df[full_df['date'] >= test_start_date].copy()
    
    print(f"\n📊 Train: {len(train_df)} partidos (hasta {test_start_date})")
    print(f"📊 Test: {len(test_df)} partidos (desde {test_start_date})")
    
    # Entrenar solo con datos históricos
    predictor = FootballPredictor()
    predictor.matches_df = train_df
    predictor.train_gradient_boosting(train_split=0.9)  # Usar todo para train
    predictor.train_elo_system()
    
    # Predecir partidos de test
    correct_predictions = 0
    total_log_loss = 0
    predictions_list = []
    
    for idx, row in test_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        try:
            preds = predictor.predict_match(home, away, method='ensemble')
            
            if 'ensemble' not in preds:
                continue
            
            pred = preds['ensemble']
            
            # Resultado real
            if row['home_score'] > row['away_score']:
                actual = 'home'
            elif row['home_score'] < row['away_score']:
                actual = 'away'
            else:
                actual = 'draw'
            
            # Resultado predicho
            probs = {
                'home': pred['prob_home'],
                'draw': pred['prob_draw'],
                'away': pred['prob_away']
            }
            predicted = max(probs, key=probs.get)
            
            # Accuracy
            if predicted == actual:
                correct_predictions += 1
            
            # Log Loss
            actual_prob = probs[actual]
            total_log_loss += -np.log(max(actual_prob, 1e-10))
            
            predictions_list.append({
                'date': row['date'],
                'match': f"{home} vs {away}",
                'actual': actual,
                'predicted': predicted,
                'correct': predicted == actual,
                'prob_home': pred['prob_home'],
                'prob_draw': pred['prob_draw'],
                'prob_away': pred['prob_away']
            })
            
            # Actualizar ELO para próxima predicción
            predictor.update_elo(home, away, row['home_score'], row['away_score'])
            
        except Exception as e:
            print(f"Error en {home} vs {away}: {e}")
            continue
    
    # Resultados
    accuracy = correct_predictions / len(predictions_list) * 100
    avg_log_loss = total_log_loss / len(predictions_list)
    
    print(f"\n{'='*70}")
    print("RESULTADOS DEL BACKTESTING")
    print(f"{'='*70}")
    print(f"✓ Partidos evaluados: {len(predictions_list)}")
    print(f"✓ Accuracy (1X2): {accuracy:.2f}%")
    print(f"✓ Log Loss promedio: {avg_log_loss:.4f}")
    
    # Benchmark: predicción aleatoria sería ~33%
    print(f"\n📊 Mejora vs predicción aleatoria: {accuracy - 33.33:.2f} puntos porcentuales")
    
    return pd.DataFrame(predictions_list)


# ============================================================================
# EJEMPLO 2: Simular torneo completo
# ============================================================================

def simulate_tournament(predictor, upcoming_matches, n_simulations=1000):
    """
    Simula resultados de un torneo usando Monte Carlo
    """
    print("="*70)
    print("SIMULACIÓN DE TORNEO - Método Monte Carlo")
    print("="*70)
    
    teams_points = {}
    
    for _ in range(n_simulations):
        points = {}
        
        for home, away in upcoming_matches:
            preds = predictor.predict_match(home, away, method='ensemble')
            
            if 'ensemble' not in preds:
                continue
            
            pred = preds['ensemble']
            
            # Simular resultado basado en probabilidades
            outcome = np.random.choice(
                ['home', 'draw', 'away'],
                p=[pred['prob_home'], pred['prob_draw'], pred['prob_away']]
            )
            
            # Asignar puntos
            if home not in points:
                points[home] = 0
            if away not in points:
                points[away] = 0
            
            if outcome == 'home':
                points[home] += 3
            elif outcome == 'away':
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1
        
        # Acumular puntos de esta simulación
        for team, pts in points.items():
            if team not in teams_points:
                teams_points[team] = []
            teams_points[team].append(pts)
    
    # Calcular estadísticas
    results = []
    for team, points_list in teams_points.items():
        results.append({
            'Equipo': team,
            'Puntos Esperados': np.mean(points_list),
            'Puntos Min': np.min(points_list),
            'Puntos Max': np.max(points_list),
            'Desv. Estándar': np.std(points_list),
            'P90': np.percentile(points_list, 90),
            'P10': np.percentile(points_list, 10)
        })
    
    results_df = pd.DataFrame(results).sort_values('Puntos Esperados', ascending=False)
    results_df.index = range(1, len(results_df) + 1)
    
    print(f"\n{n_simulations} simulaciones completadas\n")
    print(results_df.to_string())
    
    return results_df


# ============================================================================
# EJEMPLO 3: Value Betting - Encontrar oportunidades de apuesta
# ============================================================================

def find_value_bets(predictor, upcoming_matches, bookmaker_odds, threshold=0.05):
    """
    Identifica apuestas con valor positivo comparando con odds del mercado
    
    bookmaker_odds: dict de {(home, away): {'home': odds, 'draw': odds, 'away': odds}}
    threshold: diferencia mínima de probabilidad para considerar value
    """
    print("="*70)
    print("VALUE BETTING - Búsqueda de Oportunidades")
    print("="*70)
    
    value_bets = []
    
    for home, away in upcoming_matches:
        if (home, away) not in bookmaker_odds:
            continue
        
        # Predicciones del modelo
        preds = predictor.predict_match(home, away, method='ensemble')
        if 'ensemble' not in preds:
            continue
        
        pred = preds['ensemble']
        model_probs = {
            'home': pred['prob_home'],
            'draw': pred['prob_draw'],
            'away': pred['prob_away']
        }
        
        # Probabilidades implícitas del mercado
        odds = bookmaker_odds[(home, away)]
        market_probs = {
            'home': 1 / odds['home'],
            'draw': 1 / odds['draw'],
            'away': 1 / odds['away']
        }
        
        # Buscar value
        for outcome in ['home', 'draw', 'away']:
            model_prob = model_probs[outcome]
            market_prob = market_probs[outcome]
            edge = model_prob - market_prob
            
            if edge > threshold:
                expected_value = (model_prob * odds[outcome]) - 1
                
                value_bets.append({
                    'Partido': f"{home} vs {away}",
                    'Apuesta': outcome.upper(),
                    'Cuota': odds[outcome],
                    'Prob. Modelo': f"{model_prob*100:.1f}%",
                    'Prob. Mercado': f"{market_prob*100:.1f}%",
                    'Edge': f"{edge*100:.1f}%",
                    'EV%': f"{expected_value*100:.1f}%"
                })
    
    if value_bets:
        print(f"\n✅ Encontradas {len(value_bets)} oportunidades de value betting:\n")
        value_df = pd.DataFrame(value_bets)
        print(value_df.to_string(index=False))
    else:
        print("\n❌ No se encontraron value bets con el threshold actual")
    
    return value_bets


# ============================================================================
# EJEMPLO 4: Análisis de tendencias por liga
# ============================================================================

def analyze_league_trends(matches_df, league_name):
    """
    Analiza tendencias específicas de una liga
    """
    print("="*70)
    print(f"ANÁLISIS DE TENDENCIAS - {league_name}")
    print("="*70)
    
    league_matches = matches_df[matches_df['league'] == league_name].copy()
    
    if len(league_matches) == 0:
        print(f"\n❌ No se encontraron partidos de {league_name}")
        return
    
    # Métricas generales
    total_matches = len(league_matches)
    total_goals = (league_matches['home_score'] + league_matches['away_score']).sum()
    avg_goals = total_goals / total_matches
    
    home_wins = len(league_matches[league_matches['home_score'] > league_matches['away_score']])
    draws = len(league_matches[league_matches['home_score'] == league_matches['away_score']])
    away_wins = len(league_matches[league_matches['home_score'] < league_matches['away_score']])
    
    # xG analysis
    total_xg = (league_matches['xg_home'] + league_matches['xg_away']).sum()
    avg_xg = total_xg / total_matches
    
    # Over/Under 2.5
    over_25 = len(league_matches[(league_matches['home_score'] + league_matches['away_score']) > 2.5])
    
    # BTTS (Both Teams To Score)
    btts = len(league_matches[(league_matches['home_score'] > 0) & (league_matches['away_score'] > 0)])
    
    print(f"\n📊 ESTADÍSTICAS GENERALES")
    print(f"  Partidos analizados: {total_matches}")
    print(f"  Goles totales: {total_goals}")
    print(f"  Promedio goles/partido: {avg_goals:.2f}")
    print(f"  Promedio xG/partido: {avg_xg:.2f}")
    
    print(f"\n🏆 DISTRIBUCIÓN DE RESULTADOS")
    print(f"  Victorias locales: {home_wins} ({home_wins/total_matches*100:.1f}%)")
    print(f"  Empates: {draws} ({draws/total_matches*100:.1f}%)")
    print(f"  Victorias visitantes: {away_wins} ({away_wins/total_matches*100:.1f}%)")
    
    print(f"\n⚽ MERCADOS POPULARES")
    print(f"  Over 2.5 goles: {over_25/total_matches*100:.1f}%")
    print(f"  Under 2.5 goles: {(total_matches-over_25)/total_matches*100:.1f}%")
    print(f"  Ambos marcan (BTTS): {btts/total_matches*100:.1f}%")
    
    # Tendencia temporal
    league_matches['month'] = pd.to_datetime(league_matches['date']).dt.to_period('M')
    monthly_avg = league_matches.groupby('month').agg({
        'home_score': 'sum',
        'away_score': 'sum'
    })
    monthly_avg['total_goals'] = monthly_avg['home_score'] + monthly_avg['away_score']
    monthly_avg['matches'] = league_matches.groupby('month').size()
    monthly_avg['avg_goals'] = monthly_avg['total_goals'] / monthly_avg['matches']
    
    print(f"\n📈 TENDENCIA DE GOLES POR MES")
    print(monthly_avg[['matches', 'avg_goals']].tail(6).to_string())


# ============================================================================
# EJEMPLO 5: Optimización de estrategia de apuestas (Kelly Criterion)
# ============================================================================

def kelly_criterion(prob_win, odds, kelly_fraction=0.25):
    """
    Calcula el porcentaje óptimo del bankroll a apostar usando Kelly Criterion
    
    prob_win: probabilidad de ganar (0-1)
    odds: cuota decimal
    kelly_fraction: fracción del Kelly completo (default: 1/4 Kelly para ser conservador)
    """
    # Kelly formula: f = (bp - q) / b
    # donde b = odds - 1, p = prob_win, q = 1 - prob_win
    
    b = odds - 1
    p = prob_win
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Aplicar fracción conservadora
    kelly_bet = kelly * kelly_fraction
    
    # No apostar si Kelly es negativo (no hay edge)
    return max(0, kelly_bet)


def optimal_betting_strategy(predictor, upcoming_matches, bookmaker_odds, bankroll=1000):
    """
    Sugiere estrategia óptima de apuestas usando Kelly Criterion
    """
    print("="*70)
    print("ESTRATEGIA ÓPTIMA DE APUESTAS - Kelly Criterion (1/4 Kelly)")
    print("="*70)
    print(f"\nBankroll inicial: ${bankroll:.2f}\n")
    
    suggestions = []
    
    for home, away in upcoming_matches:
        if (home, away) not in bookmaker_odds:
            continue
        
        preds = predictor.predict_match(home, away, method='ensemble')
        if 'ensemble' not in preds:
            continue
        
        pred = preds['ensemble']
        odds = bookmaker_odds[(home, away)]
        
        outcomes = {
            'home': (pred['prob_home'], odds['home'], 'Victoria Local'),
            'draw': (pred['prob_draw'], odds['draw'], 'Empate'),
            'away': (pred['prob_away'], odds['away'], 'Victoria Visitante')
        }
        
        for outcome_key, (prob, odd, outcome_name) in outcomes.items():
            market_prob = 1 / odd
            
            # Solo considerar si tenemos edge
            if prob > market_prob + 0.03:  # Threshold de 3%
                kelly_pct = kelly_criterion(prob, odd, kelly_fraction=0.25)
                
                if kelly_pct > 0.01:  # Mínimo 1% del bankroll
                    stake = bankroll * kelly_pct
                    potential_profit = stake * (odd - 1)
                    
                    suggestions.append({
                        'Partido': f"{home} vs {away}",
                        'Apuesta': outcome_name,
                        'Cuota': f"{odd:.2f}",
                        'Prob. Modelo': f"{prob*100:.1f}%",
                        'Edge': f"{(prob - market_prob)*100:.1f}%",
                        'Kelly %': f"{kelly_pct*100:.1f}%",
                        'Stake': f"${stake:.2f}",
                        'Ganancia Potencial': f"${potential_profit:.2f}"
                    })
    
    if suggestions:
        df = pd.DataFrame(suggestions).sort_values('Kelly %', ascending=False)
        print(df.to_string(index=False))
        
        total_stake = sum([float(s['Stake'].replace('$', '')) for s in suggestions])
        print(f"\n💰 Total a apostar: ${total_stake:.2f} ({total_stake/bankroll*100:.1f}% del bankroll)")
    else:
        print("❌ No se encontraron oportunidades con suficiente edge")
    
    return suggestions


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Paths a tus archivos
    MATCHES_PATH = "matches.xlsx"
    STATS_PATH = "stats.xlsx"
    
    print("\n🚀 Cargando sistema...")
    predictor = FootballPredictor()
    matches_df, stats_df = predictor.load_data(MATCHES_PATH, STATS_PATH)
    predictor.train_gradient_boosting(train_split=0.85)
    predictor.train_elo_system()
    
    # Ejemplo 1: Backtesting
    print("\n" + "="*70)
    input("Presioná ENTER para ejecutar BACKTESTING...")
    backtest_results = backtest_model("matches.xlsx", "stats.xlsx", test_start_date='2025-01-01')
    
    # Ejemplo 2: Simulación de torneo
    print("\n" + "="*70)
    input("Presioná ENTER para SIMULAR TORNEO...")
    
    # Definir partidos pendientes (ejemplo)
    upcoming = [
        ("River Plate", "Boca Juniors"),
        ("Racing", "Independiente"),
        ("San Lorenzo", "Huracán"),
        # Agregar más partidos...
    ]
    
    tournament_results = simulate_tournament(predictor, upcoming, n_simulations=1000)
    
    # Ejemplo 3: Value betting
    print("\n" + "="*70)
    input("Presioná ENTER para buscar VALUE BETS...")
    
    # Odds de ejemplo (reemplazar con odds reales)
    example_odds = {
        ("River Plate", "Boca Juniors"): {'home': 2.10, 'draw': 3.20, 'away': 3.50},
        ("Racing", "Independiente"): {'home': 1.90, 'draw': 3.30, 'away': 4.00},
    }
    
    value_bets = find_value_bets(predictor, upcoming, example_odds, threshold=0.05)
    
    # Ejemplo 4: Análisis de liga
    print("\n" + "="*70)
    input("Presioná ENTER para ANALIZAR LIGA...")
    analyze_league_trends(matches_df, matches_df['league'].iloc[0])
    
    # Ejemplo 5: Estrategia óptima
    print("\n" + "="*70)
    input("Presioná ENTER para calcular ESTRATEGIA ÓPTIMA...")
    betting_strategy = optimal_betting_strategy(predictor, upcoming, example_odds, bankroll=1000)
    
    print("\n" + "="*70)
    print("✅ Todos los ejemplos ejecutados exitosamente!")
    print("="*70)