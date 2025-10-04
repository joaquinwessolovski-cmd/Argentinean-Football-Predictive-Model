import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from football_predictor import FootballPredictor
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Configuración de página
st.set_page_config(
    page_title="Predictor Fútbol Argentino",
    page_icon="⚽",
    layout="wide"
)

# Clásicos del fútbol argentino
CLASICOS = {
    "Superclásico": ("River Plate", "Boca Juniors"),
    "Clásico de Avellaneda": ("Racing Club", "Independiente"),
    "Clásico de Boedo": ("San Lorenzo", "Huracán"),
    "Clásico Rosarino": ("Rosario Central", "Newell's OB"),
    "Clásico Platense": ("Gimnasia–LP", "Estudiantes–LP"),
    "Clásico Cordobes": ("Belgrano", "Talleres"),
    "Clásico Santafesino": ("Colón", "Unión"),
    "Clásico de Zona Norte": ("Tigre", "Platense"),
    "Clásico del Sur": ("Banfield", "Lanús"),
    "Clásico Cuyano": ("Godoy Cruz", "San Martín de San Juan"),
}

# Inicializar session state
if 'predictor' not in st.session_state:
    st.session_state['predictor'] = None
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'elo_history' not in st.session_state:
    st.session_state['elo_history'] = {}
if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = {}

def load_default_data():
    """Intenta cargar datos por defecto si existen"""
    default_matches = "matches.xlsx"
    default_stats = "stats.xlsx"
    
    if os.path.exists(default_matches) and os.path.exists(default_stats):
        return default_matches, default_stats
    return None, None

def train_elo_with_history(predictor, matches_df):
    """Entrena ELO y guarda el historial completo por fecha"""
    predictor.initialize_elo()
    
    elo_history = {team: [] for team in predictor.elo_ratings.keys()}
    
    for _, row in matches_df.iterrows():
        predictor.update_elo(
            row['home_team'], 
            row['away_team'],
            row['home_score'],
            row['away_score']
        )
        
        for team, rating in predictor.elo_ratings.items():
            elo_history[team].append({'date': row['date'], 'elo': rating})
    
    return elo_history

def calculate_model_metrics(predictor, matches_df, split_ratio=0.85):
    """Calcula métricas de evaluación de los modelos"""
    metrics = {}
    
    # Split temporal
    split_idx = int(len(matches_df) * split_ratio)
    train_df = matches_df[:split_idx]
    test_df = matches_df[split_idx:]
    
    if len(test_df) < 10:
        return metrics
    
    # Evaluar predicciones
    y_true = []
    gb_preds = []
    elo_preds = []
    ensemble_preds = []
    
    for _, row in test_df.iterrows():
        # Resultado real
        if row['home_score'] > row['away_score']:
            y_true.append(0)  # Local
        elif row['home_score'] < row['away_score']:
            y_true.append(2)  # Visitante
        else:
            y_true.append(1)  # Empate
        
        # Predicciones
        try:
            preds = predictor.predict_match(row['home_team'], row['away_team'], method='all')
            
            if 'gradient_boosting' in preds:
                gb_pred = preds['gradient_boosting']
                gb_probs = [gb_pred['prob_home'], gb_pred['prob_draw'], gb_pred['prob_away']]
                gb_preds.append(gb_probs)
            
            if 'elo' in preds:
                elo_pred = preds['elo']
                elo_probs = [elo_pred['prob_home'], elo_pred['prob_draw'], elo_pred['prob_away']]
                elo_preds.append(elo_probs)
            
            if 'ensemble' in preds:
                ens_pred = preds['ensemble']
                ens_probs = [ens_pred['prob_home'], ens_pred['prob_draw'], ens_pred['prob_away']]
                ensemble_preds.append(ens_probs)
        except:
            continue
    
    # Calcular métricas
    if len(gb_preds) > 0:
        gb_predictions = [np.argmax(p) for p in gb_preds]
        metrics['gradient_boosting'] = {
            'accuracy': accuracy_score(y_true[:len(gb_predictions)], gb_predictions),
            'log_loss': log_loss(y_true[:len(gb_predictions)], gb_preds)
        }
    
    if len(elo_preds) > 0:
        elo_predictions = [np.argmax(p) for p in elo_preds]
        metrics['elo'] = {
            'accuracy': accuracy_score(y_true[:len(elo_predictions)], elo_predictions),
            'log_loss': log_loss(y_true[:len(elo_predictions)], elo_preds)
        }
    
    if len(ensemble_preds) > 0:
        ens_predictions = [np.argmax(p) for p in ensemble_preds]
        metrics['ensemble'] = {
            'accuracy': accuracy_score(y_true[:len(ens_predictions)], ens_predictions),
            'log_loss': log_loss(y_true[:len(ens_predictions)], ensemble_preds)
        }
    
    return metrics

def simulate_league_tournament(predictor, matches, n_simulations=1000):
    """Simula torneo de liga completo"""
    team_points = {}
    team_positions = {}
    
    for _ in range(n_simulations):
        points = {}
        
        for home, away in matches:
            preds = predictor.predict_match(home, away, method='ensemble')
            if 'ensemble' not in preds:
                preds = predictor.predict_match(home, away, method='elo')
            
            pred = preds.get('ensemble', preds.get('elo'))
            
            outcome = np.random.choice(
                ['home', 'draw', 'away'],
                p=[pred['prob_home'], pred['prob_draw'], pred['prob_away']]
            )
            
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
        
        sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
        
        for position, (team, pts) in enumerate(sorted_teams, 1):
            if team not in team_points:
                team_points[team] = []
                team_positions[team] = []
            
            team_points[team].append(pts)
            team_positions[team].append(position)
    
    return team_points, team_positions

def simulate_knockout_round(predictor, matches, neutral_venue=False):
    """Simula una ronda de eliminación directa"""
    winners = []
    results = []
    
    for home, away in matches:
        preds = predictor.predict_match(home, away, method='ensemble')
        if 'ensemble' not in preds:
            preds = predictor.predict_match(home, away, method='elo')
        
        pred = preds.get('ensemble', preds.get('elo'))
        
        # En cancha neutral, eliminar ventaja de local
        if neutral_venue:
            avg_prob = (pred['prob_home'] + pred['prob_away']) / 2
            prob_home = avg_prob
            prob_away = avg_prob
            prob_draw = pred['prob_draw']
            total = prob_home + prob_draw + prob_away
            prob_home /= total
            prob_draw /= total
            prob_away /= total
        else:
            prob_home = pred['prob_home']
            prob_draw = pred['prob_draw']
            prob_away = pred['prob_away']
        
        # En eliminación no hay empates (penales)
        if prob_home > prob_away:
            winner = home
        else:
            winner = away
        
        winners.append(winner)
        results.append({
            'home': home,
            'away': away,
            'winner': winner,
            'prob_home': prob_home,
            'prob_away': prob_away
        })
    
    return winners, results

def simulate_knockout_tournament(predictor, bracket):
    """Simula torneo de eliminación directa"""
    results = {}
    
    for round_name, matches in bracket.items():
        results[round_name] = []
        
        for home, away in matches:
            preds = predictor.predict_match(home, away, method='ensemble')
            if 'ensemble' not in preds:
                preds = predictor.predict_match(home, away, method='elo')
            
            pred = preds.get('ensemble', preds.get('elo'))
            
            if pred['prob_home'] > pred['prob_away']:
                winner = home
            else:
                winner = away
            
            results[round_name].append({
                'home': home,
                'away': away,
                'winner': winner,
                'prob_home': pred['prob_home'],
                'prob_away': pred['prob_away']
            })
    
    return results
                
                
def simulate_group_stage_knockout(predictor, zone_a, zone_b, n_simulations=1000, neutral_playoffs=False):
    """Simula torneo con fase de grupos + eliminación directa"""
    champion_count = {}
    
    for _ in range(n_simulations):
        # Fase de grupos - Zona A
        zone_a_matches = [(t1, t2) for t1 in zone_a for t2 in zone_a if t1 != t2]
        zone_a_points, _ = simulate_league_tournament(predictor, zone_a_matches, n_simulations=1)
        zone_a_standings = sorted(zone_a_points.items(), key=lambda x: np.mean(x[1]), reverse=True)
        zone_a_qualified = [team for team, _ in zone_a_standings[:len(zone_a)//2]]
        
        # Fase de grupos - Zona B
        zone_b_matches = [(t1, t2) for t1 in zone_b for t2 in zone_b if t1 != t2]
        zone_b_points, _ = simulate_league_tournament(predictor, zone_b_matches, n_simulations=1)
        zone_b_standings = sorted(zone_b_points.items(), key=lambda x: np.mean(x[1]), reverse=True)
        zone_b_qualified = [team for team, _ in zone_b_standings[:len(zone_b)//2]]
        
        # Playoffs (cruzados: 1A vs 2B, 2A vs 1B, etc)
        all_qualified = zone_a_qualified + zone_b_qualified
        
        # Semifinales
        if len(all_qualified) >= 4:
            semi_matches = [
                (zone_a_qualified[0], zone_b_qualified[1] if len(zone_b_qualified) > 1 else zone_b_qualified[0]),
                (zone_b_qualified[0], zone_a_qualified[1] if len(zone_a_qualified) > 1 else zone_a_qualified[0])
            ]
            semi_winners, _ = simulate_knockout_round(predictor, semi_matches, neutral_playoffs)
            
            # Final
            if len(semi_winners) == 2:
                final_match = [(semi_winners[0], semi_winners[1])]
                final_winner, _ = simulate_knockout_round(predictor, final_match, neutral_playoffs)
                champion = final_winner[0]
                
                champion_count[champion] = champion_count.get(champion, 0) + 1
    
    return champion_count

def kelly_criterion(prob, odds, fraction=0.25):
    """Calcula stake óptimo usando Kelly Criterion"""
    if odds <= 1:
        return 0
    
    b = odds - 1
    q = 1 - prob
    kelly = (b * prob - q) / b
    
    return max(0, kelly * fraction)

def calculate_value_bets(predictor, matches, bookmaker_odds, threshold=0.03):
    """Identifica value bets"""
    value_bets = []
    
    for home, away in matches:
        if (home, away) not in bookmaker_odds:
            continue
        
        preds = predictor.predict_match(home, away, method='ensemble')
        if 'ensemble' not in preds:
            preds = predictor.predict_match(home, away, method='elo')
        
        pred = preds.get('ensemble', preds.get('elo'))
        odds = bookmaker_odds[(home, away)]
        
        outcomes = {
            'home': (pred['prob_home'], odds.get('home', 0), 'Victoria Local'),
            'draw': (pred['prob_draw'], odds.get('draw', 0), 'Empate'),
            'away': (pred['prob_away'], odds.get('away', 0), 'Victoria Visitante')
        }
        
        for outcome_key, (prob, odd, outcome_name) in outcomes.items():
            if odd > 0:
                market_prob = 1 / odd
                edge = prob - market_prob
                
                if edge > threshold:
                    kelly_pct = kelly_criterion(prob, odd)
                    
                    if kelly_pct > 0.01:
                        value_bets.append({
                            'Partido': f"{home} vs {away}",
                            'Apuesta': outcome_name,
                            'Cuota': f"{odd:.2f}",
                            'Prob. Modelo': f"{prob*100:.1f}%",
                            'Prob. Mercado': f"{market_prob*100:.1f}%",
                            'Edge': f"{edge*100:.1f}%",
                            'Kelly %': f"{kelly_pct*100:.1f}%",
                            'EV': f"{((prob * odd) - 1)*100:.1f}%"
                        })
    
    return value_bets

# Título
st.title("⚽ Predictor de Fútbol Argentino")
st.markdown("Sistema de predicción basado en xG, Gradient Boosting y ELO dinámico")

# Sidebar
with st.sidebar:
    st.header("📁 Gestión de Datos")
    
    use_default = st.checkbox("Usar datos precargados del repositorio", value=True)
    
    if use_default:
        default_matches, default_stats = load_default_data()
        if default_matches and default_stats:
            st.success("✅ Datos precargados encontrados")
            matches_file = default_matches
            stats_file = default_stats
            auto_load = True
        else:
            st.warning("⚠️ No se encontraron datos precargados")
            use_default = False
            auto_load = False
    
    if not use_default:
        matches_file = st.file_uploader("Archivo de Partidos (matches.xlsx)", type=['xlsx'])
        stats_file = st.file_uploader("Archivo de Estadísticas (stats.xlsx)", type=['xlsx'])
        auto_load = False
    
    # Cargar más datos
    st.markdown("---")
    st.subheader("📥 Actualizar Datos")
    new_matches_file = st.file_uploader("Agregar más partidos", type=['xlsx'], key='new_data')
    
    if new_matches_file and st.session_state['trained']:
        if st.button("🔄 Actualizar y Re-entrenar"):
            with st.spinner("Actualizando datos y re-entrenando..."):
                try:
                    new_matches = pd.read_excel(new_matches_file)
                    current_matches = st.session_state['matches_df']
                    
                    # Combinar datos
                    combined_matches = pd.concat([current_matches, new_matches], ignore_index=True)
                    combined_matches = combined_matches.drop_duplicates(subset=['date', 'home_team', 'away_team'])
                    combined_matches = combined_matches.sort_values('date').reset_index(drop=True)
                    
                    # Re-entrenar
                    predictor = st.session_state['predictor']
                    predictor.matches_df = combined_matches
                    predictor.train_gradient_boosting(train_split=0.85)
                    elo_history = train_elo_with_history(predictor, combined_matches)
                    metrics = calculate_model_metrics(predictor, combined_matches)
                    
                    st.session_state['matches_df'] = combined_matches
                    st.session_state['elo_history'] = elo_history
                    st.session_state['model_metrics'] = metrics
                    
                    st.success(f"✅ Datos actualizados! Ahora tenés {len(combined_matches)} partidos")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if (matches_file and stats_file) or auto_load:
        if st.button("🚀 Entrenar Modelos") or (auto_load and not st.session_state['trained']):
            with st.spinner("Entrenando modelos..."):
                try:
                    if not auto_load:
                        with open("temp_matches.xlsx", "wb") as f:
                            f.write(matches_file.getbuffer())
                        with open("temp_stats.xlsx", "wb") as f:
                            f.write(stats_file.getbuffer())
                        matches_path = "temp_matches.xlsx"
                        stats_path = "temp_stats.xlsx"
                    else:
                        matches_path = matches_file
                        stats_path = stats_file
                    
                    predictor = FootballPredictor()
                    matches_df, stats_df = predictor.load_data(matches_path, stats_path)
                    
                    predictor.train_gradient_boosting(train_split=0.85)
                    elo_history = train_elo_with_history(predictor, matches_df)
                    metrics = calculate_model_metrics(predictor, matches_df)
                    
                    st.session_state['predictor'] = predictor
                    st.session_state['matches_df'] = matches_df
                    st.session_state['stats_df'] = stats_df
                    st.session_state['trained'] = True
                    st.session_state['elo_history'] = elo_history
                    st.session_state['model_metrics'] = metrics
                    
                    st.success("✅ Modelos entrenados exitosamente!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📊 Estado del Sistema")
    if st.session_state['trained']:
        st.success("✅ Sistema listo")
        st.metric("Partidos cargados", len(st.session_state['matches_df']))
        st.metric("Equipos", len(st.session_state['predictor'].get_team_list()))
        
        # Mostrar métricas si existen
        if st.session_state['model_metrics']:
            with st.expander("📈 Métricas de Modelos"):
                for model, metrics in st.session_state['model_metrics'].items():
                    st.markdown(f"**{model.replace('_', ' ').title()}**")
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                    st.metric("Log Loss", f"{metrics['log_loss']:.3f}")
    else:
        st.warning("⚠️ Cargá los datos y entrenár")

# Main content
if st.session_state['trained']:
    predictor = st.session_state['predictor']
    matches_df = st.session_state['matches_df']
    teams = predictor.get_team_list()
    
    # Obtener divisiones si existen
    divisions = []
    if 'division' in matches_df.columns:
        # Crear mapping de equipo -> división
        team_division = {}
        for _, row in matches_df.iterrows():
            if pd.notna(row.get('division')):
                team_division[row['home_team']] = row['division']
                team_division[row['away_team']] = row['division']
        
        divisions = sorted(list(set(team_division.values())))
    
    tabs = st.tabs([
        "🎯 Predicción Individual", 
        "📋 Predicción Múltiple",
        "🏆 Simulador de Torneos",
        "💰 Apuestas & Value Bets",
        "⚙️ Configuración de Modelos",
        "📈 Rankings & ELO",
        "⚔️ Head to Head",
        "🔥 Clásicos",
        "📊 Análisis & Histórico"
    ])
    
    # TAB 1: PREDICCIÓN INDIVIDUAL
    with tabs[0]:
        st.header("Predecir Resultado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Equipo Local", teams, key='home')
        
        with col2:
            away_team = st.selectbox("Equipo Visitante", teams, key='away')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            method_gb = st.checkbox("Gradient Boosting + Poisson", value=True)
        with col2:
            method_elo = st.checkbox("Sistema ELO", value=True)
        with col3:
            method_ensemble = st.checkbox("Ensemble", value=True)
        
        if st.button("🔮 Predecir", type="primary", use_container_width=True):
            if home_team == away_team:
                st.error("⚠️ Seleccioná equipos diferentes")
            else:
                with st.spinner("Calculando predicción..."):
                    predictions = predictor.predict_match(home_team, away_team, method='all')
                    
                    st.success(f"### {home_team} vs {away_team}")
                    
                    methods_to_show = []
                    if method_gb and 'gradient_boosting' in predictions:
                        methods_to_show.append(('Gradient Boosting', 'gradient_boosting'))
                    if method_elo and 'elo' in predictions:
                        methods_to_show.append(('ELO', 'elo'))
                    if method_ensemble and 'ensemble' in predictions:
                        methods_to_show.append(('Ensemble', 'ensemble'))
                    
                    if len(methods_to_show) > 0:
                        cols = st.columns(len(methods_to_show))
                        
                        for idx, (name, key) in enumerate(methods_to_show):
                            with cols[idx]:
                                st.markdown(f"#### {name}")
                                pred = predictions[key]
                                
                                if 'xg_home' in pred:
                                    st.metric("xG Local", f"{pred['xg_home']:.2f}")
                                    st.metric("xG Visitante", f"{pred['xg_away']:.2f}")
                                
                                if 'home_elo' in pred:
                                    st.metric("ELO Local", f"{pred['home_elo']:.0f}")
                                    st.metric("ELO Visit", f"{pred['away_elo']:.0f}")
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=['Local', 'Empate', 'Visitante'],
                                        y=[pred['prob_home']*100, pred['prob_draw']*100, pred['prob_away']*100],
                                        marker_color=['#00D9FF', '#FFB800', '#FF4B4B'],
                                        text=[f"{pred['prob_home']*100:.1f}%", 
                                              f"{pred['prob_draw']*100:.1f}%", 
                                              f"{pred['prob_away']*100:.1f}%"],
                                        textposition='auto',
                                    )
                                ])
                                fig.update_layout(
                                    title="Probabilidades",
                                    yaxis_title="Probabilidad (%)",
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 💡 Recomendación")
                        
                        if 'ensemble' in predictions:
                            main_pred = predictions['ensemble']
                        elif 'gradient_boosting' in predictions:
                            main_pred = predictions['gradient_boosting']
                        else:
                            main_pred = predictions['elo']
                        
                        probs = [
                            ('Local', main_pred['prob_home']),
                            ('Empate', main_pred['prob_draw']),
                            ('Visitante', main_pred['prob_away'])
                        ]
                        probs.sort(key=lambda x: x[1], reverse=True)
                        
                        confidence = probs[0][1] - probs[1][1]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Resultado más probable", probs[0][0])
                        with col2:
                            st.metric("Probabilidad", f"{probs[0][1]*100:.1f}%")
                        with col3:
                            confidence_level = "Alta" if confidence > 0.2 else "Media" if confidence > 0.1 else "Baja"
                            st.metric("Confianza", confidence_level)
    
    # TAB 2: PREDICCIÓN MÚLTIPLE
    with tabs[1]:
        st.header("Predicción de Múltiples Partidos")
        
        st.markdown("Predecí varios partidos a la vez. Podés ingresar los enfrentamientos manualmente o cargar un archivo.")
        
        input_method = st.radio("Método de entrada", ["Manual", "Cargar CSV"])
        
        matches_to_predict = []
        
        if input_method == "Manual":
            st.subheader("Ingresá los partidos")
            
            num_matches = st.number_input("¿Cuántos partidos querés predecir?", min_value=1, max_value=20, value=5)
            
            for i in range(num_matches):
                col1, col2 = st.columns(2)
                with col1:
                    home = st.selectbox(f"Local {i+1}", teams, key=f"multi_home_{i}")
                with col2:
                    away = st.selectbox(f"Visitante {i+1}", teams, key=f"multi_away_{i}")
                
                if home != away:
                    matches_to_predict.append((home, away))
        
        else:
            st.subheader("Cargar archivo CSV")
            st.markdown("El archivo debe tener columnas: `home_team`, `away_team`")
            
            csv_file = st.file_uploader("Subir CSV", type=['csv'])
            if csv_file:
                try:
                    df_matches = pd.read_csv(csv_file)
                    matches_to_predict = [(row['home_team'], row['away_team']) 
                                         for _, row in df_matches.iterrows()]
                    st.success(f"✅ {len(matches_to_predict)} partidos cargados")
                except Exception as e:
                    st.error(f"Error al cargar CSV: {e}")
        
        if st.button("🔮 Predecir Todos", type="primary") and len(matches_to_predict) > 0:
            st.subheader("Resultados")
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, (home, away) in enumerate(matches_to_predict):
                try:
                    preds = predictor.predict_match(home, away, method='all')
                    
                    if 'ensemble' in preds:
                        pred = preds['ensemble']
                    elif 'gradient_boosting' in preds:
                        pred = preds['gradient_boosting']
                    else:
                        pred = preds['elo']
                    
                    probs = {
                        'Local': pred['prob_home'],
                        'Empate': pred['prob_draw'],
                        'Visitante': pred['prob_away']
                    }
                    most_likely = max(probs, key=probs.get)
                    confidence = probs[most_likely]
                    
                    results.append({
                        'Partido': f"{home} vs {away}",
                        'Prob Local': f"{pred['prob_home']*100:.1f}%",
                        'Prob Empate': f"{pred['prob_draw']*100:.1f}%",
                        'Prob Visitante': f"{pred['prob_away']*100:.1f}%",
                        'Predicción': most_likely,
                        'Confianza': f"{confidence*100:.1f}%"
                    })
                except Exception as e:
                    results.append({
                        'Partido': f"{home} vs {away}",
                        'Error': str(e)
                    })
                
                progress_bar.progress((idx + 1) / len(matches_to_predict))
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Resultados CSV",
                data=csv,
                file_name="predicciones_multiples.csv",
                mime="text/csv"
            )
    
    # TAB 3: SIMULADOR DE TORNEOS
    with tabs[2]:
        st.header("Simulador de Torneos")
        
        tournament_type = st.radio("Tipo de torneo", ["Liga (Todos contra todos)", "Eliminación Directa"])
        
        if tournament_type == "Liga (Todos contra todos)":
            st.subheader("Simulación de Liga")
            
            st.markdown("Seleccioná los equipos participantes y se generarán todos los partidos (ida y vuelta)")
            
            selected_teams = st.multiselect("Equipos del torneo", teams, default=teams[:6] if len(teams) >= 6 else teams)
            
            if len(selected_teams) >= 2:
                matches = []
                for home in selected_teams:
                    for away in selected_teams:
                        if home != away:
                            matches.append((home, away))
                
                st.info(f"📊 Total de partidos: {len(matches)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    n_sims = st.slider("Simulaciones", 100, 10000, 1000, step=100)
                with col2:
                    show_fixtures = st.checkbox("Mostrar fixture completo", value=False)
                
                if show_fixtures:
                    st.subheader("Fixture del Torneo")
                    fixture_df = pd.DataFrame(matches, columns=['Local', 'Visitante'])
                    fixture_df.index += 1
                    st.dataframe(fixture_df, use_container_width=True)
                
                if st.button("🎲 Simular Torneo", type="primary"):
                    with st.spinner(f"Simulando {n_sims} torneos..."):
                        team_points, team_positions = simulate_league_tournament(predictor, matches, n_sims)
                        
                        results = []
                        for team in selected_teams:
                            if team in team_points:
                                results.append({
                                    'Equipo': team,
                                    'Puntos Promedio': f"{np.mean(team_points[team]):.1f}",
                                    'Puntos Min': int(np.min(team_points[team])),
                                    'Puntos Max': int(np.max(team_points[team])),
                                    'Posición Promedio': f"{np.mean(team_positions[team]):.1f}",
                                    '% Campeón': f"{(team_positions[team].count(1) / n_sims * 100):.1f}%",
                                    '% Top 4': f"{(sum(1 for p in team_positions[team] if p <= 4) / n_sims * 100):.1f}%"
                                })
                        
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values('Puntos Promedio', ascending=False, key=lambda x: x.str.replace(',', '.').astype(float))
                        results_df.index = range(1, len(results_df) + 1)
                        
                        st.success(f"✅ {n_sims} simulaciones completadas")
                        st.dataframe(results_df, use_container_width=True)
                        
                        st.subheader("Distribución de Posiciones Finales")
                        
                        fig = go.Figure()
                        for team in selected_teams[:5]:
                            if team in team_positions:
                                positions_count = [team_positions[team].count(i) for i in range(1, len(selected_teams)+1)]
                                fig.add_trace(go.Bar(
                                    name=team,
                                    x=list(range(1, len(selected_teams)+1)),
                                    y=positions_count
                                ))
                        
                        fig.update_layout(
                            title="Frecuencia de cada posición (Top 5 equipos)",
                            xaxis_title="Posición Final",
                            yaxis_title="Frecuencia",
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.subheader("Simulación de Eliminación Directa")
            
            st.markdown("Definí las llaves del torneo (debe ser potencia de 2: 4, 8, 16 equipos)")
            
            num_teams_knockout = st.selectbox("Número de equipos", [4, 8, 16])
            
            selected_teams_ko = st.multiselect(
                "Seleccioná los equipos", 
                teams, 
                max_selections=num_teams_knockout
            )
            
            if len(selected_teams_ko) == num_teams_knockout:
                st.info("🎯 Orden de equipos = Orden del bracket")
                
                bracket = {}
                
                if num_teams_knockout == 4:
                    bracket['Semifinales'] = [
                        (selected_teams_ko[0], selected_teams_ko[1]),
                        (selected_teams_ko[2], selected_teams_ko[3])
                    ]
                elif num_teams_knockout == 8:
                    bracket['Cuartos de Final'] = [
                        (selected_teams_ko[i], selected_teams_ko[i+1]) 
                        for i in range(0, 8, 2)
                    ]
                elif num_teams_knockout == 16:
                    bracket['Octavos de Final'] = [
                        (selected_teams_ko[i], selected_teams_ko[i+1]) 
                        for i in range(0, 16, 2)
                    ]
                
                if st.button("🏆 Simular Eliminación Directa", type="primary"):
                    with st.spinner("Simulando torneo..."):
                        results = simulate_knockout_tournament(predictor, bracket)
                        
                        for round_name, matches_result in results.items():
                            st.subheader(round_name)
                            
                            for match in matches_result:
                                col1, col2, col3 = st.columns([2, 1, 2])
                                
                                with col1:  
                                    st.markdown(f"**{match['home']}**")
                                    st.caption(f"{match['prob_home']*100:.1f}%")
                                
                                with col2:
                                    st.markdown("vs")
                                    if match['winner'] == match['home']:
                                        st.success("→")
                                    else:
                                        st.success("←")
                                
                                with col3:
                                    st.markdown(f"**{match['away']}**")
                                    st.caption(f"{match['prob_away']*100:.1f}%")
                            
                            st.markdown("---")
    
    # TAB 6: RANKINGS & ELO
    with tabs[5]:
        st.header("Rankings y Evolución ELO")
        
        sub_tabs = st.tabs(["📊 Ranking Actual", "📈 Evolución ELO", "⚖️ Comparar Equipos"])
        
        with sub_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Ranking ELO")
                elo_df = pd.DataFrame([
                    {'Equipo': team, 'Rating ELO': rating}
                    for team, rating in sorted(predictor.elo_ratings.items(), key=lambda x: x[1], reverse=True)
                ]).reset_index(drop=True)
                elo_df.index += 1
                
                st.dataframe(elo_df.head(20), use_container_width=True, height=600)
            
            with col2:
                st.subheader("📊 Distribución ELO")
                fig = px.histogram(elo_df, x='Rating ELO', nbins=30, title="Distribución de Ratings ELO")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📈 Top 10 ELO")
                fig = px.bar(elo_df.head(10), x='Rating ELO', y='Equipo', orientation='h', title="Top 10 Equipos por ELO")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[1]:
            st.subheader("📈 Evolución del Rating ELO")
            
            selected_teams_evo = st.multiselect(
                "Seleccioná equipos para ver su evolución",
                teams,
                default=teams[:3] if len(teams) >= 3 else teams
            )
            
            if selected_teams_evo and st.session_state['elo_history']:
                fig = go.Figure()
                
                for team in selected_teams_evo:
                    if team in st.session_state['elo_history']:
                        history = st.session_state['elo_history'][team]
                        dates = [h['date'] for h in history]
                        elos = [h['elo'] for h in history]
                        
                        fig.add_trace(go.Scatter(
                            x=dates, y=elos, mode='lines', name=team, line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title="Evolución del Rating ELO",
                    xaxis_title="Fecha",
                    yaxis_title="Rating ELO",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📊 Estadísticas de Evolución")
                
                for team in selected_teams_evo:
                    if team in st.session_state['elo_history']:
                        history = st.session_state['elo_history'][team]
                        if len(history) > 0:
                            initial_elo = history[0]['elo']
                            current_elo = history[-1]['elo']
                            max_elo = max(h['elo'] for h in history)
                            min_elo = min(h['elo'] for h in history)
                            change = current_elo - initial_elo
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric(team, f"{current_elo:.0f}")
                            with col2:
                                st.metric("Cambio Total", f"{change:+.0f}")
                            with col3:
                                st.metric("ELO Máximo", f"{max_elo:.0f}")
                            with col4:
                                st.metric("ELO Mínimo", f"{min_elo:.0f}")
                            with col5:
                                volatility = np.std([h['elo'] for h in history])
                                st.metric("Volatilidad", f"{volatility:.0f}")
        
        with sub_tabs[2]:
            st.subheader("⚖️ Comparar Equipos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Equipo 1", teams, key="compare1")
            with col2:
                team2 = st.selectbox("Equipo 2", teams, key="compare2")
            
            if team1 != team2:
                elo1 = predictor.elo_ratings.get(team1, 1500)
                elo2 = predictor.elo_ratings.get(team2, 1500)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(team1, f"{elo1:.0f}")
                with col2:
                    diff = elo1 - elo2
                    st.metric("Diferencia", f"{abs(diff):.0f}", delta=f"{team1 if diff > 0 else team2} superior")
                with col3:
                    st.metric(team2, f"{elo2:.0f}")
                
                st.subheader("Historial de Enfrentamientos")
                
                h2h = matches_df[
                    ((matches_df['home_team'] == team1) & (matches_df['away_team'] == team2)) |
                    ((matches_df['home_team'] == team2) & (matches_df['away_team'] == team1))
                ]
                
                if len(h2h) > 0:
                    wins_team1 = len(h2h[
                        ((h2h['home_team'] == team1) & (h2h['home_score'] > h2h['away_score'])) |
                        ((h2h['away_team'] == team1) & (h2h['away_score'] > h2h['home_score']))
                    ])
                    wins_team2 = len(h2h[
                        ((h2h['home_team'] == team2) & (h2h['home_score'] > h2h['away_score'])) |
                        ((h2h['away_team'] == team2) & (h2h['away_score'] > h2h['home_score']))
                    ])
                    draws = len(h2h) - wins_team1 - wins_team2
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Partidos", len(h2h))
                    with col2:
                        st.metric(f"Victorias {team1}", wins_team1)
                    with col3:
                        st.metric("Empates", draws)
                    with col4:
                        st.metric(f"Victorias {team2}", wins_team2)
                else:
                    st.info("No hay enfrentamientos directos en el historial")
    
    # TAB 7: HEAD TO HEAD
    with tabs[6]:
        st.header("⚔️ Head to Head - Enfrentamientos Directos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            h2h_team1 = st.selectbox("Equipo 1", teams, key="h2h_team1")
        with col2:
            h2h_team2 = st.selectbox("Equipo 2", teams, key="h2h_team2")
        
        if h2h_team1 != h2h_team2:
            h2h_matches = matches_df[
                ((matches_df['home_team'] == h2h_team1) & (matches_df['away_team'] == h2h_team2)) |
                ((matches_df['home_team'] == h2h_team2) & (matches_df['away_team'] == h2h_team1))
            ].sort_values('date', ascending=False)
            
            if len(h2h_matches) > 0:
                st.success(f"📊 {len(h2h_matches)} enfrentamientos encontrados")
                
                st.subheader("Resumen del Enfrentamiento")
                
                wins_1 = len(h2h_matches[
                    ((h2h_matches['home_team'] == h2h_team1) & (h2h_matches['home_score'] > h2h_matches['away_score'])) |
                    ((h2h_matches['away_team'] == h2h_team1) & (h2h_matches['away_score'] > h2h_matches['home_score']))
                ])
                wins_2 = len(h2h_matches[
                    ((h2h_matches['home_team'] == h2h_team2) & (h2h_matches['home_score'] > h2h_matches['away_score'])) |
                    ((h2h_matches['away_team'] == h2h_team2) & (h2h_matches['away_score'] > h2h_matches['home_score']))
                ])
                draws = len(h2h_matches) - wins_1 - wins_2
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Victorias {h2h_team1}", wins_1, f"{wins_1/len(h2h_matches)*100:.1f}%")
                with col2:
                    st.metric("Empates", draws, f"{draws/len(h2h_matches)*100:.1f}%")
                with col3:
                    st.metric(f"Victorias {h2h_team2}", wins_2, f"{wins_2/len(h2h_matches)*100:.1f}%")
                
                fig = go.Figure(data=[go.Pie(
                    labels=[f'{h2h_team1}', 'Empates', f'{h2h_team2}'],
                    values=[wins_1, draws, wins_2],
                    marker_colors=['#00D9FF', '#FFB800', '#FF4B4B']
                )])
                fig.update_layout(title="Distribución de Resultados")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Historial Completo de Partidos")
                
                h2h_display = []
                for _, row in h2h_matches.iterrows():
                    if row['home_team'] == h2h_team1:
                        result = '✅' if row['home_score'] > row['away_score'] else ('❌' if row['home_score'] < row['away_score'] else '➖')
                    else:
                        result = '✅' if row['away_score'] > row['home_score'] else ('❌' if row['away_score'] < row['home_score'] else '➖')
                    
                    h2h_display.append({
                        'Fecha': row['date'].strftime('%d/%m/%Y'),
                        'Local': row['home_team'],
                        'Resultado': f"{row['home_score']:.0f} - {row['away_score']:.0f}",
                        'Visitante': row['away_team'],
                        'xG': f"{row['xg_home']:.2f} - {row['xg_away']:.2f}",
                        f'Resultado ({h2h_team1})': result
                    })
                
                st.dataframe(pd.DataFrame(h2h_display), use_container_width=True, hide_index=True)
            else:
                st.warning("No se encontraron enfrentamientos directos entre estos equipos")
    
    # TAB 8: CLÁSICOS
    with tabs[7]:
        st.header("🔥 Clásicos del Fútbol Argentino")
        
        st.markdown("Análisis de los clásicos más importantes del fútbol argentino con predicciones y estadísticas históricas.")
        
        for clasico_name, (team1, team2) in CLASICOS.items():
            with st.expander(f"⚔️ {clasico_name}: {team1} vs {team2}", expanded=False):
                if team1 not in teams or team2 not in teams:
                    st.warning(f"Uno o ambos equipos no están en la base de datos")
                    continue
                
                st.subheader("🔮 Predicción del Próximo Encuentro")
                
                try:
                    preds = predictor.predict_match(team1, team2, method='all')
                    
                    col1, col2, col3 = st.columns(3)
                    
                    methods_available = []
                    if 'gradient_boosting' in preds:
                        methods_available.append(('GB', preds['gradient_boosting']))
                    if 'elo' in preds:
                        methods_available.append(('ELO', preds['elo']))
                    if 'ensemble' in preds:
                        methods_available.append(('Ensemble', preds['ensemble']))
                    
                    for idx, (method_name, pred) in enumerate(methods_available):
                        with [col1, col2, col3][idx]:
                            st.markdown(f"**{method_name}**")
                            st.metric(f"{team1}", f"{pred['prob_home']*100:.1f}%")
                            st.metric("Empate", f"{pred['prob_draw']*100:.1f}%")
                            st.metric(f"{team2}", f"{pred['prob_away']*100:.1f}%")
                
                except Exception as e:
                    st.error(f"Error en predicción: {e}")
                
                clasico_matches = matches_df[
                    ((matches_df['home_team'] == team1) & (matches_df['away_team'] == team2)) |
                    ((matches_df['home_team'] == team2) & (matches_df['away_team'] == team1))
                ]
                
                if len(clasico_matches) > 0:
                    st.subheader(f"📊 Historial ({len(clasico_matches)} partidos)")
                    
                    wins_1 = len(clasico_matches[
                        ((clasico_matches['home_team'] == team1) & (clasico_matches['home_score'] > clasico_matches['away_score'])) |
                        ((clasico_matches['away_team'] == team1) & (clasico_matches['away_score'] > clasico_matches['home_score']))
                    ])
                    wins_2 = len(clasico_matches[
                        ((clasico_matches['home_team'] == team2) & (clasico_matches['home_score'] > clasico_matches['away_score'])) |
                        ((clasico_matches['away_team'] == team2) & (clasico_matches['away_score'] > clasico_matches['home_score']))
                    ])
                    draws = len(clasico_matches) - wins_1 - wins_2
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(team1, wins_1)
                    with col2:
                        st.metric("Empates", draws)
                    with col3:
                        st.metric(team2, wins_2)
                    
                    st.markdown("**Últimos 5 Partidos**")
                    recent = clasico_matches.sort_values('date', ascending=False).head(5)
                    
                    for _, row in recent.iterrows():
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            st.markdown(f"**{row['home_team']}**")
                        with col2:
                            st.markdown(f"**{row['home_score']:.0f} - {row['away_score']:.0f}**")
                            st.caption(row['date'].strftime('%d/%m/%Y'))
                        with col3:
                            st.markdown(f"**{row['away_team']}**")
                else:
                    st.info("No hay enfrentamientos en el historial")
    
    # TAB 9: ANÁLISIS & HISTÓRICO
    with tabs[8]:
        st.header("Análisis y Explorador de Datos")
        
        sub_tabs = st.tabs(["📊 Análisis de Equipo", "🔍 Explorador Histórico"])
        
        with sub_tabs[0]:
            selected_team = st.selectbox("Seleccionar Equipo", teams, key='analysis_team')
            
            team_matches = matches_df[
                (matches_df['home_team'] == selected_team) | 
                (matches_df['away_team'] == selected_team)
            ].copy()
            
            team_matches['result'] = team_matches.apply(
                lambda x: 'Victoria' if (
                    (x['home_team'] == selected_team and x['home_score'] > x['away_score']) or
                    (x['away_team'] == selected_team and x['away_score'] > x['home_score'])
                ) else ('Empate' if x['home_score'] == x['away_score'] else 'Derrota'),
                axis=1
            )
            
            total_matches = len(team_matches)
            wins = len(team_matches[team_matches['result'] == 'Victoria'])
            draws = len(team_matches[team_matches['result'] == 'Empate'])
            losses = len(team_matches[team_matches['result'] == 'Derrota'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Partidos", total_matches)
            with col2:
                st.metric("Victorias", wins, f"{wins/total_matches*100:.1f}%")
            with col3:
                st.metric("Empates", draws, f"{draws/total_matches*100:.1f}%")
            with col4:
                st.metric("Derrotas", losses, f"{losses/total_matches*100:.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Distribución de Resultados")
                result_counts = team_matches['result'].value_counts()
                fig = px.pie(
                    values=result_counts.values,
                    names=result_counts.index,
                    color=result_counts.index,
                    color_discrete_map={'Victoria': '#00D9FF', 'Empate': '#FFB800', 'Derrota': '#FF4B4B'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⚽ Goles vs xG")
                
                team_matches['goals'] = team_matches.apply(
                    lambda x: x['home_score'] if x['home_team'] == selected_team else x['away_score'],
                    axis=1
                )
                team_matches['xg'] = team_matches.apply(
                    lambda x: x['xg_home'] if x['home_team'] == selected_team else x['xg_away'],
                    axis=1
                )
                
                avg_goals = team_matches['goals'].mean()
                avg_xg = team_matches['xg'].mean()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Goles Reales', x=['Promedio'], y=[avg_goals], marker_color='#00D9FF'))
                fig.add_trace(go.Bar(name='xG Promedio', x=['Promedio'], y=[avg_xg], marker_color='#FFB800'))
                fig.update_layout(title="Comparación Goles vs xG", yaxis_title="Promedio por partido")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📈 Evolución de xG en el tiempo")
            
            team_matches_sorted = team_matches.sort_values('date').tail(30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=team_matches_sorted['date'],
                y=team_matches_sorted['xg'],
                mode='lines+markers',
                name='xG a favor',
                line=dict(color='#00D9FF', width=2)
            ))
            
            team_matches_sorted['xg_against'] = team_matches_sorted.apply(
                lambda x: x['xg_away'] if x['home_team'] == selected_team else x['xg_home'],
                axis=1
            )
            
            fig.add_trace(go.Scatter(
                x=team_matches_sorted['date'],
                y=team_matches_sorted['xg_against'],
                mode='lines+markers',
                name='xG en contra',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            fig.update_layout(
                title=f"Últimos 30 partidos de {selected_team}",
                xaxis_title="Fecha",
                yaxis_title="xG",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🗓️ Últimos 10 Partidos")
            
            recent = team_matches.sort_values('date', ascending=False).head(10)
            display_df = recent[['date', 'home_team', 'home_score', 'away_score', 'away_team', 'result']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%d/%m/%Y')
            display_df.columns = ['Fecha', 'Local', 'Goles L', 'Goles V', 'Visitante', 'Resultado']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with sub_tabs[1]:
            st.subheader("Historial de Partidos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_team = st.selectbox("Filtrar por equipo (opcional)", ['Todos'] + teams, key='filter_team')
            
            with col2:
                filter_league = st.multiselect("Liga", matches_df['league'].unique(), default=matches_df['league'].unique())
            
            with col3:
                min_date = matches_df['date'].min()
                max_date = matches_df['date'].max()
                date_range = st.date_input(
                    "Rango de fechas",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            filtered_df = matches_df.copy()
            
            if filter_team != 'Todos':
                filtered_df = filtered_df[
                    (filtered_df['home_team'] == filter_team) | 
                    (filtered_df['away_team'] == filter_team)
                ]
            
            filtered_df = filtered_df[filtered_df['league'].isin(filter_league)]
            
            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['date'] >= pd.to_datetime(date_range[0])) &
                    (filtered_df['date'] <= pd.to_datetime(date_range[1]))
                ]
            
            st.metric("Partidos encontrados", len(filtered_df))
            
            display_df = filtered_df[['date', 'league', 'home_team', 'home_score', 'xg_home', 
                                       'away_score', 'xg_away', 'away_team']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%d/%m/%Y')
            display_df.columns = ['Fecha', 'Liga', 'Local', 'Goles L', 'xG L', 'Goles V', 'xG V', 'Visitante']
            
            st.dataframe(
                display_df.sort_values('Fecha', ascending=False),
                use_container_width=True,
                hide_index=True,
                height=600
            )
            
            st.subheader("📊 Estadísticas del Periodo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_goals = (filtered_df['home_score'] + filtered_df['away_score']).mean()
                st.metric("Goles por partido", f"{avg_goals:.2f}")
            
            with col2:
                avg_xg = (filtered_df['xg_home'] + filtered_df['xg_away']).mean()
                st.metric("xG promedio", f"{avg_xg:.2f}")
            
            with col3:
                home_wins = len(filtered_df[filtered_df['home_score'] > filtered_df['away_score']])
                st.metric("Victorias locales", f"{home_wins/len(filtered_df)*100:.1f}%")
            
            with col4:
                draws = len(filtered_df[filtered_df['home_score'] == filtered_df['away_score']])
                st.metric("Empates", f"{draws/len(filtered_df)*100:.1f}%")

else:
    st.info("👈 Cargá los archivos de datos en la barra lateral y entrenás los modelos para comenzar")
    
    st.markdown("""
    ### 📋 Instrucciones
    
    1. **Cargá tus archivos Excel:**
       - Podés usar los datos precargados del repositorio (recomendado)
       - O subir tus propios archivos `matches.xlsx` y `stats.xlsx`
    
    2. **Entrenás los modelos:**
       - Hacé click en "🚀 Entrenar Modelos"
       - El sistema va a entrenar Gradient Boosting y ELO
    
    3. **Usá las funcionalidades:**
       - **Predicción Individual**: Un partido a la vez con análisis detallado
       - **Predicción Múltiple**: Varios partidos simultáneamente
       - **Simulador de Torneos**: Liga o eliminación directa
       - **Rankings & ELO**: Rankings actuales, evolución temporal y comparación
       - **Head to Head**: Historial detallado entre dos equipos
       - **Clásicos**: Análisis de los clásicos del fútbol argentino
       - **Análisis & Histórico**: Estadísticas por equipo y explorador de datos
    
    ### 🎯 Métodos de Predicción
    
    - **Gradient Boosting + Poisson**: Predice xG usando machine learning
    - **Sistema ELO**: Ratings dinámicos actualizados
    - **Ensemble**: Combina ambos métodos (65% GB + 35% ELO)
    """)

st.markdown("---")
st.markdown("🔬 Sistema de Predicción de Fútbol Argentino | Basado en xG, ML y ELO")
