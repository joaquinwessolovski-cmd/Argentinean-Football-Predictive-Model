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
            try:
                preds = predictor.predict_match(home, away, method='ensemble')
                if 'ensemble' not in preds:
                    preds = predictor.predict_match(home, away, method='elo')
                
                pred = preds.get('ensemble', preds.get('elo'))
                if pred is None:
                    continue
                
                # CORRECCIÓN: Usar las probabilidades correctamente
                prob_home = pred['prob_home']
                prob_draw = pred['prob_draw']
                prob_away = pred['prob_away']
                
                # Simular resultado basado en probabilidades
                rand = np.random.random()
                
                if rand < prob_home:
                    outcome = 'home'
                elif rand < (prob_home + prob_draw):
                    outcome = 'draw'
                else:
                    outcome = 'away'
                
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
            except:
                continue
        
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
        try:
            preds = predictor.predict_match(home, away, method='ensemble')
            if 'ensemble' not in preds:
                preds = predictor.predict_match(home, away, method='elo')
            
            pred = preds.get('ensemble', preds.get('elo'))
            if pred is None:
                winners.append(home)
                continue
            
            # CORRECCIÓN: En cancha neutral, ajustar probabilidades
            if neutral_venue:
                # Eliminar ventaja de local redistribuyendo probabilidades
                total_no_draw = pred['prob_home'] + pred['prob_away']
                prob_home = pred['prob_home'] / total_no_draw * (1 - pred['prob_draw'] * 0.5)
                prob_away = pred['prob_away'] / total_no_draw * (1 - pred['prob_draw'] * 0.5)
                prob_draw = pred['prob_draw']
                
                # Normalizar
                total = prob_home + prob_draw + prob_away
                prob_home /= total
                prob_draw /= total
                prob_away /= total
            else:
                prob_home = pred['prob_home']
                prob_draw = pred['prob_draw']
                prob_away = pred['prob_away']
            
            # En eliminación directa, si hay empate va a penales (50-50)
            rand = np.random.random()
            
            if rand < prob_home:
                winner = home
            elif rand < (prob_home + prob_draw):
                # Empate → penales (50-50)
                winner = home if np.random.random() < 0.5 else away
            else:
                winner = away
            
            winners.append(winner)
            results.append({
                'home': home,
                'away': away,
                'winner': winner,
                'prob_home': prob_home,
                'prob_draw': prob_draw,
                'prob_away': prob_away
            })
        except Exception as e:
            winners.append(home)
            results.append({
                'home': home,
                'away': away,
                'winner': home,
                'prob_home': 0.5,
                'prob_draw': 0.0,
                'prob_away': 0.5
            })
    
    return winners, results

def simulate_group_stage_knockout(predictor, zone_a, zone_b, n_simulations=1000, neutral_playoffs=False):
    """Simula torneo con fase de grupos + eliminación directa"""
    champion_count = {}
    
    for _ in range(n_simulations):
        try:
            # Fase de grupos - Zona A
            zone_a_matches = [(t1, t2) for t1 in zone_a for t2 in zone_a if t1 != t2]
            zone_a_points, _ = simulate_league_tournament(predictor, zone_a_matches, n_simulations=1)
            
            # Ordenar por puntos promedio
            zone_a_standings = []
            for team in zone_a:
                if team in zone_a_points and len(zone_a_points[team]) > 0:
                    zone_a_standings.append((team, np.mean(zone_a_points[team])))
            zone_a_standings.sort(key=lambda x: x[1], reverse=True)
            
            # Clasificados de zona A (mitad superior)
            n_qualified = max(1, len(zone_a) // 2)
            zone_a_qualified = [team for team, _ in zone_a_standings[:n_qualified]]
            
            # Fase de grupos - Zona B
            zone_b_matches = [(t1, t2) for t1 in zone_b for t2 in zone_b if t1 != t2]
            zone_b_points, _ = simulate_league_tournament(predictor, zone_b_matches, n_simulations=1)
            
            zone_b_standings = []
            for team in zone_b:
                if team in zone_b_points and len(zone_b_points[team]) > 0:
                    zone_b_standings.append((team, np.mean(zone_b_points[team])))
            zone_b_standings.sort(key=lambda x: x[1], reverse=True)
            
            zone_b_qualified = [team for team, _ in zone_b_standings[:n_qualified]]
            
            # Playoffs (cruzados)
            if len(zone_a_qualified) >= 2 and len(zone_b_qualified) >= 2:
                # Semifinales cruzadas
                semi_matches = [
                    (zone_a_qualified[0], zone_b_qualified[1]),
                    (zone_b_qualified[0], zone_a_qualified[1])
                ]
                semi_winners, _ = simulate_knockout_round(predictor, semi_matches, neutral_playoffs)
                
                # Final
                if len(semi_winners) == 2:
                    final_match = [(semi_winners[0], semi_winners[1])]
                    final_winner, _ = simulate_knockout_round(predictor, final_match, neutral_playoffs)
                    if len(final_winner) > 0:
                        champion = final_winner[0]
                        champion_count[champion] = champion_count.get(champion, 0) + 1
        except:
            continue
    
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
        
        # Filtros por división
        teams_home = teams
        teams_away = teams
        
        if divisions:
            col1, col2 = st.columns(2)
            with col1:
                filter_div_home = st.selectbox("Filtrar división (Local)", ['Todas'] + divisions, key='filter_home_div')
                if filter_div_home != 'Todas':
                    teams_home = [t for t in teams if team_division.get(t) == filter_div_home]
            with col2:
                filter_div_away = st.selectbox("Filtrar división (Visitante)", ['Todas'] + divisions, key='filter_away_div')
                if filter_div_away != 'Todas':
                    teams_away = [t for t in teams if team_division.get(t) == filter_div_away]
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Equipo Local", teams_home, key='home')
        
        with col2:
            away_team = st.selectbox("Equipo Visitante", teams_away, key='away')
        
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
    
    # ============================================================================
    # TAB 3: SIMULADOR DE TORNEOS (VERSIÓN MEJORADA)
    # ============================================================================


    with tabs[2]:
        st.header("Simulador de Torneos")
    
        tournament_type = st.radio("Tipo de torneo", [
        "Liga (Todos contra todos)", 
        "Eliminación Directa", 
        "Grupos + Playoffs"
        ])
    
        if tournament_type == "Liga (Todos contra todos)":
            st.subheader("Simulación de Liga")
        
            selected_teams = st.multiselect("Equipos del torneo", teams, default=teams[:6])
        
            if len(selected_teams) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    n_rounds = st.radio("Número de vueltas", [1, 2], index=1)
                with col2:
                    n_sims = st.slider("Simulaciones", 100, 10000, 1000, step=100)
            
                # Generar fixture
                matches = []
                for home in selected_teams:
                    for away in selected_teams:
                        if home != away:
                            matches.append((home, away))
            
                # Si es doble vuelta, duplicar
                if n_rounds == 2:
                    matches = matches * 2
            
                st.info(f"📊 Total de partidos: {len(matches)}")
            
                if st.button("🎲 Simular Torneo", type="primary"):
                    with st.spinner(f"Simulando {n_sims} torneos..."):
                        team_points, team_positions = simulate_league_tournament(predictor, matches, n_sims)
                    
                    results = []
                    for team in selected_teams:
                        if team in team_points:
                            results.append({
                                'Equipo': team,
                                'Puntos Promedio': f"{np.mean(team_points[team]):.1f}",
                                'Posición Promedio': f"{np.mean(team_positions[team]):.1f}",
                                '% Campeón': f"{(team_positions[team].count(1) / n_sims * 100):.1f}%"
                            })
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
    
        elif tournament_type == "Eliminación Directa":
            st.subheader("Simulación de Eliminación Directa")
        
            num_teams_ko = st.selectbox("Número de equipos", [4, 8, 16])
            neutral_venue = st.checkbox("Partidos en cancha neutral", value=False)
        
            selected_teams_ko = st.multiselect("Seleccioná equipos", teams, max_selections=num_teams_ko)
        
            if len(selected_teams_ko) == num_teams_ko:
                if st.button("🏆 Simular Torneo Completo", type="primary"):
                    with st.spinner("Simulando todas las rondas..."):
                        current_round_teams = selected_teams_ko.copy()
                        round_names = []
                    
                    # Determinar nombres de rondas
                    if num_teams_ko == 16:
                        round_names = ["Octavos", "Cuartos", "Semifinales", "Final"]
                    elif num_teams_ko == 8:
                        round_names = ["Cuartos", "Semifinales", "Final"]
                    else:
                        round_names = ["Semifinales", "Final"]
                    
                    for round_idx, round_name in enumerate(round_names):
                        st.subheader(f"🏆 {round_name} de Final")
                        
                        # Crear llaves
                        round_matches = []
                        for i in range(0, len(current_round_teams), 2):
                            if i + 1 < len(current_round_teams):
                                round_matches.append((current_round_teams[i], current_round_teams[i+1]))
                        
                        # Simular ronda
                        winners, results = simulate_knockout_round(predictor, round_matches, neutral_venue)
                        
                        # Mostrar resultados
                        for match_result in results:
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown(f"**{match_result['home']}**")
                                st.caption(f"{match_result['prob_home']*100:.1f}%")
                            with col2:
                                if match_result['winner'] == match_result['away']:
                                    st.success("✅→")
                                else:
                                    st.success("←✅")
                            with col3:
                                st.markdown(f"**{match_result['away']}**")
                                st.caption(f"{match_result['prob_away']*100:.1f}%")
                        
                        current_round_teams = winners
                        st.markdown("---")
                    
                    if len(current_round_teams) == 1:
                        st.balloons()
                        st.success(f"🏆 **CAMPEÓN: {current_round_teams[0]}** 🏆")
    
        else:  # Grupos + Playoffs
            st.subheader("Simulación: Fase de Grupos + Playoffs")
        
            st.markdown("Los equipos se dividen en 2 zonas. La mitad superior de cada zona avanza a playoffs.")
        
            num_teams_total = st.selectbox("Equipos por zona", [4, 6, 8])
            neutral_playoffs = st.checkbox("Playoffs en cancha neutral", value=True)
        
            selected_teams_groups = st.multiselect(
                "Seleccioná equipos (se dividirán en 2 zonas)", 
                teams, 
                max_selections=num_teams_total*2
            )
        
            if len(selected_teams_groups) == num_teams_total * 2:
                n_sims = st.slider("Simulaciones", 100, 5000, 1000, step=100)
            
                if st.button("🎲 Simular Torneo", type="primary"):
                    with st.spinner(f"Simulando {n_sims} torneos..."):
                        # Dividir en zonas
                        zone_a = selected_teams_groups[:num_teams_total]
                        zone_b = selected_teams_groups[num_teams_total:]
                    
                        st.markdown(f"**Zona A:** {', '.join(zone_a)}")
                        st.markdown(f"**Zona B:** {', '.join(zone_b)}")
                    
                        champion_count = simulate_group_stage_knockout(
                        predictor, zone_a, zone_b, n_sims, neutral_playoffs
                    )   
                    
                    # Mostrar resultados
                    st.subheader("Probabilidades de ser Campeón")
                    results = []
                    for team, count in sorted(champion_count.items(), key=lambda x: x[1], reverse=True):
                            results.append({
                            'Equipo': team,
                            'Campeonatos': count,
                            'Probabilidad': f"{count/n_sims*100:.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
    

    # ============================================================================
    # TAB 4: APUESTAS & VALUE BETS (NUEVO)
    # ============================================================================


    with tabs[3]:
        st.header("💰 Sistema de Apuestas & Value Betting")
        sub_tabs = st.tabs(["🎯 Value Bets", "📊 Kelly Criterion", "📈 Simulador de Bankroll"])
    
        with sub_tabs[0]:
            st.subheader("Identificar Value Bets")
        
        # Input de partidos y cuotas
        num_bets = st.number_input("Número de partidos a analizar", 1, 10, 3)
        
        matches_odds = []
        
        for i in range(num_bets):
            st.markdown(f"**Partido {i+1}**")
            col1, col2 = st.columns(2)
            
            with col1:
                home = st.selectbox(f"Local", teams, key=f"vb_home_{i}")
            with col2:
                away = st.selectbox(f"Visitante", teams, key=f"vb_away_{i}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                odd_home = st.number_input(f"Cuota Local", 1.01, 50.0, 2.0, 0.01, key=f"odd_h_{i}")
            with col2:
                odd_draw = st.number_input(f"Cuota Empate", 1.01, 50.0, 3.2, 0.01, key=f"odd_d_{i}")
            with col3:
                odd_away = st.number_input(f"Cuota Visitante", 1.01, 50.0, 3.5, 0.01, key=f"odd_a_{i}")
            
            if home != away:
                matches_odds.append({
                    'match': (home, away),
                    'odds': {'home': odd_home, 'draw': odd_draw, 'away': odd_away}
                })
            
            st.markdown("---")
        
        threshold = st.slider("Threshold de Edge (%)", 1.0, 10.0, 3.0, 0.5) / 100
        
        if st.button("🔍 Buscar Value Bets", type="primary"):
            if len(matches_odds) > 0:
                bookmaker_odds = {m['match']: m['odds'] for m in matches_odds}
                value_bets = calculate_value_bets(
                    predictor, 
                    [m['match'] for m in matches_odds],
                    bookmaker_odds,
                    threshold
                )
                
                if value_bets:
                    st.success(f"✅ Encontradas {len(value_bets)} oportunidades")
                    st.dataframe(pd.DataFrame(value_bets), use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No se encontraron value bets con el threshold actual")
    
        with sub_tabs[1]:
            st.subheader("Calculadora Kelly Criterion")
        
        
            col1, col2 = st.columns(2)
        
            with col1:
                prob_model = st.slider("Probabilidad del Modelo", 0.0, 1.0, 0.45, 0.01)
                odd_market = st.number_input("Cuota del Mercado", 1.01, 50.0, 2.20, 0.01)
        
            with col2:
                bankroll = st.number_input("Bankroll Total ($)", 100.0, 1000000.0, 1000.0, 100.0)
                kelly_fraction = st.slider("Fracción de Kelly", 0.1, 1.0, 0.25, 0.05)
        
            prob_market = 1 / odd_market
            edge = prob_model - prob_market
        
            if edge > 0:
                kelly_pct = kelly_criterion(prob_model, odd_market, kelly_fraction)
                stake = bankroll * kelly_pct
                potential_win = stake * (odd_market - 1)
                ev = (prob_model * odd_market - 1) * 100
            
                col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                    st.metric("Edge", f"{edge*100:.2f}%", delta="Positivo" if edge > 0 else "Negativo")
                with col2:
                    st.metric("Kelly %", f"{kelly_pct*100:.2f}%")
                with col3:
                    st.metric("Stake Sugerido", f"${stake:.2f}")
                with col4:
                    st.metric("Ganancia Potencial", f"${potential_win:.2f}")
            
                st.info(f"💡 Expected Value: {ev:+.2f}%")
            
                if kelly_pct > 0.05:
                    st.warning("⚠️ Apuesta grande. Considerá reducir la fracción de Kelly.")
                else:
                        st.error("❌ No hay edge positivo. No se recomienda apostar.")
    
        with sub_tabs[2]:
            st.subheader("Simulador de Bankroll")
        
            st.markdown("Simulá la evolución de tu bankroll apostando con Kelly Criterion.")
        
            # Configuración
            initial_bankroll = st.number_input("Bankroll Inicial ($)", 100.0, 100000.0, 1000.0, 100.0)
            num_bets_sim = st.slider("Número de apuestas a simular", 10, 100, 50)
            kelly_frac_sim = st.slider("Fracción Kelly", 0.1, 0.5, 0.25, 0.05)
        
            if st.button("📈 Simular", type="primary"):
                # Simulación simple (datos aleatorios para demo)
                bankroll_history = [initial_bankroll]
                current_bankroll = initial_bankroll
            
                for _ in range(num_bets_sim):
                # Generar apuesta aleatoria con edge positivo
                    prob = np.random.uniform(0.45, 0.65)
                    odd = np.random.uniform(1.8, 3.5)
                
                if prob * odd > 1:  # Tiene edge
                    kelly_stake = kelly_criterion(prob, odd, kelly_frac_sim)
                    stake_amount = current_bankroll * kelly_stake
                    
                    # Simular resultado
                    if np.random.random() < prob:
                        current_bankroll += stake_amount * (odd - 1)
                    else:
                        current_bankroll -= stake_amount
                
                bankroll_history.append(max(0, current_bankroll))
            
                # Graficar
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                x=list(range(len(bankroll_history))),
                y=bankroll_history,
                mode='lines',
                name='Bankroll',
                fill='tozeroy'
                ))
                fig.add_hline(y=initial_bankroll, line_dash="dash", line_color="gray", annotation_text="Inicial")
                fig.update_layout(
                title="Evolución del Bankroll",
                xaxis_title="Número de Apuestas",
                yaxis_title="Bankroll ($)",
                height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
                final_bankroll = bankroll_history[-1]
                roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
            
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bankroll Final", f"${final_bankroll:.2f}")
                with col2:
                    st.metric("ROI", f"{roi:+.2f}%")
                with col3:
                    max_br = max(bankroll_history)
                    st.metric("Pico Máximo", f"${max_br:.2f}")
    
    
    # TAB 4: RANKINGS & ELO
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
    
    # TAB 5: HEAD TO HEAD
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
    
    # TAB 6: CLÁSICOS
    with tabs[7]:
        st.header("🔥 Clásicos del Fútbol Argentino")
        
        st.markdown("Análisis de los clásicos más importantes del fútbol argentino con predicciones y estadísticas históricas.")
        
        for clasico_name, (team1, team2) in CLASICOS.items():
            with st.expander(f"⚔️ {clasico_name}: {team1} vs {team2}", expanded=False):
                if team1 not in teams or team2 not in  teams:
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
    
    # TAB 7: ANÁLISIS & HISTÓRICO
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
                
                
    # ============================================================================
    # TAB 5: CONFIGURACIÓN Y EVALUACIÓN DE MODELOS
    # Inserta este código en streamlit_app.py como tabs[4]
    # ============================================================================

    with tabs[4]:
        st.header("⚙️ Configuración y Evaluación de Modelos")
    
        config_tabs = st.tabs(["📊 Métricas Actuales", "🔧 Ajustar Parámetros", "🔄 Re-entrenar"])
    
        # ==================== SUB-TAB 1: MÉTRICAS ACTUALES ====================
        with config_tabs[0]:
            st.subheader("Métricas de Evaluación de Modelos")
        
            if st.session_state['model_metrics']:
                # Resumen general
                st.markdown("### Resumen General")

                metrics_summary = []
                for model_name, metrics in st.session_state['model_metrics'].items():
                    metrics_summary.append({
                    'Modelo': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                    'Log Loss': f"{metrics['log_loss']:.4f}"
                })
            
            summary_df = pd.DataFrame(metrics_summary)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Detalles por modelo
            st.markdown("---")
            st.markdown("### Detalles por Modelo")
            
            for model_name, metrics in st.session_state['model_metrics'].items():
                with st.expander(f"📈 {model_name.replace('_', ' ').title()}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy (1X2)", f"{metrics['accuracy']*100:.2f}%")
                        st.caption("Porcentaje de aciertos en el resultado exacto")
                    
                    with col2:
                        st.metric("Log Loss", f"{metrics['log_loss']:.4f}")
                        st.caption("Menor es mejor. Mide calidad probabilística")
                    
                    with col3:
                        baseline_acc = 33.33
                        improvement = (metrics['accuracy'] * 100) - baseline_acc
                        st.metric("Mejora vs Azar", f"+{improvement:.2f}pp")
                        st.caption("Comparado con predicción aleatoria (33%)")
                    
                    # Interpretación
                    if metrics['accuracy'] > 0.50:
                        st.success("✅ Excelente performance - Modelo muy preciso")
                    elif metrics['accuracy'] > 0.45:
                        st.info("✓ Buena performance - Dentro del rango esperado")
                    elif metrics['accuracy'] > 0.40:
                        st.warning("⚠️ Performance moderada - Considerar ajustes")
                    else:
                        st.error("❌ Performance baja - Re-entrenar con más datos o ajustar parámetros")
            
            # Comparación visual
            if len(st.session_state['model_metrics']) > 1:
                st.markdown("---")
                st.markdown("### Comparación Entre Modelos")
                
                comparison_data = []
                for model, metrics in st.session_state['model_metrics'].items():
                    comparison_data.append({
                        'Modelo': model.replace('_', ' ').title(),
                        'Accuracy': metrics['accuracy'] * 100,
                        'Log Loss': metrics['log_loss']
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=comp_df['Modelo'],
                        y=comp_df['Accuracy'],
                        marker_color='#00D9FF',
                        text=comp_df['Accuracy'].round(2),
                        textposition='auto'
                    ))
                    fig.add_hline(y=33.33, line_dash="dash", line_color="red", 
                                 annotation_text="Azar (33%)")
                    fig.update_layout(
                        title="Accuracy por Modelo", 
                        yaxis_title="Accuracy (%)",
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=comp_df['Modelo'],
                        y=comp_df['Log Loss'],
                        marker_color='#FFB800',
                        text=comp_df['Log Loss'].round(4),
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Log Loss por Modelo (menor es mejor)", 
                        yaxis_title="Log Loss",
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recomendación
                best_model = max(st.session_state['model_metrics'].items(), 
                               key=lambda x: x[1]['accuracy'])
                st.info(f"💡 **Mejor modelo:** {best_model[0].replace('_', ' ').title()} "
                       f"con {best_model[1]['accuracy']*100:.2f}% accuracy")
                
            # Tabla de diferencias
                st.markdown("#### Diferencias entre Modelos")
                if len(comparison_data) >= 2:
                    model1 = comparison_data[0]
                    model2 = comparison_data[1]
                    diff_acc = model1['Accuracy'] - model2['Accuracy']
                    diff_ll = model1['Log Loss'] - model2['Log Loss']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            f"Diferencia Accuracy",
                            f"{abs(diff_acc):.2f}pp",
                            delta=f"{model1['Modelo']} mejor" if diff_acc > 0 else f"{model2['Modelo']} mejor"
                        )
                    with col2:
                        st.metric(
                            f"Diferencia Log Loss",
                            f"{abs(diff_ll):.4f}",
                            delta=f"{model1['Modelo']} mejor" if diff_ll < 0 else f"{model2['Modelo']} mejor",
                            delta_color="inverse"
                        )
                else:
                    st.warning("⚠️ No hay métricas disponibles. Entrená los modelos primero.")
                    st.info("💡 Las métricas se calculan automáticamente al entrenar los modelos usando una validación temporal (train/test split).")
    
        # ==================== SUB-TAB 2: AJUSTAR PARÁMETROS ====================
        with config_tabs[1]:
            st.subheader("Ajustar Parámetros de los Modelos")
        
            model_to_config = st.selectbox("Seleccionar Modelo", ["Gradient Boosting", "Sistema ELO"])
        
            if model_to_config == "Gradient Boosting":
                st.markdown("### Parámetros de Gradient Boosting Regressor")
            
                st.info("💡 Estos parámetros se aplicarán al re-entrenar el modelo en la siguiente pestaña")
            
                with st.expander("ℹ️ Guía de Parámetros", expanded=False):
                    st.markdown("""
                **n_estimators**: Número de árboles de decisión
                - Más árboles = más precisión pero más lento
                - Rango típico: 50-300
                - Default: 100
                
                **learning_rate**: Tasa de aprendizaje
                - Controla cuánto se ajusta en cada iteración
                - Menor = más preciso pero necesita más árboles
                - Rango típico: 0.01-0.3
                - Default: 0.1
                
                **max_depth**: Profundidad máxima de cada árbol
                - Mayor = más complejo, puede sobreajustar
                - Rango típico: 2-8
                - Default: 4
                
                **min_samples_split**: Mínimo de muestras para dividir un nodo
                - Mayor = más regularización, menos sobreajuste
                - Rango típico: 2-20
                - Default: 2
                
                **min_samples_leaf**: Mínimo de muestras en una hoja
                - Mayor = más regularización
                - Rango típico: 1-10
                - Default: 1
                
                **subsample**: Fracción de muestras para cada árbol
                - < 1.0 = stochastic gradient boosting
                - Ayuda a prevenir sobreajuste
                - Rango típico: 0.5-1.0
                - Default: 1.0
                """)
            
                col1, col2 = st.columns(2)
            
                with col1:
                    n_estimators = st.slider(
                    "n_estimators",
                    50, 300, 100, 10,
                    help="Número de árboles. Más árboles = más precisión pero más lento"
                )
                
                    learning_rate = st.slider(
                    "learning_rate",
                    0.01, 0.3, 0.1, 0.01,
                    help="Tasa de aprendizaje. Menor = más preciso pero necesita más árboles"
                )
                
                    max_depth = st.slider(
                    "max_depth",
                    2, 8, 4, 1,
                    help="Profundidad máxima de cada árbol. Mayor = más complejo"
                )
            
                with col2:
                    min_samples_split = st.slider(
                    "min_samples_split",
                    2, 20, 2, 1,
                    help="Mínimo de muestras para dividir un nodo. Mayor = más regularización"
                    )
                
                    min_samples_leaf = st.slider(
                    "min_samples_leaf",
                    1, 10, 1, 1,
                    help="Mínimo de muestras en hoja. Mayor = más regularización"
                )
                
                    subsample = st.slider(
                    "subsample",
                    0.5, 1.0, 1.0, 0.05,
                    help="Fracción de muestras para entrenar cada árbol"
                )
            
                # Preview de configuración
                st.markdown("---")
                st.markdown("### Preview de Configuración")
            
                config_preview = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample
                }
            
                st.json(config_preview)
            
                # Guardar en session state
                if st.button("💾 Guardar Configuración", type="primary", use_container_width=True):
                    st.session_state['gb_params'] = config_preview
                    st.success("✅ Configuración guardada. Andá a la pestaña 'Re-entrenar' para aplicar los cambios.")
                    st.balloons()
        
            else:  # Sistema ELO
                st.markdown("### Parámetros del Sistema ELO")
            
                st.info("💡 Estos parámetros afectan cómo se actualizan los ratings después de cada partido")
            
                with st.expander("ℹ️ Guía de Parámetros", expanded=False):
                    st.markdown("""
                **Factor K**: Controla la volatilidad de los ratings
                - K bajo (10-20): Cambios lentos y estables
                - K medio (20-30): Balanceado (recomendado)
                - K alto (30-40): Se adapta rápido a cambios de forma
                - K muy alto (40+): Muy volátil, puede sobrereaccionar
                
                **Ventaja de Local**: Puntos ELO extra para el equipo que juega de local
                - 0: Sin ventaja (cancha neutral)
                - 50-80: Ventaja moderada
                - 100-120: Estándar (recomendado para Argentina)
                - 150+: Ventaja fuerte (para ligas con mucha localía)
                
                **Rating Inicial**: Rating asignado a equipos nuevos o al inicio
                - 1500: Estándar (promedio)
                - Equipos fuertes empiezan más alto
                - Se ajusta rápidamente en los primeros partidos
                """)
            
                col1, col2 = st.columns(2)
            
                with col1:
                    k_factor = st.slider(
                    "Factor K",
                    10, 60, 30, 5,
                    help="Volatilidad de los ratings. Mayor = cambios más bruscos"
                )
                
                    st.markdown("**Interpretación del Factor K:**")
                if k_factor <= 20:
                    st.info("🐌 Muy estable - Cambios lentos")
                elif k_factor <= 30:
                    st.success("✓ Balanceado - Recomendado")
                elif k_factor <= 40:
                    st.warning("⚡ Dinámico - Se adapta rápido")
                else:
                    st.error("🔥 Muy volátil - Puede sobrereaccionar")
            
                with col2:
                    home_advantage = st.slider(
                    "Ventaja de Local (puntos ELO)",
                    0, 200, 100, 10,
                    help="Puntos extra de ELO para el equipo local"
                )
                
                st.markdown("**Ventaja de Local Típica:**")
                if home_advantage == 0:
                    st.info("⚽ Sin ventaja (cancha neutral)")
                elif home_advantage < 80:
                    st.info("🏠 Ventaja moderada")
                elif home_advantage <= 120:
                    st.success("✓ Estándar (recomendado)")
                else:
                    st.warning("🏟️ Ventaja fuerte")
            
                initial_rating = st.number_input(
                "Rating Inicial",
                1000, 2000, 1500, 50,
                help="Rating asignado a equipos nuevos"
                )
            
                # Preview de configuración
                st.markdown("---")
                st.markdown("### Preview de Configuración")
            
                config_preview = {
                'k_factor': k_factor,
                'home_advantage': home_advantage,
                'initial_rating': initial_rating
                }
            
                st.json(config_preview)
            
                # Guardar en session state
                if st.button("💾 Guardar  Configuración", type="primary", use_container_width=True):
                    st.session_state['elo_params'] = config_preview
                    st.success("✅ Configuración guardada. Andá a la pestaña 'Re-entrenar' para aplicar los cambios.")
                    st.balloons()
    
        # ==================== SUB-TAB 3: RE-ENTRENAR ====================
        with config_tabs[2]:
            st.subheader("Re-entrenar Modelos")
        
            st.markdown("""
            Re-entrená los modelos con:
            - 📊 Nuevos datos agregados
            - ⚙️ Nuevos parámetros configurados  
            - 📈 Diferentes proporciones de train/test
            """)
        
            col1, col2 = st.columns(2)
        
            with col1:
                train_split = st.slider(
                "Proporción Train/Test",
                0.7, 0.95, 0.85, 0.05,
                help="% de datos para entrenamiento"
            )
                st.caption(f"Train: {train_split*100:.0f}% | Test: {(1-train_split)*100:.0f}%")
        
            with col2:
                models_to_retrain = st.multiselect(
                "Modelos a re-entrenar",
                ["Gradient Boosting", "Sistema ELO"],
                default=["Gradient Boosting", "Sistema ELO"]
            )
        
        
        
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
