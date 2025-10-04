import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

class FootballPredictor:
    def __init__(self):
        self.gb_model_home = None
        self.gb_model_away = None
        self.elo_ratings = {}
        self.team_stats = {}
        self.matches_df = None
        self.stats_df = None
        
    def load_data(self, matches_path, stats_path):
        """Carga datos desde archivos Excel"""
        self.matches_df = pd.read_excel(matches_path)
        self.stats_df = pd.read_excel(stats_path)
        
        # Convertir fecha a datetime
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
        self.matches_df = self.matches_df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Cargados {len(self.matches_df)} partidos desde {self.matches_df['date'].min()} hasta {self.matches_df['date'].max()}")
        print(f"✓ Estadísticas de {len(self.stats_df)} equipos")
        
        return self.matches_df, self.stats_df
    
    def calculate_rolling_features(self, df, windows=[5, 10]):
        """Calcula features rolling por equipo"""
        features_df = df.copy()
        all_teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        # Diccionarios para almacenar historial
        team_history = {team: {'xg': [], 'xga': [], 'gf': [], 'ga': []} for team in all_teams}
        
        rolling_features = []
        valid_indices = []  # Guardar índices de filas válidas
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Solo calcular features si ambos equipos tienen suficiente historial
            min_window = min(windows)
            home_has_history = len(team_history[home_team]['xg']) >= min_window
            away_has_history = len(team_history[away_team]['xg']) >= min_window
            
            if home_has_history and away_has_history:
                # Features para equipo local
                home_features = {}
                for window in windows:
                    h_xg = team_history[home_team]['xg'][-window:]
                    h_xga = team_history[home_team]['xga'][-window:]
                    h_gf = team_history[home_team]['gf'][-window:]
                    
                    home_features[f'home_xg_avg_{window}'] = np.mean(h_xg)
                    home_features[f'home_xga_avg_{window}'] = np.mean(h_xga)
                    home_features[f'home_gf_avg_{window}'] = np.mean(h_gf)
                    home_features[f'home_form_{window}'] = np.mean(h_xg) - np.mean(h_xga)
                
                # Features para equipo visitante
                away_features = {}
                for window in windows:
                    a_xg = team_history[away_team]['xg'][-window:]
                    a_xga = team_history[away_team]['xga'][-window:]
                    a_gf = team_history[away_team]['gf'][-window:]
                    
                    away_features[f'away_xg_avg_{window}'] = np.mean(a_xg)
                    away_features[f'away_xga_avg_{window}'] = np.mean(a_xga)
                    away_features[f'away_gf_avg_{window}'] = np.mean(a_gf)
                    away_features[f'away_form_{window}'] = np.mean(a_xg) - np.mean(a_xga)
                
                rolling_features.append({**home_features, **away_features})
                valid_indices.append(idx)
            
            # Actualizar historial para todos los partidos
            team_history[home_team]['xg'].append(row['xg_home'])
            team_history[home_team]['xga'].append(row['xg_away'])
            team_history[home_team]['gf'].append(row['home_score'])
            team_history[home_team]['ga'].append(row['away_score'])
            
            team_history[away_team]['xg'].append(row['xg_away'])
            team_history[away_team]['xga'].append(row['xg_home'])
            team_history[away_team]['gf'].append(row['away_score'])
            team_history[away_team]['ga'].append(row['home_score'])
        
        # Crear DataFrame solo con filas válidas
        rolling_df = pd.DataFrame(rolling_features)
        features_df = features_df.loc[valid_indices].reset_index(drop=True)
        features_df = pd.concat([features_df, rolling_df.reset_index(drop=True)], axis=1)
        
        return features_df
    
    def train_gradient_boosting(self, train_split=0.8):
        """Entrena modelos Gradient Boosting para predecir xG"""
        print("\n🔄 Entrenando Gradient Boosting...")
        
        # Calcular features
        df_features = self.calculate_rolling_features(self.matches_df)
        
        print(f"✓ Features calculadas para {len(df_features)} partidos (de {len(self.matches_df)} totales)")
        
        # Verificar que tengamos suficientes datos
        if len(df_features) < 20:
            raise ValueError(f"❌ No hay suficientes datos para entrenar. Se necesitan al menos 20 partidos con historial completo, pero solo hay {len(df_features)}. Cargá más datos históricos.")
        
        # Preparar features
        feature_cols = [col for col in df_features.columns if 'avg' in col or 'form' in col]
        X = df_features[feature_cols]
        y_home = df_features['xg_home']
        y_away = df_features['xg_away']
        
        # Split temporal
        split_idx = int(len(X) * train_split)
        
        if split_idx < 10:
            print("⚠️  Pocos datos para split. Usando 90% train / 10% test")
            split_idx = int(len(X) * 0.9)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
        y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
        
        print(f"✓ Train: {len(X_train)} partidos | Test: {len(X_test)} partidos")
        
        # Entrenar modelos
        self.gb_model_home = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4,
            random_state=42
        )
        self.gb_model_away = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4,
            random_state=42
        )
        
        self.gb_model_home.fit(X_train, y_home_train)
        self.gb_model_away.fit(X_train, y_away_train)
        
        # Evaluar
        pred_home = self.gb_model_home.predict(X_test)
        pred_away = self.gb_model_away.predict(X_test)
        
        mae_home = mean_absolute_error(y_home_test, pred_home)
        mae_away = mean_absolute_error(y_away_test, pred_away)
        
        print(f"✓ MAE Home xG: {mae_home:.3f}")
        print(f"✓ MAE Away xG: {mae_away:.3f}")
        
        # Guardar feature columns para predicción
        self.feature_cols = feature_cols
        
        return self.gb_model_home, self.gb_model_away
    
    def initialize_elo(self, initial_rating=1500):
        """Inicializa ratings ELO para todos los equipos"""
        all_teams = list(set(self.matches_df['home_team'].unique()) | 
                        set(self.matches_df['away_team'].unique()))
        
        self.elo_ratings = {team: initial_rating for team in all_teams}
        print(f"\n✓ Inicializados {len(all_teams)} equipos con ELO {initial_rating}")
    
    def update_elo(self, home_team, away_team, home_score, away_score, k=30, home_advantage=100):
        """Actualiza ratings ELO después de un partido"""
        # Rating actual
        home_rating = self.elo_ratings.get(home_team, 1500)
        away_rating = self.elo_ratings.get(away_team, 1500)
        
        # Expected score
        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
        expected_away = 1 - expected_home
        
        # Actual score
        if home_score > away_score:
            actual_home, actual_away = 1, 0
        elif home_score < away_score:
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Update
        self.elo_ratings[home_team] = home_rating + k * (actual_home - expected_home)
        self.elo_ratings[away_team] = away_rating + k * (actual_away - expected_away)
    
    def train_elo_system(self):
        """Entrena sistema ELO con historial completo"""
        print("\n🔄 Entrenando Sistema ELO...")
        self.initialize_elo()
        
        for _, row in self.matches_df.iterrows():
            self.update_elo(
                row['home_team'], 
                row['away_team'],
                row['home_score'],
                row['away_score']
            )
        
        # Mostrar top 10
        sorted_elo = sorted(self.elo_ratings.items(), key=lambda x: x[1], reverse=True)
        print("\n📊 Top 10 ELO:")
        for i, (team, rating) in enumerate(sorted_elo[:10], 1):
            print(f"  {i}. {team}: {rating:.0f}")
        
        return self.elo_ratings
    
    def poisson_bivariate(self, lambda_home, lambda_away, rho=0.0, max_goals=7):
        """Calcula probabilidades usando Poisson bivariado"""
        # Matriz de probabilidades para cada resultado
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                # Probabilidad base de Poisson
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
                # Ajuste por correlación (simplificado)
                if i == 0 and j == 0:
                    prob *= (1 + rho)
                elif (i == 0 and j == 1) or (i == 1 and j == 0):
                    prob *= (1 - rho/2)
                
                prob_matrix[i, j] = prob
        
        # Normalizar
        prob_matrix /= prob_matrix.sum()
        
        # Calcular probabilidades 1X2
        prob_home_win = np.tril(prob_matrix, -1).sum()  # Goles local > visitante
        prob_draw = np.trace(prob_matrix)  # Goles iguales
        prob_away_win = np.triu(prob_matrix, 1).sum()  # Goles visitante > local
        
        return {
            'prob_home': prob_home_win,
            'prob_draw': prob_draw,
            'prob_away': prob_away_win,
            'matrix': prob_matrix
        }
    
    def predict_match(self, home_team, away_team, method='all'):
        """Predice resultado de un partido"""
        predictions = {}
        
        # 1. Gradient Boosting (si está entrenado)
        if self.gb_model_home is not None and method in ['gb', 'all']:
            try:
                # Calcular features para ambos equipos
                features = self._calculate_team_features(home_team, away_team)
                
                if features is not None:
                    X_pred = pd.DataFrame([features])[self.feature_cols]
                    
                    xg_home_pred = max(0.1, self.gb_model_home.predict(X_pred)[0])
                    xg_away_pred = max(0.1, self.gb_model_away.predict(X_pred)[0])
                    
                    # Poisson bivariado
                    poisson_result = self.poisson_bivariate(xg_home_pred, xg_away_pred, rho=-0.1)
                    
                    predictions['gradient_boosting'] = {
                        'xg_home': xg_home_pred,
                        'xg_away': xg_away_pred,
                        'prob_home': poisson_result['prob_home'],
                        'prob_draw': poisson_result['prob_draw'],
                        'prob_away': poisson_result['prob_away']
                    }
            except Exception as e:
                print(f"⚠️  No se pudo calcular GB prediction: {e}")
        
        # 2. Sistema ELO
        if method in ['elo', 'all']:
            home_elo = self.elo_ratings.get(home_team, 1500)
            away_elo = self.elo_ratings.get(away_team, 1500)
            
            # Probabilidad esperada con ventaja de local
            prob_home_elo = 1 / (1 + 10 ** ((away_elo - (home_elo + 100)) / 400))
            prob_away_elo = 1 - prob_home_elo
            prob_draw_elo = 0.27  # Aprox histórico del fútbol argentino
            
            # Normalizar para que sumen 1
            total = prob_home_elo + prob_draw_elo + prob_away_elo
            
            predictions['elo'] = {
                'home_elo': home_elo,
                'away_elo': away_elo,
                'prob_home': prob_home_elo / total,
                'prob_draw': prob_draw_elo / total,
                'prob_away': prob_away_elo / total
            }
        
        # 3. Ensemble (si ambos disponibles)
        if 'gradient_boosting' in predictions and 'elo' in predictions:
            predictions['ensemble'] = {
                'prob_home': 0.65 * predictions['gradient_boosting']['prob_home'] + 0.35 * predictions['elo']['prob_home'],
                'prob_draw': 0.65 * predictions['gradient_boosting']['prob_draw'] + 0.35 * predictions['elo']['prob_draw'],
                'prob_away': 0.65 * predictions['gradient_boosting']['prob_away'] + 0.35 * predictions['elo']['prob_away']
            }
        
        return predictions
    
    def _calculate_team_features(self, home_team, away_team):
        """Calcula features para un partido específico basado en historial reciente"""
        # Filtrar partidos de ambos equipos
        home_matches = self.matches_df[
            (self.matches_df['home_team'] == home_team) | 
            (self.matches_df['away_team'] == home_team)
        ].tail(15)  # Últimos 15 partidos
        
        away_matches = self.matches_df[
            (self.matches_df['home_team'] == away_team) | 
            (self.matches_df['away_team'] == away_team)
        ].tail(15)
        
        # Verificar que haya suficiente historial
        if len(home_matches) < 5 or len(away_matches) < 5:
            return None
        
        features = {}
        
        # Features para equipo local
        for window in [5, 10]:
            home_data = home_matches.tail(window)
            
            home_xg = []
            home_xga = []
            home_gf = []
            
            for _, row in home_data.iterrows():
                if row['home_team'] == home_team:
                    home_xg.append(row['xg_home'])
                    home_xga.append(row['xg_away'])
                    home_gf.append(row['home_score'])
                else:
                    home_xg.append(row['xg_away'])
                    home_xga.append(row['xg_home'])
                    home_gf.append(row['away_score'])
            
            features[f'home_xg_avg_{window}'] = np.mean(home_xg) if home_xg else 1.2
            features[f'home_xga_avg_{window}'] = np.mean(home_xga) if home_xga else 1.2
            features[f'home_gf_avg_{window}'] = np.mean(home_gf) if home_gf else 1.0
            features[f'home_form_{window}'] = features[f'home_xg_avg_{window}'] - features[f'home_xga_avg_{window}']
        
        # Features para equipo visitante
        for window in [5, 10]:
            away_data = away_matches.tail(window)
            
            away_xg = []
            away_xga = []
            away_gf = []
            
            for _, row in away_data.iterrows():
                if row['away_team'] == away_team:
                    away_xg.append(row['xg_away'])
                    away_xga.append(row['xg_home'])
                    away_gf.append(row['away_score'])
                else:
                    away_xg.append(row['xg_home'])
                    away_xga.append(row['xg_away'])
                    away_gf.append(row['home_score'])
            
            features[f'away_xg_avg_{window}'] = np.mean(away_xg) if away_xg else 1.0
            features[f'away_xga_avg_{window}'] = np.mean(away_xga) if away_xga else 1.0
            features[f'away_gf_avg_{window}'] = np.mean(away_gf) if away_gf else 1.0
            features[f'away_form_{window}'] = features[f'away_xg_avg_{window}'] - features[f'away_xga_avg_{window}']
        
        return features
    
    def get_team_list(self):
        """Retorna lista de equipos disponibles"""
        return sorted(list(set(self.matches_df['home_team'].unique()) | 
                           set(self.matches_df['away_team'].unique())))


# Función helper para formato de salida
def format_prediction(predictions):
    """Formatea predicciones para mostrar"""
    output = []
    
    for method, pred in predictions.items():
        output.append(f"\n{'='*50}")
        output.append(f"Método: {method.upper().replace('_', ' ')}")
        output.append(f"{'='*50}")
        
        if 'xg_home' in pred:
            output.append(f"xG Esperado Local: {pred['xg_home']:.2f}")
            output.append(f"xG Esperado Visitante: {pred['xg_away']:.2f}")
        
        if 'home_elo' in pred:
            output.append(f"ELO Local: {pred['home_elo']:.0f}")
            output.append(f"ELO Visitante: {pred['away_elo']:.0f}")
        
        output.append(f"\nProbabilidades:")
        output.append(f"  Victoria Local: {pred['prob_home']*100:.1f}%")
        output.append(f"  Empate:         {pred['prob_draw']*100:.1f}%")
        output.append(f"  Victoria Visit: {pred['prob_away']*100:.1f}%")
    
    return '\n'.join(output)