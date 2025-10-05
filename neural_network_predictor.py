"""
Neural Network Predictor para Fútbol Argentino
==============================================

Modelo de redes neuronales que usa estadísticas avanzadas para predicción de partidos.
Incluye arquitectura LSTM para capturar patrones temporales.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, regularizers
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Instalar con: pip install tensorflow")


class NeuralNetworkPredictor:
    """
    Red neuronal para predicción de resultados de fútbol usando estadísticas avanzadas.
    
    Arquitectura:
    - Capa de embedding para equipos
    - LSTM para secuencias temporales
    - Dense layers con dropout para regularización
    - Output con softmax para probabilidades 1X2
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.feature_columns = None
        self.history = None
        self.is_trained = False
        
    def load_advanced_stats(self, stats_path):
        """
        Carga estadísticas avanzadas desde Excel con múltiples hojas
        
        Args:
            stats_path: Path al archivo stats.xlsx
        
        Returns:
            dict con DataFrames de cada categoría de stats
        """
        try:
            # Cargar todas las hojas
            xl_file = pd.ExcelFile(stats_path)
            
            stats_dict = {}
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(stats_path, sheet_name=sheet_name)
                stats_dict[sheet_name] = df
                print(f"✓ Cargada hoja '{sheet_name}': {len(df)} equipos")
            
            return stats_dict
        
        except Exception as e:
            print(f"❌ Error cargando stats avanzadas: {e}")
            return None
    
    def merge_advanced_stats(self, stats_dict):
        """
        Combina todas las hojas de estadísticas en un DataFrame consolidado
        
        Args:
            stats_dict: Diccionario con DataFrames de cada hoja
        
        Returns:
            DataFrame consolidado con todas las estadísticas por equipo
        """
        if not stats_dict:
            return None
        
        # Comenzar con la primera hoja (asumiendo que tiene 'Squad')
        main_df = None
        
        for sheet_name, df in stats_dict.items():
            if 'Squad' in df.columns:
                if main_df is None:
                    main_df = df.copy()
                else:
                    # Merge por Squad, eliminando columnas duplicadas
                    cols_to_use = [col for col in df.columns if col not in main_df.columns or col == 'Squad']
                    main_df = main_df.merge(df[cols_to_use], on='Squad', how='outer', suffixes=('', f'_{sheet_name}'))
        
        # Renombrar 'Squad' a 'team' para consistencia
        if 'Squad' in main_df.columns:
            main_df = main_df.rename(columns={'Squad': 'team'})
        
        print(f"✓ Stats consolidadas: {len(main_df)} equipos, {len(main_df.columns)} features")
        
        return main_df
    
    def create_features_from_matches(self, matches_df, advanced_stats_df, window_size=10):
        """
        Crea features para entrenamiento combinando historial de partidos con stats avanzadas
        
        Args:
            matches_df: DataFrame con historial de partidos
            advanced_stats_df: DataFrame con estadísticas avanzadas por equipo
            window_size: Tamaño de ventana temporal para features rolling
        
        Returns:
            X: Features para entrenamiento
            y: Labels (0=Local, 1=Empate, 2=Visitante)
            feature_names: Lista de nombres de features
        """
        features = []
        labels = []
        
        # Preparar stats avanzadas indexadas por equipo
        if advanced_stats_df is not None:
            stats_dict = advanced_stats_df.set_index('team').to_dict('index')
        else:
            stats_dict = {}
        
        # Procesar cada partido
        matches_sorted = matches_df.sort_values('date').reset_index(drop=True)
        
        # Diccionario para tracking de historial por equipo
        team_history = {}
        
        for idx, row in matches_sorted.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Verificar que ambos equipos tengan suficiente historial
            if home_team not in team_history:
                team_history[home_team] = []
            if away_team not in team_history:
                team_history[away_team] = []
            
            if len(team_history[home_team]) >= window_size and len(team_history[away_team]) >= window_size:
                # Crear features para este partido
                match_features = []
                
                # 1. Features de historial reciente (rolling)
                home_recent = team_history[home_team][-window_size:]
                away_recent = team_history[away_team][-window_size:]
                
                # Promedios rolling
                home_xg_avg = np.mean([m['xg'] for m in home_recent])
                home_xga_avg = np.mean([m['xga'] for m in home_recent])
                home_goals_avg = np.mean([m['goals'] for m in home_recent])
                home_win_rate = np.mean([m['win'] for m in home_recent])
                
                away_xg_avg = np.mean([m['xg'] for m in away_recent])
                away_xga_avg = np.mean([m['xga'] for m in away_recent])
                away_goals_avg = np.mean([m['goals'] for m in away_recent])
                away_win_rate = np.mean([m['win'] for m in away_recent])
                
                match_features.extend([
                    home_xg_avg, home_xga_avg, home_goals_avg, home_win_rate,
                    away_xg_avg, away_xga_avg, away_goals_avg, away_win_rate
                ])
                
                # 2. Features de stats avanzadas (si disponibles)
                if home_team in stats_dict:
                    home_stats = stats_dict[home_team]
                    # Seleccionar features más relevantes (evitar NaN)
                    relevant_stats = ['Gls', 'Ast', 'xG', 'xAG', 'Poss', 'Sh', 'SoT', 
                                     'Tkl', 'Int', 'Save%', 'CS%']
                    for stat in relevant_stats:
                        if stat in home_stats and pd.notna(home_stats[stat]):
                            match_features.append(float(home_stats[stat]))
                        else:
                            match_features.append(0.0)
                else:
                    match_features.extend([0.0] * 11)  # Placeholder si no hay stats
                
                if away_team in stats_dict:
                    away_stats = stats_dict[away_team]
                    for stat in relevant_stats:
                        if stat in away_stats and pd.notna(away_stats[stat]):
                            match_features.append(float(away_stats[stat]))
                        else:
                            match_features.append(0.0)
                else:
                    match_features.extend([0.0] * 11)
                
                # 3. Features contextuales
                match_features.extend([
                    1,  # home advantage (1 para local, 0 para visitante)
                    idx / len(matches_sorted),  # posición en temporada (normalizada)
                ])
                
                features.append(match_features)
                
                # Label: resultado del partido
                if row['home_score'] > row['away_score']:
                    labels.append(0)  # Victoria local
                elif row['home_score'] < row['away_score']:
                    labels.append(2)  # Victoria visitante
                else:
                    labels.append(1)  # Empate
            
            # Actualizar historial
            # Para home team
            team_history[home_team].append({
                'xg': row['xg_home'],
                'xga': row['xg_away'],
                'goals': row['home_score'],
                'win': 1 if row['home_score'] > row['away_score'] else 0
            })
            
            # Para away team
            team_history[away_team].append({
                'xg': row['xg_away'],
                'xga': row['xg_home'],
                'goals': row['away_score'],
                'win': 1 if row['away_score'] > row['home_score'] else 0
            })
        
        X = np.array(features)
        y = np.array(labels)
        
        # Nombres de features para referencia
        feature_names = [
            'home_xg_avg', 'home_xga_avg', 'home_goals_avg', 'home_win_rate',
            'away_xg_avg', 'away_xga_avg', 'away_goals_avg', 'away_win_rate'
        ]
        feature_names.extend([f'home_{s}' for s in ['Gls', 'Ast', 'xG', 'xAG', 'Poss', 'Sh', 'SoT', 'Tkl', 'Int', 'Save%', 'CS%']])
        feature_names.extend([f'away_{s}' for s in ['Gls', 'Ast', 'xG', 'xAG', 'Poss', 'Sh', 'SoT', 'Tkl', 'Int', 'Save%', 'CS%']])
        feature_names.extend(['home_advantage', 'season_progress'])
        
        self.feature_columns = feature_names
        
        print(f"✓ Features creadas: {X.shape[0]} muestras, {X.shape[1]} features")
        print(f"✓ Distribución de labels: Local={np.sum(y==0)}, Empate={np.sum(y==1)}, Visitante={np.sum(y==2)}")
        
        return X, y, feature_names
    
    def build_model(self, input_dim, architecture='deep'):
        """
        Construye la arquitectura de red neuronal
        
        Args:
            input_dim: Número de features de entrada
            architecture: 'simple', 'deep', o 'lstm'
        
        Returns:
            Modelo de Keras compilado
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no disponible")
        
        if architecture == 'simple':
            # Red simple: 2 capas densas
            model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(3, activation='softmax')
            ])
        
        elif architecture == 'deep':
            # Red profunda con regularización
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,),
                           kernel_regularizer=regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(16, activation='relu'),
                
                layers.Dense(3, activation='softmax')
            ])
        
        else:  # lstm
            # LSTM para capturar patrones temporales
            # Reshape input para LSTM: (samples, timesteps, features)
            model = models.Sequential([
                layers.Reshape((1, input_dim), input_shape=(input_dim,)),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(3, activation='softmax')
            ])
        
        # Compilar
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ Modelo '{architecture}' construido")
        model.summary()
        
        return model
    
    def train(self, X, y, architecture='deep', epochs=100, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo de red neuronal
        
        Args:
            X: Features de entrenamiento
            y: Labels
            architecture: Tipo de arquitectura
            epochs: Número de épocas
            batch_size: Tamaño de batch
            validation_split: Proporción para validación
        
        Returns:
            history: Historial de entrenamiento
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no disponible. Instalar con: pip install tensorflow")
        
        print(f"\n🔄 Entrenando Red Neuronal ({architecture})...")
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convertir labels a one-hot
        y_categorical = to_categorical(y, num_classes=3)
        
        # Construir modelo
        self.model = self.build_model(X.shape[1], architecture)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Entrenar
        self.history = self.model.fit(
            X_scaled, y_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        
        # Métricas finales
        final_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        print(f"\n✅ Entrenamiento completado")
        print(f"   Accuracy (train): {final_acc*100:.2f}%")
        print(f"   Accuracy (val): {final_val_acc*100:.2f}%")
        
        return self.history
    
    def predict_match(self, home_team, away_team, matches_df, advanced_stats_df, window_size=10):
        """
        Predice resultado de un partido específico
        
        Args:
            home_team: Equipo local
            away_team: Equipo visitante
            matches_df: DataFrame con historial
            advanced_stats_df: Stats avanzadas
            window_size: Ventana para rolling features
        
        Returns:
            dict con probabilidades
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Crear features para este partido
        # (Similar a create_features_from_matches pero para un solo partido)
        
        # Obtener historial reciente de ambos equipos
        home_matches = matches_df[
            (matches_df['home_team'] == home_team) | (matches_df['away_team'] == home_team)
        ].tail(window_size)
        
        away_matches = matches_df[
            (matches_df['home_team'] == away_team) | (matches_df['away_team'] == away_team)
        ].tail(window_size)
        
        if len(home_matches) < window_size or len(away_matches) < window_size:
            return None  # No hay suficiente historial
        
        # Calcular rolling features
        match_features = []
        
        # Home team rolling
        home_xg = [row['xg_home'] if row['home_team'] == home_team else row['xg_away'] 
                   for _, row in home_matches.iterrows()]
        home_xga = [row['xg_away'] if row['home_team'] == home_team else row['xg_home'] 
                    for _, row in home_matches.iterrows()]
        home_goals = [row['home_score'] if row['home_team'] == home_team else row['away_score'] 
                      for _, row in home_matches.iterrows()]
        home_wins = [1 if ((row['home_team'] == home_team and row['home_score'] > row['away_score']) or 
                          (row['away_team'] == home_team and row['away_score'] > row['home_score'])) else 0
                     for _, row in home_matches.iterrows()]
        
        match_features.extend([
            np.mean(home_xg), np.mean(home_xga), np.mean(home_goals), np.mean(home_wins)
        ])
        
        # Away team rolling
        away_xg = [row['xg_home'] if row['home_team'] == away_team else row['xg_away'] 
                   for _, row in away_matches.iterrows()]
        away_xga = [row['xg_away'] if row['home_team'] == away_team else row['xg_home'] 
                    for _, row in away_matches.iterrows()]
        away_goals = [row['home_score'] if row['home_team'] == away_team else row['away_score'] 
                      for _, row in away_matches.iterrows()]
        away_wins = [1 if ((row['home_team'] == away_team and row['home_score'] > row['away_score']) or 
                          (row['away_team'] == away_team and row['away_score'] > row['home_score'])) else 0
                     for _, row in away_matches.iterrows()]
        
        match_features.extend([
            np.mean(away_xg), np.mean(away_xga), np.mean(away_goals), np.mean(away_wins)
        ])
        
        # Stats avanzadas
        if advanced_stats_df is not None:
            stats_dict = advanced_stats_df.set_index('team').to_dict('index')
            relevant_stats = ['Gls', 'Ast', 'xG', 'xAG', 'Poss', 'Sh', 'SoT', 'Tkl', 'Int', 'Save%', 'CS%']
            
            for team in [home_team, away_team]:
                if team in stats_dict:
                    team_stats = stats_dict[team]
                    for stat in relevant_stats:
                        if stat in team_stats and pd.notna(team_stats[stat]):
                            match_features.append(float(team_stats[stat]))
                        else:
                            match_features.append(0.0)
                else:
                    match_features.extend([0.0] * 11)
        else:
            match_features.extend([0.0] * 22)
        
        # Features contextuales
        match_features.extend([1, 0.5])  # home advantage, mid-season
        
        # Predecir
        X_pred = np.array([match_features])
        X_pred_scaled = self.scaler.transform(X_pred)
        
        probs = self.model.predict(X_pred_scaled, verbose=0)[0]
        
        return {
            'prob_home': float(probs[0]),
            'prob_draw': float(probs[1]),
            'prob_away': float(probs[2])
        }
    
    def save_model(self, filepath='models/neural_network'):
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        import os
        os.makedirs(filepath, exist_ok=True)
        
        self.model.save(f'{filepath}/model.h5')
        
        # Guardar scaler y otros objetos
        import pickle
        with open(f'{filepath}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{filepath}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"✓ Modelo guardado en {filepath}/")
    
    def load_model(self, filepath='models/neural_network'):
        """Carga un modelo previamente entrenado"""
        self.model = keras.models.load_model(f'{filepath}/model.h5')
        
        import pickle
        with open(f'{filepath}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f'{filepath}/feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        self.is_trained = True
        print(f"✓ Modelo cargado desde {filepath}/")
