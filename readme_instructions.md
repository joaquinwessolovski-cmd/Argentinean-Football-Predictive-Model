# ⚽ Sistema de Predicción de Fútbol Argentino

Sistema completo de predicción de resultados usando xG, Machine Learning (Gradient Boosting) y sistema ELO dinámico.

## 📋 Contenido

- `football_predictor.py` - Núcleo del sistema con todos los modelos
- `streamlit_app.py` - Aplicación web interactiva
- `cli_predictor.py` - Interfaz de línea de comandos
- `requirements.txt` - Dependencias del proyecto

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install pandas numpy scikit-learn scipy plotly streamlit openpyxl
```

O usando el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Preparar tus datos

Asegurate de tener dos archivos Excel:

**matches.xlsx** con columnas:
- `league` - Nombre del torneo
- `Wk` - Semana/fecha del torneo
- `date` - Fecha del partido
- `home_team` - Equipo local
- `xg_home` - Expected Goals del local
- `home_score` - Goles del local
- `away_score` - Goles del visitante
- `xg_away` - Expected Goals del visitante
- `away_team` - Equipo visitante

**stats.xlsx** con las estadísticas de equipos (se usa como referencia)

## 💻 Uso

### Opción 1: Aplicación Web con Streamlit (Recomendado)

```bash
streamlit run streamlit_app.py
```

Esto abre una interfaz web donde podés:
- ✅ Cargar tus archivos Excel
- ✅ Entrenar los modelos con un click
- ✅ Predecir resultados de partidos
- ✅ Ver rankings ELO actualizados
- ✅ Analizar equipos con gráficos interactivos
- ✅ Explorar el historial completo

### Opción 2: Línea de Comandos

```bash
python3 cli_predictor.py
```

Interfaz de terminal con menú interactivo.

### Opción 3: Uso Programático

```python
from football_predictor import FootballPredictor

# Inicializar y cargar datos
predictor = FootballPredictor()
matches_df, stats_df = predictor.load_data("matches.xlsx", "stats.xlsx")

# Entrenar modelos
predictor.train_gradient_boosting(train_split=0.85)
predictor.train_elo_system()

# Predecir un partido
predictions = predictor.predict_match("River Plate", "Boca Juniors", method='all')

# Ver resultados
for method, pred in predictions.items():
    print(f"\n{method}:")
    print(f"  Victoria Local: {pred['prob_home']*100:.1f}%")
    print(f"  Empate: {pred['prob_draw']*100:.1f}%")
    print(f"  Victoria Visitante: {pred['prob_away']*100:.1f}%")
```

## 🧠 Modelos Implementados

### 1. Gradient Boosting + Poisson Bivariado

**Cómo funciona:**
- Usa Gradient Boosting Regressor para predecir xG esperado de cada equipo
- Calcula features rolling (ventanas de 5 y 10 partidos)
- Convierte xG a probabilidades usando distribución Poisson bivariada
- Considera correlación entre goles de ambos equipos

**Ventajas:**
- ✅ Captura patrones no lineales
- ✅ Se adapta a forma reciente de equipos
- ✅ Usa xG (mejor predictor que goles reales)
- ✅ Probabilidades calibradas

### 2. Sistema ELO Dinámico

**Cómo funciona:**
- Cada equipo tiene un rating que se actualiza después de cada partido
- Victoria/empate/derrota ajustan el rating según rival
- Incluye ventaja de localía (100 puntos ELO)
- Rating inicial: 1500 puntos

**Ventajas:**
- ✅ Simple y robusto
- ✅ Se adapta rápidamente a cambios de forma
- ✅ Interpretable (rating más alto = equipo más fuerte)
- ✅ Histórico comprobado en deportes

### 3. Ensemble (Combinación)

**Cómo funciona:**
- Combina predicciones de Gradient Boosting y ELO
- Pesos: 65% GB + 35% ELO
- Aprovecha fortalezas de ambos métodos

**Ventajas:**
- ✅ Más robusto que modelos individuales
- ✅ Reduce varianza de predicciones
- ✅ Mejor performance general

## 📊 Features del Sistema

### Rolling Features
El modelo calcula promedios móviles de:
- xG a favor (últimos 5 y 10 partidos)
- xG en contra
- Goles a favor
- Forma del equipo (diferencia xG)

### Validación Temporal
- Split temporal para evitar data leakage
- Train: 85% de datos históricos
- Test: 15% más reciente
- Métricas: MAE (Mean Absolute Error)

### Poisson Bivariado
- Modela correlación entre goles de ambos equipos
- Parámetro ρ (rho) = -0.1 (correlación negativa leve)
- Genera matriz completa de probabilidades (0-0, 1-0, etc.)

## 📈 Interpretación de Resultados

### Niveles de Confianza
- **Alta**: Diferencia > 20% entre probabilidades → Predicción confiable
- **Media**: Diferencia 10-20% → Predicción moderada
- **Baja**: Diferencia < 10% → Resultado incierto

### xG (Expected Goals)
- xG > 1.5: Equipo generó muchas chances claras
- xG < 0.8: Equipo generó pocas chances
- Diferencia xG > 1.0: Dominación clara

### Ratings ELO
- 1700+: Elite (equipos grandes)
- 1550-1700: Competitivo
- 1400-1550: Promedio
- <1400: Necesita mejorar

## 🎯 Casos de Uso

### 1. Predicción Pre-Partido
```python
predictor.predict_match("Racing", "Independiente")
```

### 2. Análisis de Equipos
```python
# Ver todos los partidos de un equipo
team_matches = matches_df[
    (matches_df['home_team'] == "San Lorenzo") | 
    (matches_df['away_team'] == "San Lorenzo")
]
```

### 3. Backtesting
```python
# Entrenar solo hasta cierta fecha
historical = matches_df[matches_df['date'] < '2024-06-01']
predictor_hist = FootballPredictor()
predictor_hist.matches_df = historical
predictor_hist.train_gradient_boosting()
```

### 4. Simulación de Torneos
```python
# Predecir todas las fechas restantes
upcoming_matches = [
    ("River Plate", "Boca Juniors"),
    ("Racing", "Independiente"),
    # ... más partidos
]

for home, away in upcoming_matches:
    pred = predictor.predict_match(home, away)
    # Procesar resultados...
```

## ⚙️ Configuración Avanzada

### Ajustar Parámetros de Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

# Modificar en football_predictor.py
self.gb_model_home = GradientBoostingRegressor(
    n_estimators=150,      # Más árboles (default: 100)
    learning_rate=0.05,    # Learning rate más bajo (default: 0.1)
    max_depth=5,           # Profundidad mayor (default: 4)
    min_samples_split=10,  # Regularización
    random_state=42
)
```

### Ajustar Sistema ELO

```python
# En el método update_elo()
def update_elo(self, home_team, away_team, home_score, away_score, 
               k=40,                # Factor K más alto = más volátil
               home_advantage=120): # Ventaja de local mayor
```

### Cambiar Ventanas Rolling

```python
# En calculate_rolling_features()
def calculate_rolling_features(self, df, windows=[3, 7, 15]):
    # Usar ventanas de 3, 7 y 15 partidos
```

### Ajustar Pesos del Ensemble

```python
# En predict_match()
predictions['ensemble'] = {
    'prob_home': 0.70 * gb_prob + 0.30 * elo_prob,  # Más peso a GB
    # ...
}
```

## 📊 Métricas de Evaluación

### Para Evaluar el Modelo

```python
from sklearn.metrics import mean_absolute_error, log_loss, brier_score_loss

# MAE para xG
mae_xg = mean_absolute_error(y_true_xg, y_pred_xg)

# Log Loss para probabilidades
log_loss_score = log_loss(y_true_result, y_pred_probs)

# Brier Score (calibración de probabilidades)
brier = brier_score_loss(y_true_binary, y_pred_prob)

# Accuracy en resultados 1X2
correct = (predicted_results == actual_results).sum()
accuracy = correct / len(actual_results)
```

## 🔧 Troubleshooting

### Error: "KeyError en columnas"
- Verificá que tus archivos tengan EXACTAMENTE los nombres de columnas especificados
- Usá `df.columns` para ver las columnas reales

### Error: "No hay suficientes datos"
- Necesitás al menos 50-100 partidos para entrenar bien
- Verificá que los equipos tengan historial suficiente

### Predicciones poco realistas
- Revisá si hay outliers en xG (valores > 5.0 pueden ser errores)
- Verificá que las fechas estén ordenadas correctamente
- Aumentá el train_split si tenés muchos datos

### ELO no se actualiza correctamente
- Asegurate de llamar `train_elo_system()` después de cargar datos
- Verificá que los nombres de equipos sean consistentes (sin typos)

## 📚 Referencias y Papers

### Modelos de Predicción en Fútbol
- Dixon & Coles (1997) - "Modelling Association Football Scores"
- Constantinou & Fenton (2012) - "Solving the Problem of Inadequate Scoring Rules"

### Expected Goals (xG)
- Eggels et al. (2016) - "Expected Goals in Soccer"
- Anzer & Bauer (2021) - "A Goal Scoring Probability Model"

### Rating Systems
- Elo, A. (1978) - "The Rating of Chessplayers"
- Hvattum & Arntzen (2010) - "Using ELO ratings for match result prediction"

## 🤝 Contribuciones

### Mejoras Sugeridas
- [ ] Agregar más features (días de descanso, clima, distancia viaje)
- [ ] Implementar redes neuronales (LSTM para secuencias)
- [ ] Incorporar datos de lesiones y alineaciones
- [ ] Sistema de apuestas con Kelly Criterion
- [ ] API REST para predicciones en tiempo real
- [ ] Scraping automático de datos actualizados

## 📄 Licencia

Código de uso libre para análisis y educación. Si lo usás para fines comerciales, por favor citá la fuente.

## 💡 Tips Pro

### 1. Monitoreo Continuo
- Recalculá el modelo cada 5-10 fechas
- Monitoreá MAE y Brier Score
- Si la performance baja > 10%, re-entrenar

### 2. Ajustes Contextuales
```python
# Clásicos son más impredecibles
clasicos = [("River Plate", "Boca Juniors"), ...]
if (home, away) in clasicos:
    # Aumentar prob_draw en 5-10%
    predictions['prob_draw'] *= 1.10
```

### 3. Validación con Mercado
```python
# Comparar con odds de casas de apuestas
implied_prob = 1 / odds
if your_prob > implied_prob + 0.05:
    # Potencial value bet
```

### 4. Control de Overfitting
```python
# Cross-validation temporal
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Entrenar y evaluar
```

## 📞 Soporte

Si tenés problemas o sugerencias:
1. Revisá la sección Troubleshooting
2. Verificá que tus datos tengan el formato correcto
3. Asegurate de tener todas las dependencias instaladas

---

**¡Que disfrutes prediciendo! ⚽📊**