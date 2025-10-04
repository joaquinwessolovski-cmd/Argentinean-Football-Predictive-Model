# ⚽ Predictor de Fútbol Argentino

Sistema completo de predicción de resultados usando **Expected Goals (xG)**, **Machine Learning (Gradient Boosting)** y **Sistema ELO Dinámico**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Demo

Probá la aplicación en vivo: **[https://your-app.streamlit.app](https://your-app.streamlit.app)**

## ✨ Características Principales

### 🎯 Predicción de Partidos
- **Individual**: Análisis detallado con 3 métodos (GB, ELO, Ensemble)
- **Múltiple**: Predice hasta 20 partidos simultáneamente
- **Exportación CSV**: Descarga tus predicciones

### 🏆 Simulador de Torneos
- **Liga**: Todos contra todos con simulación Monte Carlo
- **Eliminación Directa**: Copas y playoffs (4, 8, 16 equipos)
- **Estadísticas Avanzadas**: % campeón, distribución de posiciones

### 📈 Rankings y Análisis ELO
- **Ranking Actual**: Top 20 equipos por rating
- **Evolución Temporal**: Gráficos de ELO histórico
- **Comparación**: Head-to-head con estadísticas

### ⚔️ Análisis Head to Head
- Historial completo de enfrentamientos
- Gráficos interactivos de resultados
- Estadísticas detalladas (xG, goles, promedios)

### 🔥 Clásicos del Fútbol Argentino
- Superclásico (River vs Boca)
- Avellaneda (Racing vs Independiente)
- Boedo (San Lorenzo vs Huracán)
- Rosarino (Central vs Newell's)
- Platense (Gimnasia vs Estudiantes)

## 🚀 Instalación Rápida

### Prerequisitos
- Python 3.8 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/predictor-futbol-argentino.git
cd predictor-futbol-argentino

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run streamlit_app.py
```

La app se abrirá automáticamente en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
predictor-futbol-argentino/
├── streamlit_app.py              # 🌐 Aplicación web principal
├── football_predictor.py         # 🧠 Motor de predicción
├── cli_predictor.py              # 💻 Interfaz de terminal
├── advanced_examples.py          # 📚 Ejemplos avanzados
├── data_utils.py                 # 🔧 Utilidades de datos
├── create_example_data.py        # 🎲 Generador de datos sintéticos
├── requirements.txt              # 📦 Dependencias
├── .gitignore                    # 🚫 Archivos ignorados
├── .streamlit/
│   └── config.toml              # ⚙️ Configuración de Streamlit
├── matches.xlsx                  # 📊 Datos de partidos
├── stats.xlsx                    # 📈 Estadísticas de equipos
└── README.md                     # 📖 Este archivo
```

## 🧠 Modelos de Predicción

### 1️⃣ Gradient Boosting + Poisson Bivariado

**Funcionamiento:**
- Predice xG esperado usando Gradient Boosting Regressor
- Features rolling: ventanas de 5 y 10 partidos
- Conversión a probabilidades con Poisson bivariado
- Considera correlación entre goles de ambos equipos (ρ = -0.1)

**Ventajas:**
- ✅ Captura patrones no lineales complejos
- ✅ Se adapta a forma reciente de equipos
- ✅ xG es mejor predictor que goles reales
- ✅ Probabilidades bien calibradas

**Métricas:**
- MAE (Mean Absolute Error) < 0.4 en xG
- Accuracy ~45-50% en resultados 1X2

### 2️⃣ Sistema ELO Dinámico

**Funcionamiento:**
- Rating inicial: 1500 puntos
- Actualización después de cada partido
- Ventaja de localía: 100 puntos
- Factor K: 30 (volatilidad moderada)

**Ventajas:**
- ✅ Simple y robusto
- ✅ Adaptación rápida a cambios
- ✅ Interpretable y transparente
- ✅ Histórico comprobado en deportes

**Interpretación:**
- 1700+: Elite (equipos grandes)
- 1550-1700: Muy competitivo
- 1400-1550: Promedio de liga
- <1400: Zona de descenso

### 3️⃣ Ensemble (Recomendado)

**Funcionamiento:**
- Combina predicciones: 65% GB + 35% ELO
- Aprovecha fortalezas de ambos métodos
- Reduce varianza y sobreajuste

**Ventajas:**
- ✅ Más robusto que modelos individuales
- ✅ Mejor performance general
- ✅ Mayor estabilidad en predicciones

## 📊 Formato de Datos

### matches.xlsx

| Columna | Tipo | Descripción |
|---------|------|-------------|
| league | string | Nombre del torneo |
| Wk | int | Jornada/fecha |
| date | date | Fecha del partido |
| home_team | string | Equipo local |
| xg_home | float | Expected Goals local |
| home_score | int | Goles del local |
| away_score | int | Goles del visitante |
| xg_away | float | Expected Goals visitante |
| away_team | string | Equipo visitante |

### stats.xlsx

Estadísticas generales de equipos (formato con sufijos _h, _a, sin sufijo para home/away/total).

## 🎮 Guía de Uso

### 📱 Aplicación Web (Recomendado)

1. **Iniciar app**: `streamlit run streamlit_app.py`
2. **Cargar datos**: Usa precargados o sube tus archivos
3. **Entrenar**: Click en "🚀 Entrenar Modelos"
4. **Explorar tabs**:
   - 🎯 Predicción Individual
   - 📋 Predicción Múltiple
   - 🏆 Simulador de Torneos
   - 📈 Rankings & ELO
   - ⚔️ Head to Head
   - 🔥 Clásicos
   - 📊 Análisis & Histórico

### 💻 Línea de Comandos

```bash
python cli_predictor.py
```

Menú interactivo con todas las funciones principales.

### 🐍 Uso Programático

```python
from football_predictor import FootballPredictor

# Inicializar
predictor = FootballPredictor()
predictor.load_data("matches.xlsx", "stats.xlsx")

# Entrenar modelos
predictor.train_gradient_boosting()
predictor.train_elo_system()

# Predecir partido
predictions = predictor.predict_match("River Plate", "Boca Juniors")

# Mostrar resultados
for method, pred in predictions.items():
    print(f"\n{method.upper()}:")
    print(f"  Victoria Local: {pred['prob_home']*100:.1f}%")
    print(f"  Empate: {pred['prob_draw']*100:.1f}%")
    print(f"  Victoria Visitante: {pred['prob_away']*100:.1f}%")
```

## 🔬 Casos de Uso Avanzados

### Backtesting

Evalúa la performance histórica del modelo:

```python
from advanced_examples import backtest_model

results = backtest_model(
    matches_path="matches.xlsx",
    stats_path="stats.xlsx",
    test_start_date='2024-06-01'
)
# Retorna DataFrame con accuracy, log loss, etc.
```

### Simulación de Torneos

```python
from advanced_examples import simulate_league_tournament

predictor = FootballPredictor()
# ... entrenar predictor ...

matches = [
    ("River Plate", "Boca Juniors"),
    ("Racing Club", "Independiente"),
    # ... más partidos
]

team_points, team_positions = simulate_league_tournament(
    predictor, 
    matches, 
    n_simulations=1000
)

# Analizar probabilidades de campeón
```

### Value Betting

Encuentra oportunidades comparando con odds del mercado:

```python
from advanced_examples import find_value_bets

bookmaker_odds = {
    ("River Plate", "Boca Juniors"): {
        'home': 2.10, 
        'draw': 3.20, 
        'away': 3.50
    }
}

value_bets = find_value_bets(
    predictor, 
    upcoming_matches, 
    bookmaker_odds, 
    threshold=0.05
)
```

### Kelly Criterion

Calcula stakes óptimos:

```python
from advanced_examples import optimal_betting_strategy

suggestions = optimal_betting_strategy(
    predictor, 
    upcoming_matches, 
    bookmaker_odds, 
    bankroll=1000
)
```

## 🔧 Configuración Avanzada

### Ajustar Parámetros de Gradient Boosting

Edita `football_predictor.py`:

```python
self.gb_model_home = GradientBoostingRegressor(
    n_estimators=150,      # Más árboles (default: 100)
    learning_rate=0.05,    # Learning rate más bajo
    max_depth=5,           # Mayor profundidad
    min_samples_split=10,  # Regularización
    random_state=42
)
```

### Modificar Sistema ELO

```python
def update_elo(self, home_team, away_team, home_score, away_score, 
               k=40,                # Mayor volatilidad
               home_advantage=120): # Más ventaja local
```

### Cambiar Ventanas Rolling

```python
def calculate_rolling_features(self, df, windows=[3, 7, 15]):
    # Ventanas personalizadas
```

## 📊 Métricas de Evaluación

| Métrica | Uso | Objetivo |
|---------|-----|----------|
| MAE | Error en xG predicho | < 0.40 |
| Accuracy | Acierto en resultado 1X2 | > 45% |
| Brier Score | Calibración de probabilidades | < 0.20 |
| Log Loss | Calidad probabilística | < 0.95 |
| ROI | Retorno en apuestas simuladas | > 5% |

## 🎲 Generar Datos de Ejemplo

Si no tenés datos reales:

```bash
python create_example_data.py
```

Genera:
- `matches.xlsx`: 500 partidos sintéticos
- `stats.xlsx`: Estadísticas de 28 equipos

⚠️ **Nota**: Los datos sintéticos son solo para pruebas. Para predicciones reales usa datos históricos verdaderos.

## 🌐 Deploy en Streamlit Cloud

### Paso 1: Preparar Repositorio

```bash
# Asegurate de tener los datos en el repo
git add matches.xlsx stats.xlsx
git commit -m "Add data files"
git push
```

### Paso 2: Configurar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conectá tu cuenta de GitHub
3. Seleccioná el repositorio
4. Main file: `streamlit_app.py`
5. Python version: 3.9
6. Click en "Deploy"

### Paso 3: Configuración Opcional

Crea `.streamlit/secrets.toml` para variables de entorno (no commitear):

```toml
# Ejemplo de secrets
api_key = "tu_api_key_aqui"
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas!

### Cómo Contribuir

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/MiFeature`)
3. Commit cambios (`git commit -m 'Agrego MiFeature'`)
4. Push a la rama (`git push origin feature/MiFeature`)
5. Abre un Pull Request

### Ideas para Contribuir

- [ ] Agregar más features (clima, lesiones, alineaciones)
- [ ] Implementar redes neuronales (LSTM)
- [ ] API REST con FastAPI
- [ ] Scraping automático de datos actualizados
- [ ] Dashboard de performance en tiempo real
- [ ] Integración con APIs de casas de apuestas
- [ ] Tests unitarios con pytest
- [ ] Documentación extendida
- [ ] Soporte multiliga (Brasileirao, MLS, etc.)
- [ ] Modelo de goles por minuto

### Guidelines de Código

- Seguir PEP 8
- Docstrings en funciones
- Type hints donde sea posible
- Tests para nuevas features
- Actualizar README si agregás funcionalidad

## 🐛 Troubleshooting

### Error: "Found array with 0 samples"

**Problema**: No hay suficientes datos después de calcular rolling features.

**Solución**:
- Necesitás mínimo 100-150 partidos totales
- Cada equipo debe tener al menos 10 partidos
- Reducir ventanas rolling a [3, 5] en lugar de [5, 10]

### Error: "Duplicate column names"

**Problema**: Columnas con mismo nombre en DataFrame.

**Solución**: Ya está corregido en la última versión. Si persiste, hace `git pull`.

### Los modelos no predicen bien

**Problema**: Baja accuracy o predicciones poco realistas.

**Soluciones**:
- Verificá calidad de datos (outliers en xG)
- Re-entrená cada 5-10 fechas
- Aumentá `n_estimators` en Gradient Boosting
- Revisá que fechas estén ordenadas cronológicamente

### Datos precargados no se encuentran

**Problema**: App no encuentra `matches.xlsx`.

**Solución**: Asegurate que los archivos estén en la raíz del proyecto (mismo nivel que `streamlit_app.py`).

## 📚 Referencias

### Papers y Artículos

- Dixon & Coles (1997): *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*
- Constantinou & Fenton (2012): *Solving the Problem of Inadequate Scoring Rules for Assessing Probabilistic Football Forecast Models*
- Eggels et al. (2016): *Expected Goals in Soccer: Explaining Match Results using Predictive Analytics*
- Elo, A. (1978): *The Rating of Chessplayers, Past and Present*

### Recursos Online

- [FBref](https://fbref.com/) - Datos de xG
- [Understat](https://understat.com/) - Estadísticas avanzadas
- [r/soccerbetting](https://reddit.com/r/soccerbetting) - Comunidad
- [Football Data](https://www.football-data.co.uk/) - Datos históricos

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 [Tu Nombre]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 👥 Autores

- **Tu Nombre** - *Trabajo inicial* - [@tu-usuario](https://github.com/tu-usuario)

Ver la lista completa de [contribuidores](https://github.com/tu-usuario/predictor-futbol-argentino/contributors).

## 🙏 Agradecimientos

- Anthropic Claude por asistencia en desarrollo
- Comunidad de r/soccerbetting por inspiración
- StatsBomb por investigación en xG
- Todos los contribuidores del proyecto

## ⭐ Soporte

Si este proyecto te resulta útil:

- Dale una ⭐ en GitHub
- Compartilo con otros entusiastas del fútbol
- Reportá bugs y sugerencias en [Issues](https://github.com/tu-usuario/predictor-futbol-argentino/issues)
- Contribuí con código o documentación

## 📞 Contacto

- **GitHub**: [@tu-usuario](https://github.com/tu-usuario)
- **Email**: tu-email@example.com
- **Twitter**: [@tu-twitter](https://twitter.com/tu-twitter)
- **LinkedIn**: [Tu Perfil](https://linkedin.com/in/tu-perfil)

## 🔮 Roadmap

### v2.1 (Próximo)
- [ ] Cache de predicciones con Redis
- [ ] API REST con FastAPI
- [ ] Tests automatizados
- [ ] Docker container

### v3.0 (Futuro)
- [ ] Modelo de redes neuronales
- [ ] Multi-liga support
- [ ] App móvil (React Native)
- [ ] Integración con Telegram bot

## 📈 Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/tu-usuario/predictor-futbol-argentino?style=social)
![GitHub forks](https://img.shields.io/github/forks/tu-usuario/predictor-futbol-argentino?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/tu-usuario/predictor-futbol-argentino?style=social)

---

## ⚠️ Disclaimer

**Este sistema es para fines educativos y de análisis estadístico.**

- No constituye asesoramiento de apuestas
- Las apuestas deportivas conllevan riesgos financieros
- Jugá responsablemente y solo con lo que podés permitirte perder
- Consultá las leyes locales sobre apuestas en tu jurisdicción
- El desarrollador no se hace responsable por pérdidas

**Si tenés problemas con el juego, buscá ayuda:**
- Argentina: [Jugadores Anónimos](https://www.jugadoresanonimos.org.ar/)
- Línea de ayuda: 0800-333-0333

---

<div align="center">

**Hecho con ❤️ y ⚽ para la comunidad del fútbol argentino**

[⬆ Volver arriba](#-predictor-de-fútbol-argentino)

</div>