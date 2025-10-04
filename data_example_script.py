"""
Script para crear archivos de ejemplo si no tenés datos reales.
Genera datos sintéticos realistas para probar el sistema.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Equipos de la Liga Profesional Argentina
EQUIPOS = [
    "River Plate", "Boca Juniors", "Racing Club", "Independiente",
    "San Lorenzo", "Huracán", "Vélez Sarsfield", "Estudiantes La Plata",
    "Gimnasia La Plata", "Talleres", "Belgrano", "Rosario Central",
    "Newell's Old Boys", "Colón", "Unión", "Lanús", "Banfield",
    "Argentinos Juniors", "Defensa y Justicia", "Godoy Cruz",
    "Arsenal", "Sarmiento", "Platense", "Barracas Central",
    "Instituto", "Tigre", "Central Córdoba SdE", "Atlético Tucumán"
]

def generate_match_result():
    """Genera resultado realista de un partido"""
    # xG promedio del fútbol argentino: ~1.3 por equipo
    xg_home = np.random.gamma(2, 0.7)  # Media ~1.4
    xg_away = np.random.gamma(2, 0.6)  # Media ~1.2 (desventaja visitante)
    
    # Goles basados en xG pero con varianza
    goals_home = np.random.poisson(xg_home)
    goals_away = np.random.poisson(xg_away)
    
    # Limitar goles a valores realistas
    goals_home = min(goals_home, 6)
    goals_away = min(goals_away, 6)
    
    return {
        'xg_home': round(xg_home, 2),
        'xg_away': round(xg_away, 2),
        'home_score': goals_home,
        'away_score': goals_away
    }

def create_example_matches(n_matches=500):
    """Crea archivo de partidos de ejemplo"""
    print(f"🔄 Generando {n_matches} partidos de ejemplo...")
    
    matches = []
    start_date = datetime(2021, 1, 1)
    
    for i in range(n_matches):
        # Seleccionar equipos aleatorios
        home = np.random.choice(EQUIPOS)
        away = np.random.choice([t for t in EQUIPOS if t != home])
        
        # Generar resultado
        result = generate_match_result()
        
        # Fecha progresiva
        date = start_date + timedelta(days=i*3)
        
        # Liga y jornada
        league = "Liga Profesional" if i < n_matches * 0.7 else "Copa de la Liga"
        wk = (i % 27) + 1
        
        matches.append({
            'league': league,
            'Wk': wk,
            'date': date,
            'home_team': home,
            'xg_home': result['xg_home'],
            'home_score': result['home_score'],
            'away_score': result['away_score'],
            'xg_away': result['xg_away'],
            'away_team': away
        })
    
    df = pd.DataFrame(matches)
    df.to_excel('matches.xlsx', index=False)
    print(f"✅ Archivo 'matches.xlsx' creado con {len(df)} partidos")
    print(f"   Rango: {df['date'].min()} a {df['date'].max()}")
    
    return df

def create_example_stats():
    """Crea archivo de estadísticas de ejemplo"""
    print("\n🔄 Generando estadísticas de equipos...")
    
    stats = []
    
    for team in EQUIPOS:
        # Generar estadísticas realistas
        mp = np.random.randint(20, 40)
        
        # Home stats
        mph = mp // 2
        wh = np.random.randint(mph//3, mph*2//3)
        dh = np.random.randint(0, mph//3)
        lh = mph - wh - dh
        gfh = np.random.randint(wh, wh*2 + dh)
        gah = np.random.randint(lh, lh*2 + dh)
        ptsh = wh*3 + dh
        
        # Away stats
        mpa = mp - mph
        wa = np.random.randint(mpa//4, mpa//2)
        da = np.random.randint(0, mpa//3)
        la = mpa - wa - da
        gfa = np.random.randint(wa//2, wa + da)
        gaa = np.random.randint(la, la*2 + da)
        ptsa = wa*3 + da
        
        # Overall
        w = wh + wa
        d = dh + da
        l = lh + la
        gf = gfh + gfa
        ga = gah + gaa
        pts = ptsh + ptsa
        
        stats.append({
            # Home
            'MPh': mph,
            'Wh': wh,
            'Dh': dh,
            'Lh': lh,
            'GFh': gfh,
            'Gah': gah,
            'GDh': gfh - gah,
            'Ptsh': ptsh,
            'Pts/MPh': round(ptsh/mph, 2) if mph > 0 else 0,
            'xG home': round(gfh * 0.95 + np.random.uniform(-0.5, 0.5), 2),
            'xGA home': round(gah * 0.95 + np.random.uniform(-0.5, 0.5), 2),
            'xGDh': round((gfh - gah) * 0.95, 2),
            'xGD/90h': round((gfh - gah) * 0.95 / mph * 90, 2) if mph > 0 else 0,
            
            # Away
            'Mpa': mpa,
            'Wa': wa,
            'Da': da,
            'La': la,
            'Gfa': gfa,
            'Gaa': gaa,
            'Gda': gfa - gaa,
            'Ptsa': ptsa,
            'Pts/Mpa': round(ptsa/mpa, 2) if mpa > 0 else 0,
            'xG away': round(gfa * 0.95 + np.random.uniform(-0.5, 0.5), 2),
            'xGA away': round(gaa * 0.95 + np.random.uniform(-0.5, 0.5), 2),
            'xGDa': round((gfa - gaa) * 0.95, 2),
            'xGD/90a': round((gfa - gaa) * 0.95 / mpa * 90, 2) if mpa > 0 else 0,
            
            # Overall
            'MP': mp,
            'W': w,
            'D': d,
            'L': l,
            'GF': gf,
            'GA': ga,
            'GD': gf - ga,
            'Pts': pts,
            'Pts/MP': round(pts/mp, 2) if mp > 0 else 0,
            'xG': round(gf * 0.95 + np.random.uniform(-1, 1), 2),
            'xGA': round(ga * 0.95 + np.random.uniform(-1, 1), 2),
            'xGD': round((gf - ga) * 0.95, 2),
            'xGD/90': round((gf - ga) * 0.95 / mp * 90, 2) if mp > 0 else 0
        })
    
    df = pd.DataFrame(stats)
    df.to_excel('stats.xlsx', index=False)
    print(f"✅ Archivo 'stats.xlsx' creado con {len(df)} equipos")
    
    return df

def main():
    print("="*60)
    print("GENERADOR DE DATOS DE EJEMPLO")
    print("="*60)
    print("\n⚠️  ATENCIÓN: Estos son datos sintéticos para pruebas.")
    print("Para predicciones reales, usá datos históricos verdaderos.\n")
    
    # Generar archivos
    matches_df = create_example_matches(n_matches=500)
    stats_df = create_example_stats()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"✅ matches.xlsx: {len(matches_df)} partidos")
    print(f"✅ stats.xlsx: {len(stats_df)} equipos")
    print(f"\n📊 Estadísticas de partidos:")
    print(f"   - Goles promedio: {(matches_df['home_score'].mean() + matches_df['away_score'].mean()):.2f}")
    print(f"   - xG promedio: {(matches_df['xg_home'].mean() + matches_df['xg_away'].mean()):.2f}")
    print(f"   - Victorias locales: {len(matches_df[matches_df['home_score'] > matches_df['away_score']])/len(matches_df)*100:.1f}%")
    print(f"   - Empates: {len(matches_df[matches_df['home_score'] == matches_df['away_score']])/len(matches_df)*100:.1f}%")
    
    print("\n🎉 ¡Archivos generados exitosamente!")
    print("\nAhora podés ejecutar:")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()