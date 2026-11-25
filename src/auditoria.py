from pathlib import Path
import pandas as pd

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "netflix_titles_preparado.csv"
OUTPUT_DIR = ROOT / "resultados" / "auditoria"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar dataset preparado
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("AUDITORIA DEL DATASET NETFLIX")
print("=" * 60)

# 1. Dimensiones
print("\n1. DIMENSIONES")
print("Registros:", df.shape[0])
print("Columnas:", df.shape[1])

# 2. Tipos de datos
print("\n2. TIPOS DE DATOS")
tipos = pd.DataFrame({
    "columna": df.columns,
    "tipo": df.dtypes.astype(str).values
})
print(tipos.to_string(index=False))
tipos.to_csv(OUTPUT_DIR / "tipos_datos.csv", index=False)

# 3. Valores ausentes
print("\n3. VALORES AUSENTES")
ausentes = pd.DataFrame({
    "columna": df.columns,
    "cantidad": df.isna().sum().values,
    "porcentaje": (df.isna().mean().values * 100).round(2)
})
ausentes = ausentes[ausentes["cantidad"] > 0]
print(ausentes.to_string(index=False))
ausentes.to_csv(OUTPUT_DIR / "valores_ausentes.csv", index=False)

# 4. Duplicados
print("\n4. DUPLICADOS")
print("Filas completamente duplicadas:", df.duplicated().sum())
print("Identificadores show_id duplicados:", df["show_id"].duplicated().sum())

# 5. Distribución de la variable type
print("\n5. DISTRIBUCION DE CATEGORIAS")
distribucion = df["type"].value_counts().to_frame("cantidad")
distribucion["porcentaje"] = (
    df["type"].value_counts(normalize=True) * 100
).round(2)

print(distribucion)
distribucion.to_csv(OUTPUT_DIR / "distribucion_categorias.csv")

# 6. Variables derivadas para el análisis
df["num_genres"] = df["listed_in"].str.count(",") + 1

# Duración numérica solamente para películas
peliculas = df[df["type"] == "Movie"].copy()
peliculas["duration_minutes"] = pd.to_numeric(
    peliculas["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

print("\n6. VARIABLES PARA AGRUPAMIENTO")
print("Películas:", len(peliculas))
print("Duraciones no convertidas:", peliculas["duration_minutes"].isna().sum())
print("Valores ausentes en num_genres:", df["num_genres"].isna().sum())

# 7. Estadísticas descriptivas
print("\n7. ESTADISTICAS DESCRIPTIVAS")

variables = {
    "release_year": df["release_year"],
    "num_genres": df["num_genres"],
    "duration_minutes": peliculas["duration_minutes"]
}

resumen = []

for nombre, serie in variables.items():
    serie = serie.dropna()

    resumen.append({
        "variable": nombre,
        "cantidad": len(serie),
        "media": round(serie.mean(), 2),
        "mediana": round(serie.median(), 2),
        "minimo": serie.min(),
        "maximo": serie.max(),
        "Q1": serie.quantile(0.25),
        "Q3": serie.quantile(0.75)
    })

resumen = pd.DataFrame(resumen)
print(resumen.to_string(index=False))
resumen.to_csv(OUTPUT_DIR / "estadisticas_descriptivas.csv", index=False)

# 8. Identificación de posibles valores atípicos mediante IQR
print("\n8. POSIBLES VALORES ATIPICOS (METODO IQR)")

atipicos = []
resultados_iqr = []

for nombre, serie_original in variables.items():
    serie = serie_original.dropna()

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    mascara = (
        (serie_original < limite_inferior)
        | (serie_original > limite_superior)
    )

    cantidad = int(mascara.sum())

    resultados_iqr.append({
        "variable": nombre,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "posibles_atipicos": cantidad
    })

    print(
        f"{nombre}: {cantidad} posibles valores atípicos "
        f"(límites: {limite_inferior:.2f} a {limite_superior:.2f})"
    )

    if nombre == "duration_minutes":
        identificadores = peliculas.loc[mascara, ["show_id", "title"]].copy()
    else:
        identificadores = df.loc[mascara, ["show_id", "title"]].copy()

    identificadores["variable"] = nombre
    identificadores["valor"] = serie_original[mascara].values
    atipicos.append(identificadores)

pd.DataFrame(resultados_iqr).to_csv(
    OUTPUT_DIR / "analisis_iqr.csv", index=False
)

if atipicos:
    pd.concat(atipicos, ignore_index=True).to_csv(
        OUTPUT_DIR / "posibles_atipicos.csv", index=False
    )

print("\nNo se eliminaron registros del dataset original.")

print("\nArchivos de auditoria guardados en:")
print(OUTPUT_DIR)

print("\nAUDITORIA FINALIZADA CORRECTAMENTE")
