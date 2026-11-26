from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. RUTAS DEL PROYECTO
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "netflix_titles_preparado.csv"
)

DATA_OUTPUT = ROOT / "data" / "processed"

RESULTS_OUTPUT = ROOT / "resultados" / "agrupamiento"

DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CARGAR DATASET
# --------------------------------------------------

print("=" * 60)
print("PREPARACION DE DATOS PARA AGRUPAMIENTO")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset cargado:", df.shape)


# --------------------------------------------------
# 3. SELECCIONAR PELICULAS
# --------------------------------------------------

peliculas = df[df["type"] == "Movie"].copy()

print("Peliculas seleccionadas:", len(peliculas))


# --------------------------------------------------
# 4. CREAR VARIABLES NUMERICAS
# --------------------------------------------------

# Convertir duración a minutos
peliculas["duration_minutes"] = pd.to_numeric(
    peliculas["duration"].astype("string").str.extract(r"(\d+)")[0],
    errors="coerce"
)

# Contar cantidad de géneros
peliculas["num_genres"] = (
    peliculas["listed_in"].astype("string").str.count(",") + 1
)


# --------------------------------------------------
# 5. SELECCIONAR VARIABLES
# --------------------------------------------------

features = [
    "release_year",
    "duration_minutes",
    "num_genres"
]

columnas_identificacion = ["show_id", "title"]

datos = peliculas[
    columnas_identificacion + features
].copy()

# Comprobar valores ausentes
print("\nValores ausentes en variables seleccionadas:")
print(datos[features].isna().sum())

if datos[features].isna().any().any():
    raise ValueError(
        "Existen valores ausentes. Revisar los datos antes de continuar."
    )

# Comprobar valores numéricos
if not np.isfinite(datos[features].to_numpy(dtype=float)).all():
    raise ValueError(
        "Se encontraron valores no numéricos o infinitos."
    )


# --------------------------------------------------
# 6. GUARDAR VARIABLES ORIGINALES
# --------------------------------------------------

ruta_originales = (
    DATA_OUTPUT / "peliculas_agrupamiento.csv"
)

datos.to_csv(ruta_originales, index=False)

print("\nVariables originales guardadas:")
print(ruta_originales)


# --------------------------------------------------
# 7. ESCALAR VARIABLES
# --------------------------------------------------

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(datos[features])

datos_escalados = pd.DataFrame(
    datos_escalados,
    columns=features
)

# Mantener identificadores para reconocer cada película
datos_escalados.insert(
    0, "title", datos["title"].values
)

datos_escalados.insert(
    0, "show_id", datos["show_id"].values
)

ruta_escalados = (
    DATA_OUTPUT / "peliculas_agrupamiento_escalado.csv"
)

datos_escalados.to_csv(ruta_escalados, index=False)

print("\nVariables escaladas guardadas:")
print(ruta_escalados)


# --------------------------------------------------
# 8. RESUMEN DEL ESCALAMIENTO
# --------------------------------------------------

resumen = pd.DataFrame({
    "variable": features,
    "media_original": scaler.mean_,
    "desviacion_original": scaler.scale_,
    "media_escalada": datos_escalados[features].mean().values,
    "desviacion_escalada": datos_escalados[features].std(ddof=0).values
})

resumen = resumen.round(4)

print("\nRESUMEN DEL ESCALAMIENTO")
print(resumen.to_string(index=False))

resumen.to_csv(
    RESULTS_OUTPUT / "resumen_escalamiento.csv",
    index=False
)


# --------------------------------------------------
# 9. GRAFICOS DE DISTRIBUCION
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, variable in zip(axes, features):
    ax.hist(datos[variable], bins=20, edgecolor="black")
    ax.set_title(variable)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Cantidad")

fig.suptitle("Distribución de variables originales")
plt.tight_layout()

fig.savefig(
    RESULTS_OUTPUT / "distribucion_variables.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 10. GRAFICOS DE VARIABLES ESCALADAS
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, variable in zip(axes, features):
    ax.hist(
        datos_escalados[variable],
        bins=20,
        edgecolor="black"
    )
    ax.set_title(variable)
    ax.set_xlabel("Valor escalado")
    ax.set_ylabel("Cantidad")

fig.suptitle("Distribución de variables escaladas")
plt.tight_layout()

fig.savefig(
    RESULTS_OUTPUT / "distribucion_escalada.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 11. RESULTADO FINAL
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPARACION FINALIZADA CORRECTAMENTE")
print("=" * 60)

print("Películas utilizadas:", len(datos))
print("Variables utilizadas:", features)
print("Datos originales:", ruta_originales)
print("Datos escalados:", ruta_escalados)
print("Resultados y gráficos:", RESULTS_OUTPUT)

print("\nNo se eliminaron películas ni valores atípicos.")
