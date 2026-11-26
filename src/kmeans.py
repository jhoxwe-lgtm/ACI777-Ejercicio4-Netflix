from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. RUTAS Y CONFIGURACION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "resultados" / "kmeans"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 4, 5, 6, 7]

FEATURES = [
    "release_year",
    "duration_minutes",
    "num_genres"
]


# --------------------------------------------------
# 2. CARGAR LOS DATOS
# --------------------------------------------------

print("=" * 60)
print("ANALISIS DE AGRUPAMIENTO: K-MEANS")
print("=" * 60)

originales = pd.read_csv(
    DATA_PATH / "peliculas_agrupamiento.csv"
)

escalados = pd.read_csv(
    DATA_PATH / "peliculas_agrupamiento_escalado.csv"
)

# Comprobar correspondencia de registros
if not originales["show_id"].equals(escalados["show_id"]):
    raise ValueError("Los archivos no tienen el mismo orden de peliculas.")

X = escalados[FEATURES]

print("\nPeliculas analizadas:", len(X))
print("Variables utilizadas:", FEATURES)
print("Valores de K:", K_VALUES)


# --------------------------------------------------
# 3. ENTRENAR Y EVALUAR K-MEANS
# --------------------------------------------------

resultados = []
etiquetas = originales[["show_id", "title"]].copy()
perfiles = []

for k in K_VALUES:

    print(f"\nEvaluando K = {k}...")

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
        max_iter=300
    )

    grupos = modelo.fit_predict(X)

    # Silueta aproximada usando una muestra reproducible
    silueta = silhouette_score(
        X,
        grupos,
        sample_size=min(2000, len(X)),
        random_state=42
    )

    resultados.append({
        "k": k,
        "inercia": modelo.inertia_,
        "silueta_aproximada": silueta
    })

    etiquetas[f"cluster_k_{k}"] = grupos

    # Perfil de cada grupo con variables originales
    datos_perfil = originales[FEATURES].copy()
    datos_perfil["cluster"] = grupos

    for grupo, datos_grupo in datos_perfil.groupby("cluster"):

        perfiles.append({
            "k": k,
            "cluster": grupo,
            "cantidad_peliculas": len(datos_grupo),
            "porcentaje": round(
                len(datos_grupo) / len(originales) * 100, 2
            ),
            "anio_medio": round(
                datos_grupo["release_year"].mean(), 2
            ),
            "duracion_media": round(
                datos_grupo["duration_minutes"].mean(), 2
            ),
            "generos_medios": round(
                datos_grupo["num_genres"].mean(), 2
            ),
            "duracion_mediana": round(
                datos_grupo["duration_minutes"].median(), 2
            )
        })


# --------------------------------------------------
# 4. GUARDAR RESULTADOS
# --------------------------------------------------

metricas = pd.DataFrame(resultados)
metricas.to_csv(
    RESULTS_PATH / "comparacion_kmeans.csv",
    index=False
)

etiquetas.to_csv(
    RESULTS_PATH / "etiquetas_kmeans.csv",
    index=False
)

perfiles_df = pd.DataFrame(perfiles)
perfiles_df.to_csv(
    RESULTS_PATH / "perfiles_kmeans.csv",
    index=False
)


# --------------------------------------------------
# 5. GRAFICO DE INERCIA
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    metricas["k"],
    metricas["inercia"],
    marker="o"
)

plt.title("Metodo del codo - K-means")
plt.xlabel("Cantidad de grupos (k)")
plt.ylabel("Inercia")
plt.xticks(K_VALUES)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "grafico_inercia.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 6. GRAFICO DE SILUETA
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    metricas["k"],
    metricas["silueta_aproximada"],
    marker="o"
)

plt.title("Coeficiente de silueta - K-means")
plt.xlabel("Cantidad de grupos (k)")
plt.ylabel("Silueta aproximada")
plt.xticks(K_VALUES)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "grafico_silueta.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 7. MOSTRAR RESULTADOS
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMPARACION DE CONFIGURACIONES")
print("=" * 60)

print(metricas.round(4).to_string(index=False))

print("\nPERFILES DE LOS GRUPOS")

for k in K_VALUES:

    print(f"\n--- K = {k} ---")

    tabla = perfiles_df[perfiles_df["k"] == k]

    print(tabla.to_string(index=False))


# --------------------------------------------------
# 8. FINALIZACION
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALISIS K-MEANS FINALIZADO")
print("=" * 60)

print("\nArchivos guardados en:", RESULTS_PATH)
print("No se modificaron los datasets originales.")
