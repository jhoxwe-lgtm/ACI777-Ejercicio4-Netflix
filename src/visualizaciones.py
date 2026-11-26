from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. RUTAS Y CONFIGURACION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed"
KMEANS_PATH = ROOT / "resultados" / "kmeans"
DBSCAN_PATH = ROOT / "resultados" / "dbscan"

OUTPUT_PATH = ROOT / "resultados" / "comparacion"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "release_year",
    "duration_minutes",
    "num_genres"
]

NOMBRES_VARIABLES = [
    "Año de lanzamiento",
    "Duración (minutos)",
    "Cantidad de géneros"
]


# --------------------------------------------------
# 2. CARGAR RESULTADOS
# --------------------------------------------------

print("=" * 60)
print("VISUALIZACIONES COMPARATIVAS")
print("=" * 60)

originales = pd.read_csv(
    DATA_PATH / "peliculas_agrupamiento.csv"
)

kmeans = pd.read_csv(
    KMEANS_PATH / "etiquetas_kmeans.csv"
)

dbscan = pd.read_csv(
    DBSCAN_PATH / "etiquetas_dbscan.csv"
)

# Comprobar correspondencia de los registros
if not originales["show_id"].equals(kmeans["show_id"]):
    raise ValueError("Los registros de K-means no coinciden.")

if not originales["show_id"].equals(dbscan["show_id"]):
    raise ValueError("Los registros de DBSCAN no coinciden.")

COLUMNA_KMEANS = "cluster_k_7"
COLUMNA_DBSCAN = "eps_0.5_min_10"
COLUMNA_TIPO = "tipo_eps_0.5_min_10"

for columna in [COLUMNA_KMEANS]:
    if columna not in kmeans.columns:
        raise ValueError(f"No se encontró la columna {columna}")

for columna in [COLUMNA_DBSCAN, COLUMNA_TIPO]:
    if columna not in dbscan.columns:
        raise ValueError(
            f"No se encontró {columna}. "
            "Ejecuta nuevamente src/dbscan.py"
        )

etiquetas_kmeans = kmeans[COLUMNA_KMEANS].to_numpy()
etiquetas_dbscan = dbscan[COLUMNA_DBSCAN].to_numpy()
tipos_dbscan = dbscan[COLUMNA_TIPO].to_numpy()

print("\nPelículas analizadas:", len(originales))
print("K-means seleccionado: k=7")
print("DBSCAN seleccionado: eps=0.5, min_samples=10")


# --------------------------------------------------
# 3. CREAR TABLA COMPARATIVA DE PERFILES
# --------------------------------------------------

perfiles = []

for metodo, etiquetas in [
    ("K-means", etiquetas_kmeans),
    ("DBSCAN", etiquetas_dbscan)
]:

    for grupo in sorted(np.unique(etiquetas)):

        mascara = etiquetas == grupo
        datos_grupo = originales.loc[mascara]

        if grupo == -1:
            nombre_grupo = "Ruido"
        else:
            nombre_grupo = f"Grupo {grupo}"

        perfiles.append({
            "metodo": metodo,
            "grupo": nombre_grupo,
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
            )
        })

perfiles_df = pd.DataFrame(perfiles)

perfiles_df.to_csv(
    OUTPUT_PATH / "perfiles_comparados.csv",
    index=False
)


# --------------------------------------------------
# 4. GRAFICO COMPARATIVO DE AGRUPAMIENTOS
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# K-means
grupos_k = sorted(np.unique(etiquetas_kmeans))
colores_k = plt.get_cmap("tab10", len(grupos_k))

for posicion, grupo in enumerate(grupos_k):

    mascara = etiquetas_kmeans == grupo

    axes[0].scatter(
        originales.loc[mascara, "release_year"],
        originales.loc[mascara, "duration_minutes"],
        color=colores_k(posicion),
        s=13,
        alpha=0.55,
        label=f"Grupo {grupo} ({mascara.sum()})"
    )

axes[0].set_title("K-means: 7 grupos")
axes[0].set_xlabel("Año de lanzamiento")
axes[0].set_ylabel("Duración (minutos)")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.2)


# DBSCAN
grupos_d = sorted(np.unique(etiquetas_dbscan))

colores_d = plt.get_cmap(
    "tab10",
    max(1, len(grupos_d) - 1)
)

axes[1].scatter(
    originales.loc[etiquetas_dbscan == -1, "release_year"],
    originales.loc[etiquetas_dbscan == -1, "duration_minutes"],
    color="gray",
    s=15,
    alpha=0.65,
    label=f"Ruido ({(etiquetas_dbscan == -1).sum()})"
)

grupos_sin_ruido = [g for g in grupos_d if g != -1]

for posicion, grupo in enumerate(grupos_sin_ruido):

    mascara = etiquetas_dbscan == grupo

    axes[1].scatter(
        originales.loc[mascara, "release_year"],
        originales.loc[mascara, "duration_minutes"],
        color=colores_d(posicion),
        s=13,
        alpha=0.55,
        label=f"Grupo {grupo} ({mascara.sum()})"
    )

axes[1].set_title("DBSCAN: eps=0.5, min_samples=10")
axes[1].set_xlabel("Año de lanzamiento")
axes[1].set_ylabel("Duración (minutos)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.2)

fig.suptitle(
    "Comparación visual de agrupamientos Netflix",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "comparacion_agrupamientos.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 5. GRAFICO DE PERFILES DE LOS GRUPOS
# --------------------------------------------------

def calcular_perfil_estandarizado(etiquetas):

    tabla = originales[FEATURES].copy()
    tabla["grupo"] = etiquetas

    medias = tabla.groupby("grupo")[FEATURES].mean()

    media_general = originales[FEATURES].mean()
    desviacion_general = originales[FEATURES].std(ddof=0)

    return (medias - media_general) / desviacion_general


perfil_kmeans = calcular_perfil_estandarizado(
    etiquetas_kmeans
)

perfil_dbscan = calcular_perfil_estandarizado(
    etiquetas_dbscan
)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax, perfil, titulo in [
    (axes[0], perfil_kmeans, "K-means: 7 grupos"),
    (axes[1], perfil_dbscan, "DBSCAN: 3 grupos y ruido")
]:

    imagen = ax.imshow(
        perfil.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2,
        vmax=2
    )

    ax.set_title(titulo)
    ax.set_xticks(range(len(NOMBRES_VARIABLES)))
    ax.set_xticklabels(
        NOMBRES_VARIABLES,
        rotation=25,
        ha="right"
    )

    etiquetas_y = [
        "Ruido" if grupo == -1 else f"Grupo {grupo}"
        for grupo in perfil.index
    ]

    ax.set_yticks(range(len(etiquetas_y)))
    ax.set_yticklabels(etiquetas_y)

    ax.set_ylabel("Agrupamiento")

fig.colorbar(
    imagen,
    ax=axes,
    label="Desviación respecto al promedio general"
)

fig.suptitle(
    "Perfiles comparativos de los agrupamientos",
    fontsize=14
)

plt.savefig(
    OUTPUT_PATH / "perfiles_comparativos.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 6. GRAFICO DE TAMAÑO DE LOS GRUPOS
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, etiquetas, titulo in [
    (axes[0], etiquetas_kmeans, "K-means: k=7"),
    (axes[1], etiquetas_dbscan, "DBSCAN: eps=0.5")
]:

    conteo = pd.Series(etiquetas).value_counts().sort_index()

    nombres = [
        "Ruido" if grupo == -1 else f"Grupo {grupo}"
        for grupo in conteo.index
    ]

    barras = ax.bar(nombres, conteo.values)

    ax.set_title(titulo)
    ax.set_ylabel("Cantidad de películas")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)

    for barra, cantidad in zip(barras, conteo.values):

        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            str(cantidad),
            ha="center",
            va="bottom",
            fontsize=8
        )

fig.suptitle("Tamaño de los grupos identificados")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "tamanos_grupos.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 7. GRAFICO DE TIPOS DE PUNTOS EN DBSCAN
# --------------------------------------------------

conteo_tipos = pd.Series(tipos_dbscan).value_counts()

orden = [
    tipo for tipo in ["Nucleo", "Frontera", "Ruido"]
    if tipo in conteo_tipos.index
]

conteo_tipos = conteo_tipos.reindex(orden)

fig, ax = plt.subplots(figsize=(8, 5))

barras = ax.bar(
    conteo_tipos.index,
    conteo_tipos.values
)

ax.set_title("Clasificación de puntos en DBSCAN")
ax.set_ylabel("Cantidad de películas")
ax.grid(axis="y", alpha=0.2)

for barra, cantidad in zip(barras, conteo_tipos.values):

    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height(),
        str(cantidad),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "tipos_puntos_dbscan.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. MOSTRAR RESULTADOS
# --------------------------------------------------

print("\n" + "=" * 60)
print("PERFILES COMPARADOS")
print("=" * 60)

print(perfiles_df.to_string(index=False))

print("\nDISTRIBUCION DE PUNTOS DBSCAN")
print(conteo_tipos.to_string())

print("\nArchivos guardados en:", OUTPUT_PATH)

print("\nVISUALIZACIONES FINALIZADAS CORRECTAMENTE")
