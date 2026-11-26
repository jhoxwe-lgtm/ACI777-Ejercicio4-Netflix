from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# --------------------------------------------------
# 1. RUTAS Y CONFIGURACIONES
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "resultados" / "dbscan"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "release_year",
    "duration_minutes",
    "num_genres"
]

CONFIGURACIONES = [
    {"eps": 0.3, "min_samples": 5},
    {"eps": 0.4, "min_samples": 10},
    {"eps": 0.5, "min_samples": 10}
]


# --------------------------------------------------
# 2. CARGAR LOS DATOS
# --------------------------------------------------

print("=" * 60)
print("ANALISIS DE AGRUPAMIENTO: DBSCAN")
print("=" * 60)

originales = pd.read_csv(
    DATA_PATH / "peliculas_agrupamiento.csv"
)

escalados = pd.read_csv(
    DATA_PATH / "peliculas_agrupamiento_escalado.csv"
)

if not originales["show_id"].equals(escalados["show_id"]):
    raise ValueError(
        "Los archivos no tienen el mismo orden de peliculas."
    )

X = escalados[FEATURES]

print("\nPeliculas analizadas:", len(X))
print("Variables utilizadas:", FEATURES)


# --------------------------------------------------
# 3. PREPARAR RESULTADOS
# --------------------------------------------------

resultados = []
perfiles = []

etiquetas = originales[
    ["show_id", "title"]
].copy()

puntos_ruido = []


# --------------------------------------------------
# 4. EJECUTAR CONFIGURACIONES DBSCAN
# --------------------------------------------------

for configuracion in CONFIGURACIONES:

    eps = configuracion["eps"]
    min_samples = configuracion["min_samples"]

    nombre = f"eps_{eps}_min_{min_samples}"

    print("\n" + "-" * 60)
    print(f"Configuracion: eps={eps}, min_samples={min_samples}")
    print("-" * 60)

    modelo = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    grupos = modelo.fit_predict(X)

    # Identificar grupos y ruido
    grupos_validos = sorted(
        grupo for grupo in set(grupos) if grupo != -1
    )

    cantidad_grupos = len(grupos_validos)

    cantidad_ruido = int((grupos == -1).sum())

    cantidad_agrupada = len(grupos) - cantidad_ruido

    porcentaje_ruido = (
        cantidad_ruido / len(grupos) * 100
    )

    print("Grupos detectados:", cantidad_grupos)
    print("Peliculas agrupadas:", cantidad_agrupada)
    print("Peliculas clasificadas como ruido:", cantidad_ruido)
    print(f"Porcentaje de ruido: {porcentaje_ruido:.2f}%")

    # Guardar tamaños de los grupos
    conteo = pd.Series(grupos).value_counts().sort_index()

    tamanos = []

    for grupo, cantidad in conteo.items():

        grupo = int(grupo)
        cantidad = int(cantidad)

        if grupo == -1:
            etiqueta_grupo = "Ruido"
        else:
            etiqueta_grupo = f"Grupo {grupo}"

        tamanos.append(f"{etiqueta_grupo}: {cantidad}")

        # Calcular perfiles con valores originales
        mascara = grupos == grupo
        datos_grupo = originales.loc[mascara, FEATURES]

        perfiles.append({
            "configuracion": nombre,
            "grupo": grupo,
            "tipo": etiqueta_grupo,
            "cantidad_peliculas": cantidad,
            "porcentaje_total": round(
                cantidad / len(originales) * 100, 2
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

    # Guardar resumen de la configuración
    resultados.append({
        "configuracion": nombre,
        "eps": eps,
        "min_samples": min_samples,
        "cantidad_grupos": cantidad_grupos,
        "peliculas_agrupadas": cantidad_agrupada,
        "cantidad_ruido": cantidad_ruido,
        "porcentaje_ruido": round(porcentaje_ruido, 2),
        "tamanos_grupos": " | ".join(tamanos)
    })

    # Guardar etiquetas y tipo de punto
    etiquetas[nombre] = grupos

    indices_nucleo = set(modelo.core_sample_indices_)

    etiquetas[f"tipo_{nombre}"] = [
        "Ruido" if grupo == -1 else
        ("Nucleo" if indice in indices_nucleo else "Frontera")
        for indice, grupo in enumerate(grupos)
    ]

    # Guardar películas consideradas ruido
    mascara_ruido = grupos == -1

    if mascara_ruido.any():

        ruido = originales.loc[
            mascara_ruido,
            ["show_id", "title"] + FEATURES
        ].copy()

        ruido["configuracion"] = nombre

        puntos_ruido.append(ruido)

    # --------------------------------------------------
    # 5. GRAFICO DE CADA CONFIGURACION
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    grafico = plt.scatter(
        originales["release_year"],
        originales["duration_minutes"],
        c=grupos,
        cmap="tab10",
        s=12,
        alpha=0.55
    )

    plt.colorbar(grafico, label="Grupo (-1 = ruido)")

    plt.title(
        f"DBSCAN: eps={eps}, min_samples={min_samples}\n"
        f"Grupos: {cantidad_grupos} | Ruido: {cantidad_ruido}"
    )

    plt.xlabel("Año de lanzamiento")
    plt.ylabel("Duración (minutos)")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        RESULTS_PATH / f"grafico_{nombre}.png",
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# 6. GUARDAR RESULTADOS
# --------------------------------------------------

metricas = pd.DataFrame(resultados)

metricas.to_csv(
    RESULTS_PATH / "comparacion_dbscan.csv",
    index=False
)

etiquetas.to_csv(
    RESULTS_PATH / "etiquetas_dbscan.csv",
    index=False
)

perfiles_df = pd.DataFrame(perfiles)

perfiles_df.to_csv(
    RESULTS_PATH / "perfiles_dbscan.csv",
    index=False
)

if puntos_ruido:
    ruido_df = pd.concat(
        puntos_ruido,
        ignore_index=True
    )

    ruido_df.to_csv(
        RESULTS_PATH / "puntos_ruido_dbscan.csv",
        index=False
    )


# --------------------------------------------------
# 7. MOSTRAR RESULTADOS
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMPARACION DE CONFIGURACIONES DBSCAN")
print("=" * 60)

print(metricas.to_string(index=False))

print("\nPERFILES DE LOS GRUPOS")

for configuracion in metricas["configuracion"]:

    print(f"\n--- {configuracion} ---")

    tabla = perfiles_df[
        perfiles_df["configuracion"] == configuracion
    ]

    print(tabla.to_string(index=False))


# --------------------------------------------------
# 8. FINALIZACION
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALISIS DBSCAN FINALIZADO")
print("=" * 60)

print("Resultados guardados en:", RESULTS_PATH)
print("No se eliminaron peliculas ni valores atipicos.")
