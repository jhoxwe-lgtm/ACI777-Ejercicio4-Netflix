from pathlib import Path
import subprocess
import sys

import pandas as pd


# --------------------------------------------------
# CONFIGURACION GENERAL
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "auditoria.py",
    "preparacion_agrupamiento.py",
    "clasificacion.py",
    "kmeans.py",
    "dbscan.py",
    "visualizaciones.py"
]

RESULTADOS = ROOT / "resultados"


def cargar_csv(ruta):
    """Carga un CSV y entrega un error claro si no existe."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta}")
    return pd.read_csv(ruta)


def imprimir_separador():
    print("-" * 62)


def formato_porcentaje(valor):
    return f"{float(valor) * 100:.2f}%"


# --------------------------------------------------
# 1. EJECUTAR ETAPAS SIN MOSTRAR TODO EL TEXTO INTERNO
# --------------------------------------------------

print("=" * 62)
print("ACI777 | EJERCICIO 4 | ANALISIS DE NETFLIX")
print("=" * 62)
print("Ejecutando las etapas del proyecto...\n")

for numero, script in enumerate(SCRIPTS, start=1):

    ruta_script = ROOT / "src" / script

    if not ruta_script.exists():
        raise FileNotFoundError(f"No se encontro el script: {ruta_script}")

    print(f"[{numero}/{len(SCRIPTS)}] {script} ... ", end="", flush=True)

    proceso = subprocess.run(
        [sys.executable, str(ruta_script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if proceso.returncode != 0:
        print("ERROR")
        print("\nEl programa se detuvo en:", script)

        if proceso.stdout:
            print("\nSalida del proceso:")
            print(proceso.stdout)

        if proceso.stderr:
            print("\nDetalle del error:")
            print(proceso.stderr)

        raise SystemExit(proceso.returncode)

    print("Correcto")


# --------------------------------------------------
# 2. RESUMEN DE AUDITORIA
# --------------------------------------------------

dataset = cargar_csv(
    ROOT / "data" / "processed" / "netflix_titles_preparado.csv"
)

distribucion = dataset["type"].value_counts()
total = len(dataset)
ausentes = int(dataset.isna().sum().sum())
duplicados = int(dataset.duplicated().sum())

peliculas = int(distribucion.get("Movie", 0))
series = int(distribucion.get("TV Show", 0))

print("\n")
print("=" * 62)
print("RESUMEN PRINCIPAL")
print("=" * 62)

print("\n1. DATOS")
print(f"   Registros: {total:,} | Columnas: {len(dataset.columns)}")
print(f"   Valores ausentes: {ausentes:,} | Duplicados: {duplicados}")
print(
    f"   Peliculas: {peliculas:,} ({peliculas / total * 100:.2f}%)"
)
print(
    f"   Series: {series:,} ({series / total * 100:.2f}%)"
)


# --------------------------------------------------
# 3. RESUMEN DE CLASIFICACION
# --------------------------------------------------

metricas = cargar_csv(
    RESULTADOS / "clasificacion" / "metricas_clasificacion.csv"
)

matriz_logistica = cargar_csv(
    RESULTADOS / "clasificacion" / "matriz_regresion_logistica.csv"
).set_index(matriz_columna := "Unnamed: 0")

matriz_arbol = cargar_csv(
    RESULTADOS / "clasificacion" / "matriz_arbol_de_decision.csv"
).set_index(matriz_columna)

prueba = metricas[metricas["conjunto"] == "Prueba"]

print("\n2. CLASIFICACION (RESULTADOS EN PRUEBA)")
print("   Clase positiva: TV Show")
print("   Modelo                 Accuracy  Precision  Recall     F1")

for _, fila in prueba.iterrows():
    print(
        f"   {fila['modelo']:<22}"
        f"{float(fila['accuracy']) * 100:>7.2f}%"
        f"{float(fila['precision']) * 100:>10.2f}%"
        f"{float(fila['recall']) * 100:>9.2f}%"
        f"{float(fila['f1']) * 100:>8.2f}%"
    )


def obtener_matriz(matriz):
    return (
        int(matriz.loc["Real Movie", "Predicho Movie"]),
        int(matriz.loc["Real Movie", "Predicho TV Show"]),
        int(matriz.loc["Real TV Show", "Predicho Movie"]),
        int(matriz.loc["Real TV Show", "Predicho TV Show"])
    )


tn_l, fp_l, fn_l, tp_l = obtener_matriz(matriz_logistica)
tn_a, fp_a, fn_a, tp_a = obtener_matriz(matriz_arbol)

print("   Matrices (TN / FP / FN / TP):")
print(f"   Regresion logistica: {tn_l} / {fp_l} / {fn_l} / {tp_l}")
print(f"   Arbol de decision:   {tn_a} / {fp_a} / {fn_a} / {tp_a}")
print("   Mejor resultado en prueba: regresion logistica (diferencia pequena).")


# --------------------------------------------------
# 4. RESUMEN DE K-MEANS
# --------------------------------------------------

metricas_k = cargar_csv(
    RESULTADOS / "kmeans" / "comparacion_kmeans.csv"
)

perfiles_k = cargar_csv(
    RESULTADOS / "kmeans" / "perfiles_kmeans.csv"
)

mejor_k = metricas_k.loc[
    metricas_k["silueta_aproximada"].idxmax()
]

k_elegido = int(mejor_k["k"])

perfil_seleccionado = perfiles_k[
    perfiles_k["k"] == k_elegido
].sort_values("cluster")

print("\n3. K-MEANS")
print("   Valores probados: 3, 4, 5, 6 y 7")
print(
    f"   Seleccion: K={k_elegido} | "
    f"Silueta aproximada: {float(mejor_k['silueta_aproximada']):.4f}"
)
print("   Perfiles (grupo: peliculas, duracion media, generos medios):")

for _, fila in perfil_seleccionado.iterrows():
    print(
        f"   Grupo {int(fila['cluster'])}: "
        f"{int(fila['cantidad_peliculas']):,} peliculas | "
        f"{float(fila['duracion_media']):.1f} min | "
        f"{float(fila['generos_medios']):.2f} generos"
    )


# --------------------------------------------------
# 5. RESUMEN DE DBSCAN
# --------------------------------------------------

metricas_d = cargar_csv(
    RESULTADOS / "dbscan" / "comparacion_dbscan.csv"
)

perfiles_d = cargar_csv(
    RESULTADOS / "dbscan" / "perfiles_dbscan.csv"
)

configuracion_elegida = "eps_0.5_min_10"

seleccion_d = metricas_d[
    metricas_d["configuracion"] == configuracion_elegida
]

if seleccion_d.empty:
    raise ValueError("No se encontro la configuracion DBSCAN seleccionada.")

seleccion_d = seleccion_d.iloc[0]

perfil_d = perfiles_d[
    perfiles_d["configuracion"] == configuracion_elegida
].sort_values("grupo")

etiquetas_d = cargar_csv(
    RESULTADOS / "dbscan" / "etiquetas_dbscan.csv"
)

columna_tipos = "tipo_" + configuracion_elegida

print("\n4. DBSCAN")
print("   Configuracion seleccionada: eps=0.5 | min_samples=10")
print(
    f"   Grupos: {int(seleccion_d['cantidad_grupos'])} | "
    f"Agrupadas: {int(seleccion_d['peliculas_agrupadas']):,} | "
    f"Ruido: {int(seleccion_d['cantidad_ruido'])} "
    f"({float(seleccion_d['porcentaje_ruido']):.2f}%)"
)

if columna_tipos in etiquetas_d.columns:
    conteo_tipos = etiquetas_d[columna_tipos].value_counts()

    print(
        "   Puntos: "
        f"{int(conteo_tipos.get('Nucleo', 0)):,} nucleos | "
        f"{int(conteo_tipos.get('Frontera', 0)):,} frontera | "
        f"{int(conteo_tipos.get('Ruido', 0)):,} ruido"
    )

for _, fila in perfil_d.iterrows():
    if int(fila["grupo"]) == -1:
        nombre = "Ruido"
    else:
        nombre = f"Grupo {int(fila['grupo'])}"

    print(
        f"   {nombre}: {int(fila['cantidad_peliculas']):,} peliculas | "
        f"{float(fila['duracion_media']):.1f} min promedio"
    )


# --------------------------------------------------
# 6. VISUALIZACIONES Y CIERRE
# --------------------------------------------------

print("\n5. GRAFICOS GENERADOS")
graficos = [
    "comparacion_agrupamientos.png",
    "perfiles_comparativos.png",
    "tamanos_grupos.png",
    "tipos_puntos_dbscan.png"
]

for grafico in graficos:
    ruta = RESULTADOS / "comparacion" / grafico
    if ruta.exists():
        print(f"   - {grafico}")

print("\n")
print("=" * 62)
print("PROYECTO EJECUTADO CORRECTAMENTE")
print("=" * 62)
print("Los resultados completos siguen guardados en la carpeta resultados/.")
print("Esta vista muestra solo los indicadores principales.")
