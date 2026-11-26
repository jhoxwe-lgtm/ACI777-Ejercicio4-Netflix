from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. RUTAS Y CONFIGURACION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "netflix_titles_preparado.csv"
)

RESULTS_PATH = ROOT / "resultados" / "clasificacion"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# 2. CARGAR DATASET
# --------------------------------------------------

print("=" * 60)
print("CLASIFICACION SUPERVISADA: NETFLIX")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDimensiones del dataset:", df.shape)


# --------------------------------------------------
# 3. CREAR VARIABLES
# --------------------------------------------------

df["num_genres"] = df["listed_in"].str.count(",") + 1

df["has_director"] = df["director"].notna().astype(int)

FEATURES = [
    "release_year",
    "num_genres",
    "has_director"
]

TARGET = "type"

# Conservar identificadores para los resultados
datos = df[
    ["show_id", "title"] + FEATURES + [TARGET]
].dropna()

# TV Show = 1 y Movie = 0
X = datos[FEATURES]
y = (datos[TARGET] == "TV Show").astype(int)

print("\nDistribucion de clases:")
print(datos[TARGET].value_counts())

print("\nPredictores:", FEATURES)
print("Clase positiva: TV Show")


# --------------------------------------------------
# 4. DIVISION DE ENTRENAMIENTO Y PRUEBA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTAMAÑO DE LOS CONJUNTOS")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))


# --------------------------------------------------
# 5. DEFINIR LOS MODELOS
# --------------------------------------------------

modelos = {
    "Regresion logistica": Pipeline([
        ("escalamiento", StandardScaler()),
        ("clasificador", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Arbol de decision": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE
    )
}


# --------------------------------------------------
# 6. ENTRENAMIENTO Y EVALUACION
# --------------------------------------------------

resultados_metricas = []
matrices = []
predicciones = datos.loc[X_test.index, ["show_id", "title", TARGET]].copy()

for nombre, modelo in modelos.items():

    print("\n" + "-" * 60)
    print(nombre)
    print("-" * 60)

    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    for conjunto, reales, predichos in [
        ("Entrenamiento", y_train, pred_train),
        ("Prueba", y_test, pred_test)
    ]:

        resultados_metricas.append({
            "modelo": nombre,
            "conjunto": conjunto,
            "accuracy": accuracy_score(reales, predichos),
            "precision": precision_score(
                reales, predichos, zero_division=0
            ),
            "recall": recall_score(
                reales, predichos, zero_division=0
            ),
            "f1": f1_score(
                reales, predichos, zero_division=0
            )
        })

    # Matriz de confusion del conjunto de prueba
    matriz = confusion_matrix(
        y_test,
        pred_test,
        labels=[0, 1]
    )

    matrices.append((nombre, matriz))

    print("\nMatriz de confusion (prueba)")
    print(matriz)

    predicciones[f"prediccion_{nombre}"] = [
        "TV Show" if p == 1 else "Movie"
        for p in pred_test
    ]


# --------------------------------------------------
# 7. GUARDAR METRICAS
# --------------------------------------------------

metricas = pd.DataFrame(resultados_metricas)

metricas = metricas.round(4)

metricas.to_csv(
    RESULTS_PATH / "metricas_clasificacion.csv",
    index=False
)

predicciones.to_csv(
    RESULTS_PATH / "predicciones_clasificacion.csv",
    index=False
)


# --------------------------------------------------
# 8. GUARDAR MATRICES DE CONFUSION
# --------------------------------------------------

for nombre, matriz in matrices:

    archivo = (
        nombre.lower()
        .replace(" ", "_")
        .replace("ó", "o")
    )

    matriz_df = pd.DataFrame(
        matriz,
        index=["Real Movie", "Real TV Show"],
        columns=["Predicho Movie", "Predicho TV Show"]
    )

    matriz_df.to_csv(
        RESULTS_PATH / f"matriz_{archivo}.csv"
    )


# --------------------------------------------------
# 9. GRAFICO DE DISTRIBUCION
# --------------------------------------------------

conteo = datos[TARGET].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(conteo.index, conteo.values)

plt.title("Distribucion de clases")
plt.xlabel("Tipo de contenido")
plt.ylabel("Cantidad de titulos")

plt.tight_layout()
plt.savefig(
    RESULTS_PATH / "distribucion_clases.png",
    dpi=150
)
plt.close()


# --------------------------------------------------
# 10. GRAFICOS DE MATRICES DE CONFUSION
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (nombre, matriz) in zip(axes, matrices):

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Movie", "TV Show"]
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(nombre)

fig.suptitle("Matrices de confusion - Conjunto de prueba")
plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "matrices_confusion.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. MOSTRAR RESULTADOS
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMPARACION DE MODELOS")
print("=" * 60)

print(metricas.to_string(index=False))

print("\nClase positiva: TV Show")
print("Orden de las matrices: Movie, TV Show")
print("TN = Movie correctamente identificada")
print("FP = Movie identificada como TV Show")
print("FN = TV Show identificada como Movie")
print("TP = TV Show correctamente identificada")


# --------------------------------------------------
# 12. FINALIZACION
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASIFICACION FINALIZADA CORRECTAMENTE")
print("=" * 60)

print("Resultados guardados en:", RESULTS_PATH)
