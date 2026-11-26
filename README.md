# ACI777 · Análisis de datos de Netflix

Aplicación desarrollada en Python para analizar el catálogo de **Netflix Movies and TV Shows** mediante clasificación supervisada y agrupamiento no supervisado.

El proyecto fue desarrollado para la asignatura **ACI777 – Análisis de Datos**, Universidad de Las Américas.

## Objetivo de la aplicación

La aplicación desarrolla dos líneas de análisis sobre el catálogo:

- **Clasificación supervisada:** predecir si un título corresponde a una película (*Movie*) o a una serie (*TV Show*).
- **Agrupamiento no supervisado:** identificar perfiles de películas según su año de lanzamiento, duración en minutos y cantidad de géneros.

El programa reúne la auditoría de datos, preparación de variables, entrenamiento y evaluación de modelos, generación de agrupamientos y visualizaciones.

## Dataset

Se utilizan los siguientes archivos:

- `data/raw/netflix_titles.csv`: dataset original.
- `data/processed/netflix_titles_preparado.csv`: versión preparada utilizada en el análisis.

El dataset contiene **8.807 títulos y 12 columnas**:

| Tipo de contenido | Registros |
|---|---:|
| Películas (*Movie*) | 6.131 |
| Series (*TV Show*) | 2.676 |
| **Total** | **8.807** |

La auditoría registra los tipos de datos, valores ausentes, duplicados, distribución de clases y posibles valores atípicos. Los posibles atípicos se identifican para su revisión y no se eliminan automáticamente.

## Técnicas implementadas

### 1. Clasificación supervisada

Se entrenan y comparan dos modelos:

- Regresión logística.
- Árbol de decisión.

**Variable objetivo:** `type`  
**Clase positiva:** `TV Show`

**Predictores:**

- `release_year`: año de lanzamiento.
- `num_genres`: cantidad de géneros registrados.
- `has_director`: indicador de disponibilidad de director.

Se utiliza una partición estratificada de **80 % para entrenamiento y 20 % para prueba**, con `random_state=42`.

La regresión logística se implementa mediante un `Pipeline` con `StandardScaler`, de manera que el escalador se ajusta con los datos de entrenamiento. El árbol de decisión utiliza una profundidad máxima de 5.

La variable `duration` se excluye de los predictores porque su formato distingue casi directamente películas y series, lo que produciría fuga de información.

La evaluación considera:

- Accuracy (exactitud).
- Precision (precisión).
- Recall (exhaustividad).
- F1.
- Matrices de confusión.

**Resultados obtenidos en el conjunto de prueba:**

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Regresión logística | 95,63 % | 93,05 % | 92,52 % | 92,78 % |
| Árbol de decisión | 95,57 % | 93,03 % | 92,34 % | 92,68 % |

Los resultados son muy similares. La regresión logística presenta métricas ligeramente superiores en esta partición.

### 2. Agrupamiento con K-means

K-means se aplica exclusivamente a las **6.131 películas**, utilizando tres variables:

- `release_year`.
- `duration_minutes`.
- `num_genres`.

Las variables se escalan mediante `StandardScaler`. Se comparan cinco valores de K: **3, 4, 5, 6 y 7**, considerando la inercia, el coeficiente de silueta y la interpretación de los perfiles.

La configuración utilizada en la comparación final es **K=7**, que obtuvo una silueta aproximada de **0,4110**. Este coeficiente se calculó sobre una muestra reproducible de hasta 2.000 registros; la inercia se calculó con todos los registros.

### 3. Agrupamiento con DBSCAN

Se comparan tres configuraciones:

| `eps` | `min_samples` |
|---:|---:|
| 0,3 | 5 |
| 0,4 | 10 |
| 0,5 | 10 |

Se analizan la cantidad de grupos, el tamaño de cada grupo, la cobertura y las observaciones etiquetadas como ruido. También se identifican puntos núcleo y frontera.

La configuración utilizada en la comparación final es **eps=0,5 y min_samples=10**. Identificó tres grupos y 125 películas como ruido (2,04 % del conjunto analizado).

Los puntos de ruido representan observaciones que no pertenecen a una región suficientemente densa según los parámetros seleccionados; no se consideran automáticamente errores o registros incorrectos.

## Estructura del proyecto

```text
ACI777-Ejercicio4-Netflix/
│
├── data/
│   ├── raw/
│   │   └── netflix_titles.csv
│   └── processed/
│       ├── netflix_titles_preparado.csv
│       ├── peliculas_agrupamiento.csv
│       └── peliculas_agrupamiento_escalado.csv
│
├── src/
│   ├── auditoria.py
│   ├── preparacion_agrupamiento.py
│   ├── clasificacion.py
│   ├── kmeans.py
│   ├── dbscan.py
│   ├── visualizaciones.py
│   └── ejercicio4_netflix.py
│
├── resultados/
│   ├── auditoria/
│   ├── agrupamiento/
│   ├── clasificacion/
│   ├── kmeans/
│   ├── dbscan/
│   └── comparacion/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Git y GitHub
- GitHub Codespaces

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/jhoxwe-lgtm/ACI777-Ejercicio4-Netflix.git
cd ACI777-Ejercicio4-Netflix
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Linux o GitHub Codespaces:

```bash
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución

Desde la raíz del repositorio, ejecutar:

```bash
python src/ejercicio4_netflix.py
```

El programa ejecuta automáticamente las siguientes etapas:

1. Auditoría del dataset.
2. Preparación y escalamiento de variables para agrupamiento.
3. Clasificación supervisada.
4. Comparación de configuraciones K-means.
5. Comparación de configuraciones DBSCAN.
6. Generación de visualizaciones comparativas.

Al finalizar sin errores, se muestra el mensaje:

`PROYECTO EJECUTADO CORRECTAMENTE`

## Resultados generados

Los archivos generados se organizan dentro de `resultados/`:

| Carpeta | Contenido |
|---|---|
| `auditoria/` | Tipos de datos, ausencias, duplicados, estadísticas y posibles atípicos. |
| `agrupamiento/` | Datos originales de películas para agrupamiento, datos escalados y gráficos de distribución. |
| `clasificacion/` | Métricas, predicciones, matrices de confusión y gráficos. |
| `kmeans/` | Comparación de K, inercia, silueta, etiquetas, perfiles y gráficos. |
| `dbscan/` | Configuraciones, etiquetas, perfiles, puntos de ruido y gráficos. |
| `comparacion/` | Perfiles comparados y visualizaciones de ambos métodos. |

## Consideraciones del análisis

- La clasificación utiliza una variable objetivo categórica existente en el dataset.
- Los agrupamientos describen patrones descubiertos por los algoritmos; no son clases verdaderas.
- K-means requiere definir previamente la cantidad de grupos.
- DBSCAN depende de los parámetros de densidad `eps` y `min_samples`.
- El escalamiento es necesario para que las variables de distinta magnitud sean comparables al calcular distancias.
- Los resultados describen el catálogo analizado y no permiten establecer relaciones causales.

## Reproducibilidad

Los análisis están organizados en scripts independientes y se pueden ejecutar en conjunto desde `src/ejercicio4_netflix.py`. Las dependencias están declaradas en `requirements.txt` y los resultados se generan dentro de la carpeta `resultados/`.

---

**Asignatura:** ACI777 – Análisis de Datos  
**Institución:** Universidad de Las Américas  
**Proyecto:** Clasificación y agrupamiento aplicado al catálogo de Netflix.
