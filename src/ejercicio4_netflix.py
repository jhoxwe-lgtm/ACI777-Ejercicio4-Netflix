from pathlib import Path
import subprocess
import sys

# Directorio principal del proyecto
ROOT = Path(__file__).resolve().parents[1]

# Orden de ejecución
SCRIPTS = [
    "auditoria.py",
    "preparacion_agrupamiento.py",
    "clasificacion.py",
    "kmeans.py",
    "dbscan.py"
]

print("=" * 65)
print("ACI777 - EJERCICIO 4 - PROYECTO NETFLIX")
print("=" * 65)

for numero, script in enumerate(SCRIPTS, start=1):

    ruta_script = ROOT / "src" / script

    if not ruta_script.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {ruta_script}"
        )

    print("\n")
    print("#" * 65)
    print(f"ETAPA {numero}/{len(SCRIPTS)}: {script}")
    print("#" * 65)

    subprocess.run(
        [sys.executable, str(ruta_script)],
        cwd=ROOT,
        check=True
    )

print("\n")
print("=" * 65)
print("PROYECTO EJECUTADO CORRECTAMENTE")
print("=" * 65)

print("\nTodas las etapas finalizaron sin errores.")
print("Los resultados se encuentran en la carpeta resultados/")
