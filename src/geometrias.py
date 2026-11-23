import sys
import json
import pathlib

import pandas as pd
import requests


URL_SIGPAC = "https://sigpac-hubcloud.es/servicioconsultassigpac/query/recinfo/{}.geojson"

CARPETA_DATOS = pathlib.Path(__file__).resolve().parent.parent / "Datos"


def descargar_geometrias(archivo=CARPETA_DATOS / "Parcelas.csv"):
    """Descarga el contorno de cada recinto del CSV y lo guarda en un .geojson al lado."""
    tabla = pd.read_csv(archivo)
    recintos = []

    for referencia in tabla["id"]:
        # "9:900:0:0:41:4:1" -> "9/900/0/0/41/4/1"
        ruta = referencia.replace(":", "/")
        respuesta = requests.get(URL_SIGPAC.format(ruta), timeout=20)
        features = respuesta.json()["features"]

        if len(features) == 0:
            print(f"No encontrado en el SIGPAC: {referencia}")
            continue

        recinto = features[0]
        
        recinto["properties"] = {"id": referencia}
        recintos.append(recinto)

    geojson = {"type": "FeatureCollection", "features": recintos}

    salida = pathlib.Path(archivo).with_suffix(".geojson")
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"Guardados {len(recintos)} de {len(tabla)} recintos en {salida}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        descargar_geometrias(pathlib.Path(sys.argv[1]))
    else:
        descargar_geometrias()
