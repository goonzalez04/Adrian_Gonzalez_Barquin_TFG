import pathlib
import pandas as pd
import pdfplumber

# Columnas de la tabla de parcelas de la declaración PAC
PROV, MUN, AGR, ZONA, POL, PAR, REC = 0, 1, 2, 3, 5, 6, 7
SUP = 12
PRODUCTO = 14

# Código de producto PAC -> nombre del cultivo en Cultivos.csv
PRODUCTOS_PAC = {"1": "trigo", "5": "cebada", "240": "titarro", "20": "barbecho"}


def tabla_parcelas(pagina):
    # La tabla de parcelas es la que empieza por "REFERENCIAS ALFANUMÉRICAS"
    for tabla in pagina.extract_tables():
        if (tabla[0][0] or "").startswith("REFERENCIAS"):
            return tabla
    return []


def leer_declaracion(ruta_pdf):
    filas = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            for fila in tabla_parcelas(pagina):
                
                if not (fila[PROV] or "").isdigit():
                    continue

                referencia = ":".join(fila[i] for i in (PROV, MUN, AGR, ZONA, POL, PAR, REC))
                superficie = float(fila[SUP])
                codigo = fila[PRODUCTO].split()[0]

                cultivo = PRODUCTOS_PAC.get(codigo)
                if cultivo is None:
                    print(f"Aviso: producto PAC {codigo} desconocido en {referencia}")
                    continue

                filas.append({"id": referencia, "superficie_ha": superficie, "cultivo": cultivo})

    return pd.DataFrame(filas)


if __name__ == "__main__":
    carpeta_pac = pathlib.Path("E:/TFG/PAC")
    parcelas = leer_declaracion(carpeta_pac / "2026.pdf")
    parcelas.to_csv(carpeta_pac / "Parcelas_2026.csv", index=False)

    print("Recintos:", len(parcelas))
    print("Superficie total:", parcelas["superficie_ha"].sum().round(2), "ha")
    print(parcelas.groupby("cultivo")["superficie_ha"].sum().round(2))
