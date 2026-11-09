import numpy as np

from restricciones import repetir_familia_cultivo, rotacion_cultivos, diversificacion_pac

# 0 = trigo, 1 = cebada, 2 = titarro, 3 = barbecho
ES_CEREAL = np.array([True, True, False, False])
ES_LEGUMINOSA = np.array([False, False, True, False])
N_CULTIVOS = 4


# --- Cereal como mucho 2 años seguidos, leguminosa como mucho 1 ---

def test_dos_cereales_seguidos_se_permiten():
    secuencia = np.array([[0, 1, 2]])                 # trigo, cebada, titarro
    assert repetir_familia_cultivo(secuencia, ES_CEREAL, 2) == 0


def test_tres_cereales_seguidos_es_una_violacion():
    secuencia = np.array([[0, 1, 0]])                 # trigo, cebada, trigo
    assert repetir_familia_cultivo(secuencia, ES_CEREAL, 2) == 1


def test_cuatro_cereales_seguidos_son_dos_violaciones():
    # El tercer y el cuarto año se pasan del máximo
    secuencia = np.array([[0, 1, 0, 1]])
    assert repetir_familia_cultivo(secuencia, ES_CEREAL, 2) == 2


def test_titarro_dos_años_seguidos_es_una_violacion():
    secuencia = np.array([[2, 2, 0]])
    assert repetir_familia_cultivo(secuencia, ES_LEGUMINOSA, 1) == 1


def test_no_cuenta_violaciones_que_solo_estan_en_el_historial():
    # Los 3 primeros años son historial y ya tienen 3 cereales seguidos;
    # el plan (desde el año 3) pone titarro y no añade ninguna
    secuencia = np.array([[0, 1, 0, 2]])
    assert repetir_familia_cultivo(secuencia, ES_CEREAL, 2, desde=3) == 0


def test_suma_las_violaciones_de_todas_las_parcelas():
    secuencia = np.array([
        [0, 1, 2],                                    # cumple
        [0, 1, 0],                                    # una violación
        [1, 0, 1],                                    # una violación
    ])
    assert repetir_familia_cultivo(secuencia, ES_CEREAL, 2) == 2


# --- Rotación PAC: el mismo cultivo como mucho 3 años seguidos ---

def test_mismo_cultivo_tres_años_cumple_la_rotacion_pac():
    secuencia = np.array([[0, 0, 0, 2]])
    assert rotacion_cultivos(secuencia, np.array([20.0]), N_CULTIVOS) == 0


def test_mismo_cultivo_cuatro_años_incumple_la_rotacion_pac():
    secuencia = np.array([[0, 0, 0, 0]])
    assert rotacion_cultivos(secuencia, np.array([20.0]), N_CULTIVOS) == 1


def test_explotacion_de_10_ha_o_menos_exenta_de_rotacion():
    secuencia = np.array([[0, 0, 0, 0]])
    assert rotacion_cultivos(secuencia, np.array([10.0]), N_CULTIVOS) == 0


# --- Diversificación PAC (más de 30 ha: 3 cultivos, principal ≤ 70 %, dos principales ≤ 90 %) ---

def test_un_solo_cultivo_en_mas_de_30_ha_incumple_las_tres_reglas():
    plan = np.array([[0], [0]])                       # una campaña, todo trigo
    superficie = np.array([20.0, 20.0])
    assert diversificacion_pac(plan, superficie, N_CULTIVOS) == 3


def test_tres_cultivos_a_partes_iguales_cumplen_la_diversificacion():
    plan = np.array([[0], [1], [2]])
    superficie = np.array([15.0, 15.0, 15.0])
    assert diversificacion_pac(plan, superficie, N_CULTIVOS) == 0


def test_cultivo_principal_del_80_por_ciento_incumple_dos_reglas():
    # 80 % trigo, 15 % cebada, 5 % titarro: el principal pasa del 70 %
    # y los dos principales (95 %) pasan del 90 %
    plan = np.array([[0], [1], [2]])
    superficie = np.array([32.0, 6.0, 2.0])
    assert diversificacion_pac(plan, superficie, N_CULTIVOS) == 2


def test_cuenta_las_violaciones_de_cada_campaña():
    # Dos campañas, todo trigo en las dos: 3 violaciones por campaña
    plan = np.array([[0, 0], [0, 0]])
    superficie = np.array([20.0, 20.0])
    assert diversificacion_pac(plan, superficie, N_CULTIVOS) == 6


def test_explotacion_de_10_ha_o_menos_exenta_de_diversificacion():
    plan = np.array([[0], [0]])
    superficie = np.array([5.0, 5.0])
    assert diversificacion_pac(plan, superficie, N_CULTIVOS) == 0
