from pathlib import Path
import numpy as np

from red_neuronal import RedNeuronal

CARPETA = Path(__file__).resolve().parent
ARCHIVO_PRESAS = CARPETA / "red_presas.npz"
ARCHIVO_DEPREDADORES = CARPETA / "red_depredadores.npz"
ARCHIVO_HISTORIAL = CARPETA / "historial_entrenamiento.npz"


def evaluar_red(red, X, y):
    predicciones = np.argmax(red.predecir_lote(X), axis=1)
    aciertos = np.sum(predicciones == y)
    precision = (aciertos / len(y)) * 100

    print(f"Ejemplos evaluados: {len(y)}")
    print(f"Aciertos: {aciertos}")
    print(f"Precisión: {precision:.2f}%")

    return precision


# Acciones:
# 0 = Arriba
# 1 = Abajo
# 2 = Izquierda
# 3 = Derecha
# 4 = Quieto


def elegir_accion(dx, dy):
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return 4

    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2

    return 1 if dy > 0 else 0


def generar_datos_presas(cantidad=3000):
    X = []
    y = []

    for _ in range(cantidad):
        dx_planta = np.random.uniform(-1, 1)
        dy_planta = np.random.uniform(-1, 1)

        dx_peligro = np.random.uniform(-1, 1)
        dy_peligro = np.random.uniform(-1, 1)

        distancia_planta = np.random.uniform(0, 1)
        distancia_peligro = np.random.uniform(0, 1)

        energia = np.random.uniform(0, 1)
        x = np.random.uniform(0, 1)
        posicion_y = np.random.uniform(0, 1)

        if distancia_peligro < 0.35:
            dx = -dx_peligro
            dy = -dy_peligro
        else:
            dx = dx_planta
            dy = dy_planta

        accion = elegir_accion(dx, dy)

        X.append([
            dx_planta,
            dy_planta,
            dx_peligro,
            dy_peligro,
            distancia_planta,
            distancia_peligro,
            energia,
            x,
            posicion_y,
            1.0
        ])

        y.append(accion)

    return np.array(X), np.array(y)


def generar_datos_depredadores(cantidad=3000):
    X = []
    y = []

    for _ in range(cantidad):
        dx_presa = np.random.uniform(-1, 1)
        dy_presa = np.random.uniform(-1, 1)

        distancia_presa = np.random.uniform(0, 1)
        energia = np.random.uniform(0, 1)

        x = np.random.uniform(0, 1)
        posicion_y = np.random.uniform(0, 1)

        accion = elegir_accion(dx_presa, dy_presa)

        X.append([
            dx_presa,
            dy_presa,
            0.0,
            0.0,
            distancia_presa,
            1.0,
            energia,
            x,
            posicion_y,
            1.0
        ])

        y.append(accion)

    return np.array(X), np.array(y)


def _preparar_split(X, y, proporcion_train=0.8):
    indices = np.random.permutation(len(X))
    corte = int(len(indices) * proporcion_train)
    return (
        X[indices[:corte]],
        y[indices[:corte]],
        X[indices[corte:]],
        y[indices[corte:]]
    )


def _combinar_historiales(anterior, nuevo, epocas_sumadas):
    claves = [
        "perdida",
        "exactitud",
        "perdida_validacion",
        "exactitud_validacion",
    ]
    resultado = {}
    for clave in claves:
        prev = anterior.get(clave, [])
        curr = nuevo.get(clave, [])
        resultado[clave] = list(prev) + list(curr)

    iter_prev = anterior.get("iteraciones", [])
    offset_iter = iter_prev[-1] if len(iter_prev) > 0 else 0
    iter_curr = [int(it) + int(offset_iter) for it in nuevo.get("iteraciones", [])]
    resultado["iteraciones"] = list(iter_prev) + iter_curr

    epocas_prev = int(anterior.get("epocas_totales", len(resultado["perdida"]) - epocas_sumadas))
    resultado["epocas_totales"] = epocas_prev + epocas_sumadas
    return resultado


def cargar_historial():
    """Carga el historial guardado si existe."""
    if not ARCHIVO_HISTORIAL.exists():
        return None

    datos = np.load(ARCHIVO_HISTORIAL)
    return {
        "presas": {
            "perdida": datos["presas_perdida"].tolist(),
            "exactitud": datos["presas_exactitud"].tolist(),
            "perdida_validacion": datos["presas_perdida_validacion"].tolist(),
            "exactitud_validacion": datos["presas_exactitud_validacion"].tolist(),
            "iteraciones": datos["presas_iteraciones"].tolist(),
            "epocas_totales": int(datos["presas_epocas_totales"])
            if "presas_epocas_totales" in datos
            else len(datos["presas_perdida"])
        },
        "depredadores": {
            "perdida": datos["depredadores_perdida"].tolist(),
            "exactitud": datos["depredadores_exactitud"].tolist(),
            "perdida_validacion": datos["depredadores_perdida_validacion"].tolist(),
            "exactitud_validacion": datos["depredadores_exactitud_validacion"].tolist(),
            "iteraciones": datos["depredadores_iteraciones"].tolist(),
            "epocas_totales": int(datos["depredadores_epocas_totales"])
            if "depredadores_epocas_totales" in datos
            else len(datos["depredadores_perdida"])
        }
    }


def guardar_historial(hist_presas, hist_depredadores):
    np.savez(
        ARCHIVO_HISTORIAL,
        presas_perdida=np.array(hist_presas["perdida"], dtype=float),
        presas_exactitud=np.array(hist_presas["exactitud"], dtype=float),
        presas_perdida_validacion=np.array(hist_presas["perdida_validacion"], dtype=float),
        presas_exactitud_validacion=np.array(hist_presas["exactitud_validacion"], dtype=float),
        presas_iteraciones=np.array(hist_presas["iteraciones"], dtype=int),
        presas_epocas_totales=int(hist_presas.get("epocas_totales", len(hist_presas["perdida"]))),

        depredadores_perdida=np.array(hist_depredadores["perdida"], dtype=float),
        depredadores_exactitud=np.array(hist_depredadores["exactitud"], dtype=float),
        depredadores_perdida_validacion=np.array(hist_depredadores["perdida_validacion"], dtype=float),
        depredadores_exactitud_validacion=np.array(hist_depredadores["exactitud_validacion"], dtype=float),
        depredadores_iteraciones=np.array(hist_depredadores["iteraciones"], dtype=int),
        depredadores_epocas_totales=int(hist_depredadores.get("epocas_totales", len(hist_depredadores["perdida"]))),
    )


def entrenar_redes(epocas=100, continuar=False, semilla=42):
    """
    Entrena presas y depredadores durante `epocas` (por defecto 100).
    Si `continuar=True`, carga los pesos y el historial existentes y suma `epocas`.
    """
    np.random.seed(semilla)

    historial_previo = cargar_historial() if continuar else None

    # ==================================
    # ENTRENAMIENTO DE LAS PRESAS
    # ==================================
    print(f"\n=== ENTRENAMIENTO DE LAS PRESAS (+{epocas} épocas) ===")
    X_presas, y_presas = generar_datos_presas(3000)
    X_train_p, y_train_p, X_test_p, y_test_p = _preparar_split(X_presas, y_presas)

    red_presas = RedNeuronal()
    if continuar and ARCHIVO_PRESAS.exists():
        red_presas.cargar(ARCHIVO_PRESAS)

    nuevo_hist_p = red_presas.entrenar(
        X_train_p,
        y_train_p,
        epocas=epocas,
        tasa=0.03,
        batch_size=64,
        X_validacion=X_test_p,
        y_validacion=y_test_p
    )

    if continuar and historial_previo is not None:
        hist_presas = _combinar_historiales(
            historial_previo["presas"], nuevo_hist_p, epocas
        )
    else:
        hist_presas = {
            **nuevo_hist_p,
            "epocas_totales": epocas
        }

    print("\n=== EVALUACIÓN DE LAS PRESAS ===")
    evaluar_red(red_presas, X_test_p, y_test_p)
    red_presas.guardar(ARCHIVO_PRESAS)

    # ==================================
    # ENTRENAMIENTO DE DEPREDADORES
    # ==================================
    print(f"\n=== ENTRENAMIENTO DE LOS DEPREDADORES (+{epocas} épocas) ===")
    X_depredadores, y_depredadores = generar_datos_depredadores(3000)
    X_train_d, y_train_d, X_test_d, y_test_d = _preparar_split(X_depredadores, y_depredadores)

    red_depredadores = RedNeuronal()
    if continuar and ARCHIVO_DEPREDADORES.exists():
        red_depredadores.cargar(ARCHIVO_DEPREDADORES)

    nuevo_hist_d = red_depredadores.entrenar(
        X_train_d,
        y_train_d,
        epocas=epocas,
        tasa=0.03,
        batch_size=64,
        X_validacion=X_test_d,
        y_validacion=y_test_d
    )

    if continuar and historial_previo is not None:
        hist_depredadores = _combinar_historiales(
            historial_previo["depredadores"], nuevo_hist_d, epocas
        )
    else:
        hist_depredadores = {
            **nuevo_hist_d,
            "epocas_totales": epocas
        }

    print("\n=== EVALUACIÓN DE LOS DEPREDADORES ===")
    evaluar_red(red_depredadores, X_test_d, y_test_d)
    red_depredadores.guardar(ARCHIVO_DEPREDADORES)

    guardar_historial(hist_presas, hist_depredadores)

    print(
        f"\n¡Bloque completado! Épocas acumuladas: "
        f"Presas={hist_presas['epocas_totales']} | "
        f"Depredadores={hist_depredadores['epocas_totales']}"
    )

    return red_presas, red_depredadores, {
        "presas": hist_presas,
        "depredadores": hist_depredadores
    }


if __name__ == "__main__":
    entrenar_redes(epocas=100, continuar=False)