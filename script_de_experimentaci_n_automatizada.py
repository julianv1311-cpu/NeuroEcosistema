import csv
import os
from collections import defaultdict
from pathlib import Path

# Configurar Pygame en modo sin pantalla (headless) antes de inicializar
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from configuracion import ESCENARIO_A, ESCENARIO_B, ESCENARIO_C, FPS
from ecosistema import (
    Ecosistema,
    MODO_ALEATORIO,
    MODO_ENTRENADO,
    MODO_REGLAS,
)
from entrenamiento import (
    ARCHIVO_DEPREDADORES,
    ARCHIVO_PRESAS,
    _preparar_split,
    entrenar_redes,
    generar_datos_depredadores,
    generar_datos_presas,
)
from red_neuronal import RedNeuronal

CARPETA = Path(__file__).resolve().parent
ARCHIVO_RESULTADOS = CARPETA / "resultados_experimentos.csv"
ARCHIVO_RESUMEN = CARPETA / "resumen_experimentos.csv"
ARCHIVO_PRESAS_ALT = CARPETA / "red_presas_16_8.npz"
ARCHIVO_DEPREDADORES_ALT = CARPETA / "red_depredadores_16_8.npz"

# Configuración experimental reproducible
SEMILLAS = [42, 7, 13, 21, 99]
ESCENARIOS = [ESCENARIO_A, ESCENARIO_B, ESCENARIO_C]
FRAMES_POR_EPISODIO = 1800  # 30 segundos simulados a 60 FPS
EPOCAS_ARQ_ALT = 300

CONFIGURACIONES = [
    "ALEATORIO",
    "REGLAS",
    "RED_SIN_ENTRENAR",
    "RED_32_16",
    "RED_16_8",
]


def asegurar_modelos_base():
    """Garantiza que existan los modelos base (10-32-16-5) antes de evaluar."""
    if not ARCHIVO_PRESAS.exists() or not ARCHIVO_DEPREDADORES.exists():
        print("No se encontraron modelos base. Entrenando 100 épocas iniciales...")
        entrenar_redes(epocas=100, continuar=False, semilla=42)


def preparar_arquitectura_alternativa(epocas=EPOCAS_ARQ_ALT, semilla=42):
    """
    Carga o entrena la segunda configuración neuronal (10-16-8-5)
    para la comparación obligatoria entre dos arquitecturas.
    """
    red_p = RedNeuronal(entradas=10, oculta1=16, oculta2=8, salidas=5)
    red_d = RedNeuronal(entradas=10, oculta1=16, oculta2=8, salidas=5)

    if ARCHIVO_PRESAS_ALT.exists() and ARCHIVO_DEPREDADORES_ALT.exists():
        red_p.cargar(ARCHIVO_PRESAS_ALT)
        red_d.cargar(ARCHIVO_DEPREDADORES_ALT)
        print("Arquitectura alternativa (10-16-8-5) cargada desde disco.")
        return red_p, red_d

    print(f"\nEntrenando arquitectura alternativa 10-16-8-5 ({epocas} épocas)...")
    np.random.seed(semilla)

    X_p, y_p = generar_datos_presas(3000)
    Xp_tr, yp_tr, Xp_val, yp_val = _preparar_split(X_p, y_p)
    red_p.entrenar(
        Xp_tr,
        yp_tr,
        epocas=epocas,
        tasa=0.03,
        batch_size=64,
        X_validacion=Xp_val,
        y_validacion=yp_val,
    )
    red_p.guardar(ARCHIVO_PRESAS_ALT)

    X_d, y_d = generar_datos_depredadores(3000)
    Xd_tr, yd_tr, Xd_val, yd_val = _preparar_split(X_d, y_d)
    red_d.entrenar(
        Xd_tr,
        yd_tr,
        epocas=epocas,
        tasa=0.03,
        batch_size=64,
        X_validacion=Xd_val,
        y_validacion=yd_val,
    )
    red_d.guardar(ARCHIVO_DEPREDADORES_ALT)

    return red_p, red_d


def asignar_redes(ecosistema, red_presas, red_depredadores):
    ecosistema.red_presas = red_presas
    ecosistema.red_depredadores = red_depredadores
    for presa in ecosistema.presas:
        presa.red = red_presas
    for depredador in ecosistema.depredadores:
        depredador.red = red_depredadores


def simular_episodio(escenario, configuracion, semilla, redes_alt):
    """Ejecuta un episodio completo y devuelve todas las métricas exigidas."""
    eco = Ecosistema(semilla=semilla, escenario=escenario)
    plantas_iniciales = len(eco.plantas)

    if configuracion == "ALEATORIO":
        eco.cambiar_modo(MODO_ALEATORIO)

    elif configuracion == "REGLAS":
        eco.cambiar_modo(MODO_REGLAS)

    elif configuracion == "RED_SIN_ENTRENAR":
        np.random.seed(semilla)
        red_p_ini = RedNeuronal(entradas=10, oculta1=32, oculta2=16, salidas=5)
        red_d_ini = RedNeuronal(entradas=10, oculta1=32, oculta2=16, salidas=5)
        asignar_redes(eco, red_p_ini, red_d_ini)
        eco.cambiar_modo(MODO_ENTRENADO)

    elif configuracion == "RED_32_16":
        eco.cambiar_modo(MODO_ENTRENADO)

    elif configuracion == "RED_16_8":
        red_p_alt, red_d_alt = redes_alt
        asignar_redes(eco, red_p_alt, red_d_alt)
        eco.cambiar_modo(MODO_ENTRENADO)

    else:
        raise ValueError(f"Configuración no reconocida: {configuracion}")

    capturas_totales = 0
    tiempo_primera_captura = None
    frames_ejecutados = 0

    for frame in range(1, FRAMES_POR_EPISODIO + 1):
        frames_ejecutados = frame

        # Paso 1: Presas (movimiento y alimentación)
        for presa in eco.presas:
            presa.mover(eco.plantas, eco.depredadores)
            presa.comer(eco.plantas)

        vivas_antes_caza = sum(1 for p in eco.presas if p.viva)

        # Paso 2: Depredadores (persecución y caza)
        for depredador in eco.depredadores:
            depredador.mover(eco.presas)
            depredador.cazar(eco.presas)

        vivas_despues_caza = sum(1 for p in eco.presas if p.viva)
        capturas_frame = max(0, vivas_antes_caza - vivas_despues_caza)

        if capturas_frame > 0:
            capturas_totales += capturas_frame
            if tiempo_primera_captura is None:
                tiempo_primera_captura = round(frame / FPS, 2)

        # Finalización anticipada si una especie se extingue por completo
        depredadores_vivos = sum(1 for d in eco.depredadores if d.vivo)
        if vivas_despues_caza == 0 or depredadores_vivos == 0:
            break

    presas_vivas = sum(1 for p in eco.presas if p.viva)
    presas_muertas = len(eco.presas) - presas_vivas
    depredadores_vivos = sum(1 for d in eco.depredadores if d.vivo)
    depredadores_muertos = len(eco.depredadores) - depredadores_vivos
    alimento_consumido = plantas_iniciales - len(eco.plantas)

    energia_presas = float(
        np.mean([max(0.0, p.energia) for p in eco.presas])
    )
    energia_depredadores = float(
        np.mean([max(0.0, d.energia) for d in eco.depredadores])
    )

    duracion_seg = round(frames_ejecutados / FPS, 2)

    return {
        "escenario": escenario,
        "configuracion": configuracion,
        "semilla": semilla,
        "duracion_seg": duracion_seg,
        "presas_vivas": presas_vivas,
        "presas_muertas": presas_muertas,
        "alimento_consumido": alimento_consumido,
        "capturas_depredadores": capturas_totales,
        "depredadores_vivos": depredadores_vivos,
        "depredadores_muertos": depredadores_muertos,
        "energia_final_presas": round(energia_presas, 2),
        "energia_final_depredadores": round(energia_depredadores, 2),
        "tiempo_primera_captura_seg": (
            tiempo_primera_captura if tiempo_primera_captura is not None else ""
        ),
    }


def guardar_csv_crudo(registros):
    columnas = [
        "escenario",
        "configuracion",
        "semilla",
        "duracion_seg",
        "presas_vivas",
        "presas_muertas",
        "alimento_consumido",
        "capturas_depredadores",
        "depredadores_vivos",
        "depredadores_muertos",
        "energia_final_presas",
        "energia_final_depredadores",
        "tiempo_primera_captura_seg",
    ]

    with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas, delimiter=";")
        escritor.writeheader()
        escritor.writerows(registros)

    print(f"\nDatos crudos guardados en: {ARCHIVO_RESULTADOS}")


def generar_y_mostrar_resumen(registros):
    grupos = defaultdict(list)
    for fila in registros:
        clave = (fila["escenario"], fila["configuracion"])
        grupos[clave].append(fila)

    filas_resumen = []
    print("\n" + "=" * 108)
    print("RESUMEN EXPERIMENTAL (MEDIA ± DESVIACIÓN ESTÁNDAR) — LISTO PARA LA SECCIÓN 5 DEL INFORME")
    print("=" * 108)
    encabezado = (
        f"{'Escenario':<12} | {'Configuración':<17} | {'Presas Vivas':<13} | "
        f"{'Alim. Cons.':<12} | {'Capturas':<11} | {'Energ. Presas':<14} | {'1ra Capt. (s)':<13}"
    )
    print(encabezado)
    print("-" * 108)

    for escenario in ESCENARIOS:
        for config in CONFIGURACIONES:
            datos = grupos[(escenario, config)]
            n = len(datos)

            def fmt_stat(valores):
                return f"{np.mean(valores):.2f} ± {np.std(valores):.2f}"

            presas_v = [d["presas_vivas"] for d in datos]
            alimento = [d["alimento_consumido"] for d in datos]
            capturas = [d["capturas_depredadores"] for d in datos]
            ener_p = [d["energia_final_presas"] for d in datos]
            ener_d = [d["energia_final_depredadores"] for d in datos]
            tiempos = [
                d["tiempo_primera_captura_seg"]
                for d in datos
                if d["tiempo_primera_captura_seg"] != ""
            ]
            t_cap_str = fmt_stat(tiempos) if tiempos else "Sin captura"

            print(
                f"{escenario:<12} | {config:<17} | {fmt_stat(presas_v):<13} | "
                f"{fmt_stat(alimento):<12} | {fmt_stat(capturas):<11} | "
                f"{fmt_stat(ener_p):<14} | {t_cap_str:<13}"
            )

            filas_resumen.append({
                "escenario": escenario,
                "configuracion": config,
                "n_semillas": n,
                "presas_vivas_media_std": fmt_stat(presas_v),
                "alimento_consumido_media_std": fmt_stat(alimento),
                "capturas_media_std": fmt_stat(capturas),
                "energia_presas_media_std": fmt_stat(ener_p),
                "energia_depredadores_media_std": fmt_stat(ener_d),
                "tiempo_primera_captura_media_std": t_cap_str,
            })

    with open(ARCHIVO_RESUMEN, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=list(filas_resumen[0].keys()),
            delimiter=";",
        )
        escritor.writeheader()
        escritor.writerows(filas_resumen)

    print("=" * 108)
    print(f"Resumen estadístico guardado en: {ARCHIVO_RESUMEN}")


def main():
    pygame.init()
    # Necesario para que pygame.image.load().convert_alpha() funcione en modo headless
    pygame.display.set_mode((1, 1))

    asegurar_modelos_base()
    redes_alt = preparar_arquitectura_alternativa()

    total_corridas = len(ESCENARIOS) * len(CONFIGURACIONES) * len(SEMILLAS)
    print(
        f"\nIniciando {total_corridas} simulaciones "
        f"({len(ESCENARIOS)} escenarios × {len(CONFIGURACIONES)} configuraciones × "
        f"{len(SEMILLAS)} semillas, {FRAMES_POR_EPISODIO} frames máx./episodio)..."
    )

    registros = []
    contador = 0

    for escenario in ESCENARIOS:
        for config in CONFIGURACIONES:
            for semilla in SEMILLAS:
                contador += 1
                resultado = simular_episodio(
                    escenario, config, semilla, redes_alt
                )
                registros.append(resultado)
                print(
                    f"[{contador:02d}/{total_corridas}] {escenario:<11} | "
                    f"{config:<16} | Semilla {semilla:<2} -> "
                    f"Presas: {resultado['presas_vivas']:>2} | "
                    f"Alim: {resultado['alimento_consumido']:>2} | "
                    f"Capt: {resultado['capturas_depredadores']:>2}"
                )

    guardar_csv_crudo(registros)
    generar_y_mostrar_resumen(registros)
    pygame.quit()


if __name__ == "__main__":
    main()