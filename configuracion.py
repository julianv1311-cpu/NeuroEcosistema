
# Configuración general del NeuroEcosistema

import pygame


def _obtener_dimensiones():
    if not pygame.display.get_init():
        return 500, 500

    info = pygame.display.Info()
    ancho = max(500, int(info.current_w * 0.9))
    alto = max(300, int(info.current_h * 0.9))
    return ancho, alto


ANCHO, ALTO = _obtener_dimensiones()

FPS = 60

CANTIDAD_PRESAS = 15
CANTIDAD_DEPREDADORES = 4
CANTIDAD_PLANTAS = 30

COLOR_FONDO = (35, 45, 35)
COLOR_PRESA = (80, 180, 255)
COLOR_DEPREDADOR = (230, 75, 75)
COLOR_PLANTA = (70, 210, 90)
COLOR_TEXTO = (255, 255, 255)

ESCENARIO_A = "RECURSOS"
ESCENARIO_B = "ESCASEZ"
ESCENARIO_C = "DEPREDACION"
