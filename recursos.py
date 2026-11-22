
import pygame
from pathlib import Path

# Ubicación de la carpeta de recursos
CARPETA = Path(__file__).resolve().parent / "assets"


def cargar_imagen(ruta, tamaño):
    archivo = CARPETA / ruta

    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen: {archivo}"
        )

    imagen = pygame.image.load(str(archivo)).convert_alpha()

    return pygame.transform.scale(imagen, tamaño)


def cargar_recursos():

    recursos = {
        "presa": cargar_imagen(
            "animales/presa.png", (48, 48)
        ),

        "depredador": cargar_imagen(
            "animales/depredador.png", (48, 48)
        ),

        "planta": cargar_imagen(
            "alimentos/planta.png", (40, 40)
        ),

        "arbol": cargar_imagen(
            "entorno/arbol.png", (64, 64)
        ),

        "roca": cargar_imagen(
            "entorno/roca.png", (48, 48)
        ),

        "agua": cargar_imagen(
            "entorno/agua.png", (64, 64)
        ),

        "pasto": cargar_imagen(
            "entorno/pasto.png", (64, 64)
        )
    }

    return recursos
