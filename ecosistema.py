
import pygame
import random
import math
from pathlib import Path


from configuracion import *
from red_neuronal import RedNeuronal
from recursos import cargar_recursos


# Modos de experimentación
MODO_ENTRENADO = "ENTRENADO"
MODO_ALEATORIO = "ALEATORIO"
MODO_REGLAS = "REGLAS"



class Planta:

    def __init__(self, imagen):
        self.x = random.randint(20, ANCHO - 20)
        self.y = random.randint(20, ALTO - 20)
        self.radio = 6
        self.imagen = imagen

    def dibujar(self, pantalla):
        pantalla.blit(
            self.imagen,
            (
                int(self.x - self.imagen.get_width() / 2),
                int(self.y - self.imagen.get_height() / 2)
            )
        )




class Presa:

    def __init__(self, red, imagen):
        self.x = random.randint(20, ANCHO - 20)
        self.y = random.randint(20, ALTO - 20)
        self.energia = 100
        self.velocidad = 2
        self.radio = 9
        self.viva = True
        self.red = red
        self.modo = MODO_ENTRENADO
        self.imagen = imagen
        self.obstaculos = []

    def dibujar(self, pantalla):
        if self.viva:
            pantalla.blit(
                self.imagen,
                (
                    int(self.x - self.imagen.get_width() / 2),
                    int(self.y - self.imagen.get_height() / 2)
                )
            )

    def mover_seguro(self, dx, dy):
        nuevo_x = self.x + dx
        nuevo_y = self.y + dy

        # Comprobar primero el movimiento horizontal
        if not posicion_bloqueada(
            nuevo_x, self.y, self.radio, self.obstaculos
        ):
            self.x = nuevo_x

        # Comprobar después el movimiento vertical
        if not posicion_bloqueada(
            self.x, nuevo_y, self.radio, self.obstaculos
        ):
            self.y = nuevo_y

    def mover(self, plantas, depredadores):
        if not self.viva:
            return

        if self.modo == MODO_ALEATORIO:
            accion = random.randint(0, 4)
            self.aplicar_movimiento(accion)

        elif self.modo == MODO_REGLAS:
            self.mover_por_reglas(plantas, depredadores)

        else:
            self.mover_con_red(plantas, depredadores)

        self.x = max(10, min(ANCHO - 10, self.x))
        self.y = max(10, min(ALTO - 10, self.y))

        self.energia -= 0.03

    def mover_con_red(self, plantas, depredadores):
        planta = min(
            plantas,
            key=lambda p: math.hypot(p.x - self.x, p.y - self.y),
            default=None
        )

        depredador = min(
            [d for d in depredadores if d.vivo],
            key=lambda d: math.hypot(d.x - self.x, d.y - self.y),
            default=None
        )

        dx_planta = 0
        dy_planta = 0
        distancia_planta = 1.0

        dx_peligro = 0
        dy_peligro = 0
        distancia_peligro = 1.0

        diagonal = math.hypot(ANCHO, ALTO)

        if planta:
            dx = planta.x - self.x
            dy = planta.y - self.y
            distancia = math.hypot(dx, dy)

            if distancia > 0:
                dx_planta = dx / distancia
                dy_planta = dy / distancia

            distancia_planta = min(1.0, distancia / diagonal)

        if depredador:
            dx = depredador.x - self.x
            dy = depredador.y - self.y
            distancia = math.hypot(dx, dy)

            if distancia > 0:
                dx_peligro = dx / distancia
                dy_peligro = dy / distancia

            distancia_peligro = min(1.0, distancia / diagonal)

        datos = [
            dx_planta,
            dy_planta,
            dx_peligro,
            dy_peligro,
            distancia_planta,
            distancia_peligro,
            self.energia / 100,
            self.x / ANCHO,
            self.y / ALTO,
            1.0
        ]

        accion = self.red.decidir(datos)
        self.aplicar_movimiento(accion)

    def mover_por_reglas(self, plantas, depredadores):
        dx = 0
        dy = 0

        # Buscar el depredador vivo más cercano
        peligro = min(
            [d for d in depredadores if d.vivo],
            key=lambda d: math.hypot(d.x - self.x, d.y - self.y),
            default=None
        )

        if peligro:
            distancia = math.hypot(
                peligro.x - self.x,
                peligro.y - self.y
            )

            if 0 < distancia < 100:
                dx = self.x - peligro.x
                dy = self.y - peligro.y

                self.mover_en_direccion(dx, dy)
                return

        # Si no hay peligro cerca, buscar plantas
        planta = min(
            plantas,
            key=lambda p: math.hypot(p.x - self.x, p.y - self.y),
            default=None
        )

        if planta:
            dx = planta.x - self.x
            dy = planta.y - self.y
            self.mover_en_direccion(dx, dy)

    def mover_en_direccion(self, dx, dy):
        distancia = math.hypot(dx, dy)

        if distancia > 0:
            dx = (dx / distancia) * self.velocidad
            dy = (dy / distancia) * self.velocidad

            self.mover_seguro(dx, dy)

    def aplicar_movimiento(self, accion):
        if accion == 0:  # Arriba
            self.mover_seguro(0, -self.velocidad)

        elif accion == 1:  # Abajo
            self.mover_seguro(0, self.velocidad)

        elif accion == 2:  # Izquierda
            self.mover_seguro(-self.velocidad, 0)

        elif accion == 3:  # Derecha
            self.mover_seguro(self.velocidad, 0)

        elif accion == 4:  # Quieto
            pass

    def comer(self, plantas):
        if not self.viva:
            return

        for planta in plantas[:]:
            distancia = math.hypot(
                planta.x - self.x,
                planta.y - self.y
            )

            if distancia < 15:
                plantas.remove(planta)
                self.energia = min(100, self.energia + 25)

        if self.energia <= 0:
            self.viva = False



class Depredador:
    def __init__(self, red, imagen):
        self.x = random.randint(20, ANCHO - 20)
        self.y = random.randint(20, ALTO - 20)
        self.energia = 100
        self.velocidad = 1.5
        self.radio = 12
        self.vivo = True
        self.red = red
        self.modo = MODO_ENTRENADO
        self.imagen = imagen
        self.obstaculos = []  # Añadido para evitar errores con los obstáculos

    def dibujar(self, pantalla):
        if self.vivo:
            pantalla.blit(
                self.imagen,
                (
                    int(self.x - self.imagen.get_width() / 2),
                    int(self.y - self.imagen.get_height() / 2)
                )
            )
    
    def mover_seguro(self, dx, dy):
        nuevo_x = self.x + dx
        nuevo_y = self.y + dy

        if not posicion_bloqueada(
            nuevo_x, self.y, self.radio, self.obstaculos
        ):
            self.x = nuevo_x

        if not posicion_bloqueada(
            self.x, nuevo_y, self.radio, self.obstaculos
        ):
            self.y = nuevo_y

    def mover(self, presas):
        if not self.vivo:
            return

        objetivo = min(
            [p for p in presas if p.viva],
            key=lambda p: math.hypot(p.x - self.x, p.y - self.y),
            default=None
        )

        if self.modo == MODO_ALEATORIO:
            accion = random.randint(0, 4)
            self.aplicar_movimiento(accion)

        elif self.modo == MODO_REGLAS:
            if objetivo:
                dx = objetivo.x - self.x
                dy = objetivo.y - self.y
                self.mover_en_direccion(dx, dy)

        else:
            if objetivo:
                dx = objetivo.x - self.x
                dy = objetivo.y - self.y
                distancia = math.hypot(dx, dy)

                dx_presa = 0
                dy_presa = 0

                if distancia > 0:
                    dx_presa = dx / distancia
                    dy_presa = dy / distancia

                diagonal = math.hypot(ANCHO, ALTO)
                distancia_presa = min(1.0, distancia / diagonal)

                datos = [
                    dx_presa,
                    dy_presa,
                    0.0,
                    0.0,
                    distancia_presa,
                    1.0,
                    self.energia / 100,
                    self.x / ANCHO,
                    self.y / ALTO,
                    1.0
                ]

                accion = self.red.decidir(datos)
                self.aplicar_movimiento(accion)

        self.x = max(10, min(ANCHO - 10, self.x))
        self.y = max(10, min(ALTO - 10, self.y))
        self.energia -= 0.04

    def mover_en_direccion(self, dx, dy):
        distancia = math.hypot(dx, dy)
        if distancia > 0:
            dx = (dx / distancia) * self.velocidad
            dy = (dy / distancia) * self.velocidad
            self.mover_seguro(dx, dy)

    def aplicar_movimiento(self, accion):
        if accion == 0:
            self.mover_seguro(0, -self.velocidad)
        elif accion == 1:
            self.mover_seguro(0, self.velocidad)
        elif accion == 2:
            self.mover_seguro(-self.velocidad, 0)
        elif accion == 3:
            self.mover_seguro(self.velocidad, 0)
        elif accion == 4:
            pass  # Quieto

    def cazar(self, presas):
        if not self.vivo:
            return

        for presa in presas:
            distancia = math.hypot(
                presa.x - self.x,
                presa.y - self.y
            )

            if presa.viva and distancia < 18:
                presa.viva = False
                self.energia = min(100, self.energia + 30)

        if self.energia <= 0:
            self.vivo = False


class Obstaculo:

    def __init__(self, x, y, imagen, tipo):
        self.x = x
        self.y = y
        self.imagen = imagen
        self.tipo = tipo

        # Rectangulo visual completo
        self.area_visual = pygame.Rect(
            x, y,
            imagen.get_width(),
            imagen.get_height()
        )

        # Zona que realmente bloquea el movimiento
        if tipo == "arbol":
            self.rect = pygame.Rect(x + 20, y + 35, 24, 22)

        elif tipo == "roca":
            self.rect = pygame.Rect(x + 8, y + 12, 32, 28)

        elif tipo == "agua":
            self.rect = pygame.Rect(x + 4, y + 4, 56, 56)

    def dibujar(self, pantalla):
        pantalla.blit(self.imagen, (self.x, self.y))

def posicion_bloqueada(x, y, radio, obstaculos):

    zona = pygame.Rect(
        int(x - radio),
        int(y - radio),
        radio * 2,
        radio * 2
    )

    for obstaculo in obstaculos:
        if zona.colliderect(obstaculo.rect):
            return True

    return False





class Ecosistema:

    def __init__(self, semilla=42, escenario=ESCENARIO_A):

        # Mantener las mismas condiciones iniciales
        random.seed(semilla)

        self.escenario = escenario
        self.semilla = semilla

        # Configurar cantidades según el escenario
        if escenario == ESCENARIO_A:
            cantidad_plantas = 30
            cantidad_depredadores = 2

        elif escenario == ESCENARIO_B:
            cantidad_plantas = 8
            cantidad_depredadores = 4

        elif escenario == ESCENARIO_C:
            cantidad_plantas = 30
            cantidad_depredadores = 8

        else:
            raise ValueError("Escenario no válido")

        # Buscar los archivos de las redes neuronales
        carpeta = Path(__file__).resolve().parent

        archivo_presas = carpeta / "red_presas.npz"
        archivo_depredadores = carpeta / "red_depredadores.npz"

        if not archivo_presas.exists():
            raise FileNotFoundError(
                f"No se encontró: {archivo_presas}"
            )

        if not archivo_depredadores.exists():
            raise FileNotFoundError(
                f"No se encontró: {archivo_depredadores}"
            )

        # Cargar las redes neuronales
        self.red_presas = RedNeuronal()
        self.red_presas.cargar(archivo_presas)

        self.red_depredadores = RedNeuronal()
        self.red_depredadores.cargar(archivo_depredadores)

        # Cargar las imágenes
        self.recursos = cargar_recursos()
        # Crear los obstáculos del bosque
        self.obstaculos = self.generar_obstaculos()

        self.modo = MODO_ENTRENADO

        # Crear plantas
        self.plantas = [
            Planta(self.recursos["planta"])
            for _ in range(cantidad_plantas)
        ]

        # Crear presas
        self.presas = [
            Presa(self.red_presas, self.recursos["presa"])
            for _ in range(CANTIDAD_PRESAS)
        ]

        # Crear depredadores
        self.depredadores = [
            Depredador(
                self.red_depredadores,
                self.recursos["depredador"]
            )
            for _ in range(cantidad_depredadores)
        ]

    # Asignar obstáculos a las presas
        for presa in self.presas:
            presa.obstaculos = self.obstaculos

# Asignar obstáculos a los depredadores
        for depredador in self.depredadores:
            depredador.obstaculos = self.obstaculos

# Evitar que las entidades aparezcan dentro de obstáculos
        self.corregir_posiciones()
        self.cambiar_modo(MODO_ENTRENADO)
    
    def generar_obstaculos(self):

        obstaculos = []

        elementos = (
            [("arbol", 12)] +
            [("roca", 8)] +
            [("agua", 3)]
        )

        for tipo, cantidad in elementos:

            imagen = self.recursos[tipo]

            for _ in range(cantidad):

                colocado = False

                for intento in range(100):

                    x = random.randint(10, ANCHO - imagen.get_width() - 10)
                    y = random.randint(170, ALTO - imagen.get_height() - 10)

                    nueva_area = pygame.Rect(
                        x, y,
                        imagen.get_width(),
                        imagen.get_height()
                    )

                    # Evitar que los obstáculos se superpongan
                    ocupado = any(
                        nueva_area.colliderect(o.area_visual.inflate(8, 8))
                        for o in obstaculos
                    )

                    if not ocupado:
                        obstaculos.append(
                            Obstaculo(x, y, imagen, tipo)
                        )
                        colocado = True
                        break

                if not colocado:
                    print(f"No se pudo colocar un obstáculo: {tipo}")

        return obstaculos

    def corregir_posiciones(self):

        entidades = (
            [(p, p.radio) for p in self.plantas] +
            [(p, p.radio) for p in self.presas] +
            [(d, d.radio) for d in self.depredadores]
        )

        for entidad, radio in entidades:

            intentos = 0

            while posicion_bloqueada(
                entidad.x, entidad.y, radio, self.obstaculos
            ) and intentos < 200:

                entidad.x = random.randint(20, ANCHO - 20)
                entidad.y = random.randint(180, ALTO - 20)

                intentos += 1
    

    def cambiar_modo(self, nuevo_modo):

        self.modo = nuevo_modo

        for presa in self.presas:
            presa.modo = nuevo_modo

        for depredador in self.depredadores:
            depredador.modo = nuevo_modo

    def actualizar(self):

        for presa in self.presas:
            presa.mover(self.plantas, self.depredadores)
            presa.comer(self.plantas)

        for depredador in self.depredadores:
            depredador.mover(self.presas)
            depredador.cazar(self.presas)

    
    def dibujar(self, pantalla):

        # Dibujar primero los obstáculos
        for obstaculo in self.obstaculos:
            obstaculo.dibujar(pantalla)

        # Dibujar las plantas
        for planta in self.plantas:
            planta.dibujar(pantalla)

        # Dibujar las presas
        for presa in self.presas:
            presa.dibujar(pantalla)

        # Dibujar los depredadores
        for depredador in self.depredadores:
            depredador.dibujar(pantalla)


