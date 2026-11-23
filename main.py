import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pygame

pygame.init()

from configuracion import *
from ecosistema import (
    Ecosistema,
    MODO_ALEATORIO,
    MODO_ENTRENADO,
    MODO_REGLAS
)
from entrenamiento import (
    ARCHIVO_DEPREDADORES,
    ARCHIVO_PRESAS,
    cargar_historial,
    entrenar_redes
)

PASO_EPOCAS = 100


class VentanaEntrenamiento:
    """Ventana independiente de Matplotlib para métricas y gráficas."""

    def __init__(self, historial):
        plt.ion()
        self.fig = plt.figure("NeuroEcosistema - Entrenamiento", figsize=(9.5, 6.2))
        self.cerrada = False

        # Conectar evento de cierre para no romper la simulación de Pygame
        self.fig.canvas.mpl_connect("close_event", self._al_cerrar)

        grid = self.fig.add_gridspec(3, 2, height_ratios=[0.85, 1.3, 1.3], hspace=0.45, wspace=0.28)
        self.ax_info = self.fig.add_subplot(grid[0, :])
        self.ax_perdida_p = self.fig.add_subplot(grid[1, 0])
        self.ax_exactitud_p = self.fig.add_subplot(grid[1, 1])
        self.ax_perdida_d = self.fig.add_subplot(grid[2, 0])
        self.ax_exactitud_d = self.fig.add_subplot(grid[2, 1])

        self.actualizar(historial)

    def _al_cerrar(self, _evento):
        self.cerrada = True

    def reabrir_si_cerrada(self, historial):
        if self.cerrada or not plt.fignum_exists(self.fig.number):
            self.cerrada = False
            self.__init__(historial)

    def actualizar(self, historial):
        if self.cerrada or not plt.fignum_exists(self.fig.number):
            return

        self.ax_info.clear()
        self.ax_perdida_p.clear()
        self.ax_exactitud_p.clear()
        self.ax_perdida_d.clear()
        self.ax_exactitud_d.clear()

        self.ax_info.axis("off")

        if not historial:
            self.ax_info.text(
                0.02, 0.5,
                "Sin historial guardado aún. Pulsa [T] en la ventana del bosque para entrenar +100 épocas.",
                fontsize=10, va="center"
            )
            self._refrescar_canvas()
            return

        hp = historial["presas"]
        hd = historial["depredadores"]

        epocas_p = hp.get("epocas_totales", len(hp["perdida"]))
        epocas_d = hd.get("epocas_totales", len(hd["perdida"]))
        iter_p = hp["iteraciones"][-1] if hp["iteraciones"] else 0
        iter_d = hd["iteraciones"][-1] if hd["iteraciones"] else 0

        perd_p = hp["perdida"][-1] if hp["perdida"] else 0.0
        acc_p = hp["exactitud"][-1] if hp["exactitud"] else 0.0
        perd_d = hd["perdida"][-1] if hd["perdida"] else 0.0
        acc_d = hd["exactitud"][-1] if hd["exactitud"] else 0.0

        resumen = (
            f"PRESAS  -> Épocas: {epocas_p} | Iteraciones: {iter_p} | "
            f"Pérdida: {perd_p:.4f} | Exactitud: {acc_p:.2f}%\n"
            f"DEPRED. -> Épocas: {epocas_d} | Iteraciones: {iter_d} | "
            f"Pérdida: {perd_d:.4f} | Exactitud: {acc_d:.2f}%"
        )
        self.ax_info.text(0.01, 0.5, resumen, fontsize=10, fontweight="bold", va="center")

        # Gráficas Presas
        epocas_eje_p = list(range(1, len(hp["perdida"]) + 1))
        self.ax_perdida_p.plot(epocas_eje_p, hp["perdida"], label="Entreno")
        if hp.get("perdida_validacion"):
            self.ax_perdida_p.plot(epocas_eje_p, hp["perdida_validacion"], label="Validación", linestyle="--")
        self.ax_perdida_p.set_title("Presas - Pérdida")
        self.ax_perdida_p.set_xlabel("Época")
        self.ax_perdida_p.legend(fontsize=8)
        self.ax_perdida_p.grid(True, alpha=0.3)

        self.ax_exactitud_p.plot(epocas_eje_p, hp["exactitud"], label="Entreno", color="tab:green")
        if hp.get("exactitud_validacion"):
            self.ax_exactitud_p.plot(epocas_eje_p, hp["exactitud_validacion"], label="Validación", linestyle="--", color="tab:olive")
        self.ax_exactitud_p.set_title("Presas - Exactitud (%)")
        self.ax_exactitud_p.set_xlabel("Época")
        self.ax_exactitud_p.legend(fontsize=8)
        self.ax_exactitud_p.grid(True, alpha=0.3)

        # Gráficas Depredadores
        epocas_eje_d = list(range(1, len(hd["perdida"]) + 1))
        self.ax_perdida_d.plot(epocas_eje_d, hd["perdida"], label="Entreno", color="tab:red")
        if hd.get("perdida_validacion"):
            self.ax_perdida_d.plot(epocas_eje_d, hd["perdida_validacion"], label="Validación", linestyle="--", color="tab:orange")
        self.ax_perdida_d.set_title("Depredadores - Pérdida")
        self.ax_perdida_d.set_xlabel("Época")
        self.ax_perdida_d.legend(fontsize=8)
        self.ax_perdida_d.grid(True, alpha=0.3)

        self.ax_exactitud_d.plot(epocas_eje_d, hd["exactitud"], label="Entreno", color="tab:purple")
        if hd.get("exactitud_validacion"):
            self.ax_exactitud_d.plot(epocas_eje_d, hd["exactitud_validacion"], label="Validación", linestyle="--", color="tab:pink")
        self.ax_exactitud_d.set_title("Depredadores - Exactitud (%)")
        self.ax_exactitud_d.set_xlabel("Época")
        self.ax_exactitud_d.legend(fontsize=8)
        self.ax_exactitud_d.grid(True, alpha=0.3)

        self._refrescar_canvas()

    def _refrescar_canvas(self):
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def bombear_eventos(self):
        if not self.cerrada and plt.fignum_exists(self.fig.number):
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass


pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("NeuroEcosistema")

reloj = pygame.time.Clock()

# Si no existen modelos previos, entrenar el primer bloque de 100 épocas
if not ARCHIVO_PRESAS.exists() or not ARCHIVO_DEPREDADORES.exists():
    print("No se encontraron redes entrenadas. Iniciando primer bloque de 100 épocas...")
    entrenar_redes(epocas=PASO_EPOCAS, continuar=False)

nombres_modos = {
    MODO_ENTRENADO: "REDES NEURONALES ENTRENADAS",
    MODO_ALEATORIO: "MOVIMIENTO ALEATORIO",
    MODO_REGLAS: "MOVIMIENTO BASADO EN REGLAS"
}

nombres_escenarios = {
    ESCENARIO_A: "RECURSOS SUFICIENTES",
    ESCENARIO_B: "ESCASEZ DE ALIMENTO",
    ESCENARIO_C: "MAYOR PRESION DE DEPREDACION"
}

escenario_actual = ESCENARIO_A
semilla_actual = 42

ecosistema = Ecosistema(
    semilla=semilla_actual,
    escenario=escenario_actual
)

ventana_entrenamiento = VentanaEntrenamiento(cargar_historial())

# --------------------------------
# FONDO DEL BOSQUE
# --------------------------------

imagen_pasto = ecosistema.recursos["pasto"]


def dibujar_fondo(pantalla):
    ancho = imagen_pasto.get_width()
    alto = imagen_pasto.get_height()

    for x in range(0, ANCHO, ancho):
        for y in range(0, ALTO, alto):
            pantalla.blit(imagen_pasto, (x, y))


def refrescar_redes_en_ecosistema(ecosistema_obj):
    ecosistema_obj.red_presas.cargar(ARCHIVO_PRESAS)
    ecosistema_obj.red_depredadores.cargar(ARCHIVO_DEPREDADORES)
    for presa in ecosistema_obj.presas:
        presa.red = ecosistema_obj.red_presas
    for depredador in ecosistema_obj.depredadores:
        depredador.red = ecosistema_obj.red_depredadores


fuente = pygame.font.SysFont("Arial", 14)
titulo = pygame.font.SysFont("Arial", 21, bold=True)

ejecutando = True
tiempo_inicio = pygame.time.get_ticks()


# --------------------------------
# FUNCION PARA GUARDAR RESULTADOS
# --------------------------------

def guardar_resultados():
    archivo = (
        Path(__file__).resolve().parent
        / "resultados_experimentos.csv"
    )

    presas_vivas = sum(
        1 for p in ecosistema.presas if p.viva
    )
    presas_muertas = len(ecosistema.presas) - presas_vivas

    depredadores_vivos = sum(
        1 for d in ecosistema.depredadores if d.vivo
    )
    depredadores_muertos = (
        len(ecosistema.depredadores) - depredadores_vivos
    )

    plantas_disponibles = len(ecosistema.plantas)

    tiempo = (pygame.time.get_ticks() - tiempo_inicio) // 1000
    minutos = tiempo // 60
    segundos = tiempo % 60

    existe_archivo = archivo.exists()

    with open(
        archivo,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        escritor = csv.writer(f, delimiter=";")

        if not existe_archivo:
            escritor.writerow([
                "Modo",
                "Escenario",
                "Semilla",
                "Tiempo",
                "Presas vivas",
                "Presas muertas",
                "Depredadores vivos",
                "Depredadores muertos",
                "Plantas disponibles"
            ])

        escritor.writerow([
            nombres_modos[ecosistema.modo],
            nombres_escenarios[ecosistema.escenario],
            semilla_actual,
            f"{minutos:02d}:{segundos:02d}",
            presas_vivas,
            presas_muertas,
            depredadores_vivos,
            depredadores_muertos,
            plantas_disponibles
        ])

    print(f"Resultados guardados en: {archivo}")


# --------------------------------
# BUCLE PRINCIPAL
# --------------------------------

while ejecutando:

    for evento in pygame.event.get():

        if evento.type == pygame.QUIT:
            ejecutando = False

        if evento.type == pygame.KEYDOWN:

            if evento.key == pygame.K_ESCAPE:
                ejecutando = False

            elif evento.key == pygame.K_1:
                ecosistema.cambiar_modo(MODO_ENTRENADO)

            elif evento.key == pygame.K_2:
                ecosistema.cambiar_modo(MODO_ALEATORIO)

            elif evento.key == pygame.K_3:
                ecosistema.cambiar_modo(MODO_REGLAS)

            elif evento.key == pygame.K_F1:
                modo_actual = ecosistema.modo
                escenario_actual = ESCENARIO_A
                ecosistema = Ecosistema(semilla=semilla_actual, escenario=escenario_actual)
                ecosistema.cambiar_modo(modo_actual)
                tiempo_inicio = pygame.time.get_ticks()

            elif evento.key == pygame.K_F2:
                modo_actual = ecosistema.modo
                escenario_actual = ESCENARIO_B
                ecosistema = Ecosistema(semilla=semilla_actual, escenario=escenario_actual)
                ecosistema.cambiar_modo(modo_actual)
                tiempo_inicio = pygame.time.get_ticks()

            elif evento.key == pygame.K_F3:
                modo_actual = ecosistema.modo
                escenario_actual = ESCENARIO_C
                ecosistema = Ecosistema(semilla=semilla_actual, escenario=escenario_actual)
                ecosistema.cambiar_modo(modo_actual)
                tiempo_inicio = pygame.time.get_ticks()

            elif evento.key == pygame.K_r:
                modo_actual = ecosistema.modo
                ecosistema = Ecosistema(semilla=semilla_actual, escenario=escenario_actual)
                ecosistema.cambiar_modo(modo_actual)
                tiempo_inicio = pygame.time.get_ticks()

            elif evento.key == pygame.K_s:
                guardar_resultados()

            # Sumar +100 épocas y actualizar segunda ventana + redes activas
            elif evento.key == pygame.K_t:
                _, _, hist_actualizado = entrenar_redes(
                    epocas=PASO_EPOCAS,
                    continuar=True
                )
                refrescar_redes_en_ecosistema(ecosistema)
                ventana_entrenamiento.reabrir_si_cerrada(hist_actualizado)
                ventana_entrenamiento.actualizar(hist_actualizado)

    # --------------------------------
    # ACTUALIZAR SIMULACION
    # --------------------------------

    ecosistema.actualizar()
    ventana_entrenamiento.bombear_eventos()

    presas_vivas = sum(1 for p in ecosistema.presas if p.viva)
    presas_muertas = len(ecosistema.presas) - presas_vivas

    depredadores_vivos = sum(1 for d in ecosistema.depredadores if d.vivo)
    depredadores_muertos = len(ecosistema.depredadores) - depredadores_vivos

    plantas_disponibles = len(ecosistema.plantas)

    tiempo = (pygame.time.get_ticks() - tiempo_inicio) // 1000
    minutos = tiempo // 60
    segundos = tiempo % 60

    # --------------------------------
    # DIBUJAR PANTALLA
    # --------------------------------

    dibujar_fondo(pantalla)
    ecosistema.dibujar(pantalla)

    panel_ancho, panel_alto = 950, 155
    panel_x, panel_y = 10, 10

    superficie_panel = pygame.Surface((panel_ancho, panel_alto), pygame.SRCALPHA)
    color_panel_rgba = (20, 20, 20, 150)

    pygame.draw.rect(
        superficie_panel,
        color_panel_rgba,
        (0, 0, panel_ancho, panel_alto),
        border_radius=10
    )

    pygame.draw.rect(
        superficie_panel,
        (255, 255, 255, 40),
        (0, 0, panel_ancho, panel_alto),
        width=1,
        border_radius=10
    )

    pantalla.blit(superficie_panel, (panel_x, panel_y))

    pantalla.blit(
        titulo.render("NEUROECOSISTEMA", True, COLOR_TEXTO),
        (25, 15)
    )

    columna_izquierda = [
        f"Modo: {nombres_modos[ecosistema.modo]}",
        f"Escenario: {nombres_escenarios[ecosistema.escenario]}",
        f"Semilla: {semilla_actual}",
        f"Tiempo: {minutos:02d}:{segundos:02d}",
    ]

    columna_derecha = [
        f"Presas: {presas_vivas} vivas | {presas_muertas} muertas",
        f"Depredadores: {depredadores_vivos} vivos | {depredadores_muertos} muertos",
        f"Plantas disponibles: {plantas_disponibles}",
    ]

    for i, linea in enumerate(columna_izquierda):
        texto = fuente.render(linea, True, COLOR_TEXTO)
        pantalla.blit(texto, (25, 48 + i * 20))

    for i, linea in enumerate(columna_derecha):
        texto = fuente.render(linea, True, COLOR_TEXTO)
        pantalla.blit(texto, (490, 48 + i * 20))

    controles = "-Modos: 1-Entrenado | 2-Aleatorio | 3-Reglas | T-+100 Epocas | "
    
    escenarios = "   Escenarios: F1-Recursos | F2-Escasez | F3-Depredacion         Reiniciar: R" 

    pantalla.blit(
        fuente.render(controles, True, COLOR_TEXTO),
        (25, 128)
    )
    
    pantalla.blit(
        fuente.render(escenarios, True, COLOR_TEXTO),
        (430, 128) 
    )
    

    pygame.display.flip()

    reloj.tick(FPS)

plt.close("all")
pygame.quit()