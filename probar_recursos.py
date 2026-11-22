
import pygame
from recursos import cargar_recursos

pygame.init()

pantalla = pygame.display.set_mode((500, 400))
recursos = cargar_recursos()

pantalla.fill((30, 30, 30))

pantalla.blit(recursos["presa"], (30, 30))
pantalla.blit(recursos["depredador"], (100, 30))
pantalla.blit(recursos["planta"], (170, 30))
pantalla.blit(recursos["arbol"], (230, 30))
pantalla.blit(recursos["roca"], (310, 30))
pantalla.blit(recursos["agua"], (30, 150))
pantalla.blit(recursos["pasto"], (110, 150))

pygame.display.flip()

ejecutando = True

while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

pygame.quit()
