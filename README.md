NeuroEcosistema – Depredadores y PresasProyecto Final Integrador para el espacio académico Electiva I de Profundización – Inteligencia ArtificialUniversidad Santo Tomás – Seccional Tunja | Facultad de Ingeniería de SistemasDocente: Diego Alejandro Vela BeltránDescripción GeneralNeuroEcosistema es una simulación 2D multiagente desarrollada en Python donde dos poblaciones autónomas —presas y depredadores— toman decisiones de movimiento mediante redes neuronales artificiales multicapa implementadas desde cero en NumPy.Presas: buscan plantas para recuperar energía y evitan a los depredadores cercanos.Depredadores: persiguen y capturan presas para evitar morir por agotamiento energético.Entorno: mapa de 1000 × 650 píxeles a 60 FPS con 23 obstáculos físicos (árboles, rocas y cuerpos de agua) y tres escenarios experimentales configurables.Pregunta orientadora¿En qué condiciones los agentes controlados por redes neuronales desarrollan comportamientos de búsqueda, persecución y evasión que mejoran su desempeño frente a estrategias sin aprendizaje?Arquitectura de las Redes NeuronalesCada especie utiliza un modelo neuronal independiente (red_presas.npz y red_depredadores.npz) con arquitectura 10–32–16–5 (965 parámetros entrenables por especie):CapaNeuronasActivaciónParámetrosEntrada10Identidad (vector normalizado)—Oculta 132ReLU352Oculta 216ReLU528Salida5Softmax85Entradas (10): dirección unitaria y distancia normalizada a la planta/presa más cercana, dirección y distancia al depredador más cercano, energía normalizada, posición (x, y) normalizada y término constante de sesgo.Salidas (5 acciones discretas): 0 = Arriba, 1 = Abajo, 2 = Izquierda, 3 = Derecha, 4 = Quieto.Entrenamiento: aprendizaje supervisado por descenso de gradiente estocástico en mini-lotes (batch_size = 64, tasa = 0.03, entropía cruzada categórica) sobre 3.000 ejemplos sintéticos por especie (división 80 % entrenamiento / 20 % validación).Estructura del RepositorioNeuroEcosistema/
├── assets/                     # Sprites 2D (animales, alimentos y obstáculos)
├── configuracion.py            # Constantes globales, dimensiones y escenarios
├── ecosistema.py               # Entidades, físicas, colisiones y modos de control
├── red_neuronal.py             # MLP multicapa desde cero en NumPy
├── entrenamiento.py            # Generación de datos, entrenamiento y guardado de historial
├── recursos.py                 # Carga y escalado de imágenes con Pygame
├── main.py                     # Simulación interactiva + monitor en Matplotlib
├── ejecutar_experimentos.py    # Evaluación automatizada reproducible (semillas y escenarios)
├── prueba_red.py               # Verificación unitaria de la red neuronal
├── probar_recursos.py          # Verificación visual de recursos gráficos
├── red_presas.npz              # Pesos entrenados de la red de presas
├── red_depredadores.npz        # Pesos entrenados de la red de depredadores
├── historial_entrenamiento.npz # Registro histórico de pérdida y exactitud
├── resultados_experimentos.csv # Datos crudos de las ejecuciones experimentales
└── requirements.txt            # Dependencias del proyecto
Requisitos e InstalaciónPython: 3.10 o superior (probado en Python 3.13).Dependencias principales: numpy, pygame, matplotlib.Clonar el repositorio:git clone <URL-DEL-REPOSITORIO>
cd NeuroEcosistema
(Opcional recomendado) Crear y activar un entorno virtual:python -m venv .venv
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux / macOS:
source .venv/bin/activate
Instalar dependencias:pip install -r requirements.txt
(Si requirements.txt no incluye todas las librerías, instala manualmente: pip install numpy pygame matplotlib.)Cómo Ejecutar el Proyecto1. Pruebas rápidas de verificaciónpython prueba_red.py
python probar_recursos.py
2. Reproducir el entrenamiento desde ceroGenera o sobrescribe red_presas.npz, red_depredadores.npz y historial_entrenamiento.npz usando semilla = 42:python entrenamiento.py
3. Ejecutar la simulación interactivaAbre la ventana del ecosistema en Pygame y la ventana de gráficas de entrenamiento en Matplotlib:python main.py
Controles de teclado en main.pyTeclaAcción1Cambiar a modo Redes Neuronales Entrenadas2Cambiar a modo Movimiento Aleatorio (línea base)3Cambiar a modo Movimiento Basado en Reglas (línea base)F1Cargar Escenario A (RECURSOS: 30 plantas, 2 depredadores)F2Cargar Escenario B (ESCASEZ: 8 plantas, 4 depredadores)F3Cargar Escenario C (DEPREDACION: 30 plantas, 8 depredadores)TEntrenar +100 épocas adicionales en vivo y actualizar gráficasRReiniciar el escenario actual con la misma semillaSGuardar instantánea de métricas en resultados_experimentos.csvESCSalir de la simulación4. Ejecutar la batería completa de experimentos (reproducible)Corre las 5 configuraciones (ALEATORIO, REGLAS, RED_SIN_ENTRENAR, RED_32_16 y RED_16_8) en los 3 escenarios con 5 semillas fijas ([42, 7, 13, 21, 99]):python ejecutar_experimentos.py
Al finalizar genera:resultados_experimentos.csv: registro fila por fila de las 75 simulaciones.resumen_experimentos.csv: tabla de medias ± desviación estándar por escenario y configuración.Escenarios ExperimentalesEscenarioPresasPlantasDepredadoresObstáculosPropósitoA (RECURSOS)1530223Condición favorable de forrajeo y baja amenazaB (ESCASEZ)158423Estrés por baja disponibilidad de alimentoC (DEPREDACION)1530823Alta presión de captura por depredadoresRecomendación de Limpieza del RepositorioAntes de realizar el commit final, asegúrate de incluir un archivo .gitignore con:__pycache__/
*.pyc
.venv/
y de verificar que requirements.txt liste numpy, pygame y matplotlib.