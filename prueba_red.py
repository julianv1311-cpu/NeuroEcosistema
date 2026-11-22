
import numpy as np
from red_neuronal import RedNeuronal

red = RedNeuronal()

datos = np.random.rand(10)

resultado = red.predecir(datos)

print("Probabilidades:", resultado)
print("Acción seleccionada:", red.decidir(datos))
