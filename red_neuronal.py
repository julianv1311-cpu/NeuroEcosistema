
import numpy as np


class RedNeuronal:

    def __init__(
        self,
        entradas=10,
        oculta1=32,
        oculta2=16,
        salidas=5
    ):
        self.W1 = np.random.randn(entradas, oculta1) * 0.1
        self.b1 = np.zeros(oculta1)

        self.W2 = np.random.randn(oculta1, oculta2) * 0.1
        self.b2 = np.zeros(oculta2)

        self.W3 = np.random.randn(oculta2, salidas) * 0.1
        self.b3 = np.zeros(salidas)

        self.historial = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predecir_lote(self, datos):
        z1 = datos @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        probabilidades = self.softmax(z3)

        return probabilidades

    def predecir(self, datos):
        datos = np.array(datos, dtype=float).reshape(1, -1)
        return self.predecir_lote(datos)[0]

    def decidir(self, datos):
        probabilidades = self.predecir(datos)
        return int(np.argmax(probabilidades))

    def cargar(self, archivo):
        """Carga los pesos de una red previamente entrenada."""
        datos = np.load(archivo)

        self.W1 = datos["W1"]
        self.b1 = datos["b1"]

        self.W2 = datos["W2"]
        self.b2 = datos["b2"]

        self.W3 = datos["W3"]
        self.b3 = datos["b3"]

        return self

    def guardar(self, archivo):
        """Guarda los pesos y sesgos de la red."""
        np.savez(
            archivo,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3
        )

    def calcular_perdida(self, probabilidades, y):
        cantidad = len(y)

        probabilidades_correctas = probabilidades[
            np.arange(cantidad), y
        ]

        return -np.mean(
            np.log(np.clip(probabilidades_correctas, 1e-9, 1.0))
        )

    def calcular_exactitud(self, probabilidades, y):
        predicciones = np.argmax(probabilidades, axis=1)
        return np.mean(predicciones == y) * 100

    def entrenar(
        self,
        X,
        y,
        epocas=100,
        tasa=0.01,
        batch_size=64,
        X_validacion=None,
        y_validacion=None
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        cantidad = len(X)

        historial = {
            "perdida": [],
            "exactitud": [],
            "perdida_validacion": [],
            "exactitud_validacion": [],
            "iteraciones": []
        }

        iteracion = 0

        for epoca in range(epocas):

            # Mezclar los ejemplos antes de cada época
            indices = np.random.permutation(cantidad)
            X_mezclado = X[indices]
            y_mezclado = y[indices]

            # Entrenamiento por mini-lotes
            for inicio in range(0, cantidad, batch_size):

                fin = inicio + batch_size

                X_lote = X_mezclado[inicio:fin]
                y_lote = y_mezclado[inicio:fin]

                n = len(X_lote)

                # Propagación hacia adelante
                z1 = X_lote @ self.W1 + self.b1
                a1 = self.relu(z1)

                z2 = a1 @ self.W2 + self.b2
                a2 = self.relu(z2)

                z3 = a2 @ self.W3 + self.b3
                probabilidades = self.softmax(z3)

                # Retropropagación
                dz3 = probabilidades.copy()
                dz3[np.arange(n), y_lote] -= 1
                dz3 /= n

                dW3 = a2.T @ dz3
                db3 = np.sum(dz3, axis=0)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * (z2 > 0)

                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)

                dW1 = X_lote.T @ dz1
                db1 = np.sum(dz1, axis=0)

                # Actualizar pesos y sesgos
                self.W3 -= tasa * dW3
                self.b3 -= tasa * db3

                self.W2 -= tasa * dW2
                self.b2 -= tasa * db2

                self.W1 -= tasa * dW1
                self.b1 -= tasa * db1

                iteracion += 1

            # Métricas al finalizar cada época
            predicciones = self.predecir_lote(X)

            perdida = self.calcular_perdida(predicciones, y)
            exactitud = self.calcular_exactitud(predicciones, y)

            historial["perdida"].append(float(perdida))
            historial["exactitud"].append(float(exactitud))
            historial["iteraciones"].append(iteracion)

            # Métricas de validación
            if X_validacion is not None and y_validacion is not None:
                pred_validacion = self.predecir_lote(X_validacion)

                perdida_val = self.calcular_perdida(
                    pred_validacion, y_validacion
                )

                exactitud_val = self.calcular_exactitud(
                    pred_validacion, y_validacion
                )

                historial["perdida_validacion"].append(
                    float(perdida_val)
                )

                historial["exactitud_validacion"].append(
                    float(exactitud_val)
                )

            # Mostrar progreso
            if (epoca + 1) % 20 == 0 or epoca == 0:
                print(
                    f"Época {epoca + 1}/{epocas} "
                    f"| Iteraciones: {iteracion} "
                    f"| Pérdida: {perdida:.4f} "
                    f"| Exactitud: {exactitud:.2f}%"
                )

        self.historial = historial

        return historial
