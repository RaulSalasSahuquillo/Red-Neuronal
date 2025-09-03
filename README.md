# 🧠 Red Neuronal Simple en TensorFlow

Este proyecto implementa una **red neuronal básica** con **TensorFlow + Keras** para aprender la relación lineal entre valores de entrada (**x**) y resultados (**y**) según:

$$
y = 5x + 7
$$

---

## 📌 Descripción

Se entrena un modelo con un único **input** y una sola **neurona densa** para aproximar una función lineal a partir de **10 ejemplos** conocidos y luego se prueba con un valor nuevo (`x = 7`).

---

## 🚀 Tecnologías utilizadas

* [TensorFlow](https://www.tensorflow.org/) – Librería de machine learning
* [NumPy](https://numpy.org/) – Manejo de arrays numéricos
* [Matplotlib](https://matplotlib.org/) – Visualización de datos (opcional)

---

## 🛠️ Requisitos

* **Python** 3.10–3.11
* **TensorFlow** 2.15+ (o la compatible con tu versión de Python/SO)
* **NumPy** y **Matplotlib**

> 💡 Si usas Windows y tienes problemas de instalación de TensorFlow, prueba primero `pip install --upgrade pip` y asegúrate de que tu versión de Python sea compatible con tu versión de TensorFlow. En CPU suele bastar con `pip install tensorflow`.

---

## ▶️ Ejecución

1. **Clonar repositorio**

   ```bash
   git clone https://github.com/RaulSalasSahuquillo/Red-Neuronal.git
   cd Red-Neuronal
   ```

2. **Crear entorno (opcional, recomendado)**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Instalar dependencias**

   ```bash
   pip install --upgrade pip
   pip install tensorflow numpy matplotlib
   ```

4. **Ejecutar**

   ```bash
   python main.py
   ```

---

## 📂 Estructura del código

1. **Importación de librerías** → TensorFlow, NumPy y Matplotlib.
2. **Datos de entrenamiento**:

   * `valor_x`: entradas (1–10).
   * `resultado`: salidas (según `y = 5x + 7`).
3. **Definición del modelo**:

   * `Sequential` con 1 capa `Dense(units=1)`.
4. **Compilación**:

   * Optimizador: `Adam(learning_rate=0.1)`.
   * Pérdida: `mean_squared_error`.
5. **Entrenamiento**:

   * `epochs=1000` (ajustable).
6. **Predicción**:

   * Ejemplo con `x = 7`.

---

## 📊 Ejemplo de salida

```text
¡Bienvenido a la Red Neuronal de RAÚL SALAS!

Entrenando la red........
¡Red entrenada!
Vamos a averiguar el resultado
El resultado es [[42.000004]]
El resultado redondeado es [[42.]]
```

> Nota: El valor exacto esperado es `42`. El modelo converge muy cerca tras suficiente entrenamiento; el resultado puede variar ligeramente por inicialización aleatoria y tasa de aprendizaje.

---

## 📈 Visualización del entrenamiento (opcional)

Añade esto al final de `main.py` para ver cómo baja la pérdida:

```python
import matplotlib.pyplot as plt

plt.plot(historial.history['loss'])
plt.title('Evolución del error durante el entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('MSE (loss)')
plt.grid(True)
plt.show()
```

---

## 🧪 Código de referencia (extracto)

```python
import tensorflow as tf
import numpy as np

# Datos: y = 5x + 7
valor_x = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
resultado = np.array([12,17,22,27,32,37,42,47,52,57], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

print("¡Bienvenido a la Red Neuronal de RAÚL SALAS!\n")
print("Entrenando la red........")
historial = model.fit(valor_x, resultado, epochs=1000, verbose=False)
print("¡Red entrenada!")

print("Vamos a averiguar el resultado")
resultado_predict = model.predict(np.array([7], dtype=float))
print("El resultado es " + str(resultado_predict))
print("El resultado redondeado es " + str(np.round(resultado_predict)))
```

> Sugerencia: para resultados reproducibles, fija una semilla antes de crear el modelo:
> `tf.keras.utils.set_random_seed(42)`.

---

## 🤔 Problemas comunes

* **`ModuleNotFoundError: tensorflow.python`**
  Suele indicar incompatibilidad de versiones. Actualiza `pip`, comprueba la versión de Python y reinstala una versión de TensorFlow compatible con tu entorno (CPU/GPU y SO).

* **Tiempo de entrenamiento**
  Depende del hardware. Ajusta `epochs` o baja `learning_rate` si ves inestabilidad (p. ej., `0.01` o `0.001`).

---
