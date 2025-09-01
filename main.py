# Importamos las librerías que nos permitirán programar la red neuronal
import tensorflow as tf  # Para abreviar el nombre al programar, se redefine como tf
import numpy as np  # Esta gestiona arrays numéricos
import matplotlib.pyplot as plt  # Genera gráficos

# Ejemplos de x e y (10 puntos x y 10 puntos y conocidos)
valor_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
resultado = np.array([12, 17, 22, 27, 32, 37, 42, 47, 52, 57], dtype=float)  # Ecuación lineal: y = 5x + 7

# Crear el modelo secuencial
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Entrada: un solo valor (valor de x)
    tf.keras.layers.Dense(units=1)  # Capa densa con 1 neurona (valor de y)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),  # Usa un peso de 0.1, es decir, la potencia. 0.01 y 0.001 son más estables pero menos precisos
    loss='mean_squared_error'  # Esta función nos dará el error cuadrático medio
)

print("¡Bienvenido a la Red Neuronal de RAÚL SALAS!\n")  # A partir de este mensaje, comienza a funcionar la Red Neuronal con la configuración preestablecida
print("Entrenando la red........")  # Comienza el entrenamiento
historial = model.fit(valor_x, resultado, epochs=1000, verbose=False)  # Revisa 1000 veces con epochs el resultado, y durante estas revisiones va modificando y acercando más el valor a su resultado correcto
# 1000 epochs ≈ 1 min (resultado aproximado) | 10000 epochs ≈ 10 min (resultado exacto)
print("¡Red entrenada!")  # Indicamos que ya ha hecho las 1000 revisiones para mejorar el resultado

print("Vamos a averiguar el resultado")
# Redefinimos la variable resultado para la predicción
resultado_predict = model.predict(np.array([7]).reshape(-1, 1))  # Ponemos el valor de X en 'np.array'. Reshape con la librería numpy da una nueva forma a una matriz sin cambiar sus datos
# Usamos la nueva variable para imprimir el resultado
print("El resultado es " + str(resultado_predict) + " ")  # Finalmente, entregamos el valor de la y
print("El resultado redondeado es " + str(np.round(resultado_predict)))  # Redondeamos el resultado para que sea más entendible