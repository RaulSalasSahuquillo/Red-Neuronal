# Import libraries that allow us to program the neural network
import tensorflow as tf  # Aliased as tf for convenience when programming
import numpy as np  # Manages numerical arrays
import matplotlib.pyplot as plt  # Generates plots

# Examples of x and y (10 known x points and 10 known y points)
valor_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
resultado = np.array([12, 17, 22, 27, 32, 37, 42, 47, 52, 57], dtype=float)  # Linear equation: y = 5x + 7

# Create the sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Input: a single value (x value)
    tf.keras.layers.Dense(units=1)  # Dense layer with 1 neuron (y value)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),  # Uses a learning rate of 0.1 (step size). 0.01 and 0.001 are more stable but slower to converge
    loss='mean_squared_error'  # This function calculates the mean squared error
)

print("Welcome to RAÚL SALAS's Neural Network!\n")  # From this message on, the neural network runs with the pre-established configuration
print("Training the network........")  # Training starts
historial = model.fit(valor_x, resultado, epochs=1000, verbose=False)  # Iterates 1000 times (epochs) through the dataset, adjusting weights to get closer to the target values
# 1000 epochs ≈ 1 min (approximate result) | 10000 epochs ≈ 10 min (exact result)
print("Network trained!")  # Indicates that 1000 training steps have been completed to optimize the output

print("Let's find out the result")
# Define the prediction input
resultado_predict = model.predict(np.array([7]).reshape(-1, 1))  # We put the X value into 'np.array'. Reshape with numpy reshapes the array without changing its data
# Use the prediction variable to print the result
print("The result is " + str(resultado_predict) + " ")  # Finally, we output the value of y
print("The rounded result is " + str(np.round(resultado_predict)))  # Round the result for readability