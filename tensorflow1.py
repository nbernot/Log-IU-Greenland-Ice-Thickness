import tensorflow as tf
import numpy as np
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Create dummy data

x = np.random.random((5, 3))
y = np.random.random((5, 1))

# Train the model
model.fit(x, y, epochs=1)

# Make predictions
predictions = model.predict(x)
print("Predictions:", predictions)
