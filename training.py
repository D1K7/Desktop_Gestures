import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('balanced_landmarks.csv')

X = df.iloc[:, 1:].values.astype('float32')
y_raw = df.iloc[:, 0].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(42,)), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=75, 
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

model.save('gesture_recognizer1.keras')
np.save('classes.npy', label_encoder.classes_) 
print("Model saved")