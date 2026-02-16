import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

old_model = tf.keras.models.load_model('gesture_recognizer1.keras')


base_model = tf.keras.Sequential(old_model.layers[:-1])
for layer in base_model.layers:
    layer.trainable = False 


new_model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Dense(11, activation='softmax') 
])

new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# df=pd.read_csv("balanced_landmarks.csv")
# X = df.iloc[:, 1:].values.astype('float32')
# y_raw = df.iloc[:, 0].values

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y_raw)
# num_classes = len(label_encoder.classes_)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



# new_model.fit(X_combined, y_combined, epochs=10) 