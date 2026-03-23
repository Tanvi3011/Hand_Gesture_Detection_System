import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("dataset/landmarks.csv")

# Remove empty or NaN rows
df = df.dropna()
df = df[df['label'].notna()]
df['label'] = df['label'].astype(str).str.strip().str.upper()

# Keep only A-Z letters
df = df[df['label'].str.isalpha()]
df = df[df['label'].str.len() == 1]
print(f"After filtering : {df.shape}")
print(f"Classes found   : {sorted(df['label'].unique())}")

X = df.drop("label", axis=1).values.astype(float)
y = df["label"].values

le    = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Total classes    : {len(le.classes_)}")

model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel summary:")
model.summary()

es = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining started...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

model.save("models/sign_model.h5")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel saved to   : models/sign_model.h5")
print("Encoder saved to : models/label_encoder.pkl")
print("\nDone! Ready to run main_app.py")