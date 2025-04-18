import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

# Load the dataset
df = pd.read_csv('updated_micro_doppler_dataset1.csv')

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

# Assuming all columns except 'label' are features

# ----> MODIFIED SECTION <----
# 1. Pad or truncate signals to a fixed length
def pad_or_truncate(signal, target_length):
    if len(signal) < target_length:
        return np.pad(signal, (0, target_length - len(signal)), 'constant')
    else:
        return signal[:target_length]

# 2. Apply padding/truncation to each signal and convert to NumPy array
df['flow_signal'] = df['flow_signal'].apply(lambda x: pad_or_truncate(np.fromstring(x.strip('[]'), sep=',').astype(float), target_length=100)) # Assuming a target length of 100

# Reshape data (e.g., for Conv1D input shape)
X = np.stack(df['flow_signal'].values) # Convert to a NumPy array of signals
X = X.reshape((X.shape[0], X.shape[1], 1)) # Reshape for Conv1D

# ----> END OF MODIFIED SECTION <----

y = df['label'].values  # Label column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)
# Save the model to a file
model.save('micro_doppler_model.keras')