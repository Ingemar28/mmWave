import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import layers, models, callbacks, regularizers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.config import MAX_POINTS, FILE_LABEL_MAPPING
from util.util import plot_loss
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

with open("data/density_dfs.pkl", "rb") as f:
    density_dfs = pickle.load(f)

with open("data/centroid_dfs.pkl", "rb") as f:
    centroid_dfs = pickle.load(f)

with open("data/point_dfs.pkl", "rb") as f:
    point_dfs = pickle.load(f)


# Process point_dfs to ensure all table_points have consistent shape
for table_name, df in point_dfs.items():
    for index, row in df.iterrows():
        points = row['table_points']

        # If table_points is empty, replace with zeros
        if len(points) == 0:
            df.at[index, 'table_points'] = np.zeros((MAX_POINTS, 4))  # Replace with zeros
        else:
            # If fewer than MAX_POINTS, pad with zeros
            points = np.array(points)
            # Pad or truncate to have consistent `MAX_POINTS`
            if len(points) < MAX_POINTS:
                padded_points = np.pad(points, ((0, MAX_POINTS - len(points)), (0, 0)), mode='constant')
                df.at[index, 'table_points'] = padded_points
            else:
                df.at[index, 'table_points'] = points[:MAX_POINTS]

# Now all table_points are padded/truncated to be (MAX_POINTS, 3)

# Prepare data for each file separately
file_data = {}

for file_name in FILE_LABEL_MAPPING.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_points = point_dfs[table_name]

        # Filter data for the current file
        df_points = df_points[df_points['file'] == file_name]

        # Assign labels based on file and table name
        df_points['label'] = FILE_LABEL_MAPPING[file_name][table_name]

        file_dfs[table_name] = df_points

    file_data[file_name] = file_dfs

def pointnet_block(input_shape):
    inputs = layers.Input(shape=input_shape)  # Input shape: (num_points, num_features)
    
    # Reshape to add an extra dimension for channels: (num_points, 1, num_features)
    x = layers.Reshape((input_shape[0], 1, input_shape[1]))(inputs)
    
    # Shared MLP applied to each point in the point cloud
    x = layers.Conv2D(64, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=1, activation='relu')(x)
    
    # Global max pooling
    x = layers.GlobalMaxPooling2D()(x)
    
    model = models.Model(inputs, x)
    return model

# Define the full model combining PointNet and LSTM with attention
def create_pointnet_bilstm_model(time_steps, num_points, num_features):
    point_input = layers.Input(shape=(time_steps, num_points, num_features))  # (time_steps, num_points, num_features)

    # PointNet block for spatial features
    pointnet = pointnet_block((num_points, num_features))

    # TimeDistributed PointNet over each time step
    spatial_features = layers.TimeDistributed(pointnet)(point_input)

    # Bi-Directional LSTM for temporal dependencies
    lstm_out = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(spatial_features)

    # Attention mechanism
    attention = layers.Attention()([lstm_out, lstm_out])  # Self-attention

    # Pooling across time steps
    pooled_out = layers.GlobalMaxPooling1D()(attention)

    # Fully connected layer before the output
    dense_out = layers.Dense(units=32, activation='relu')(pooled_out)

    # Output layer for 3 classes: no human, sitting, standing
    outputs = layers.Dense(units=3, activation='softmax')(dense_out)

    model = models.Model(inputs=point_input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Perform cross-validation
window_size = 50  # 50 time steps corresponding to 5 seconds
step_size = 25    # 50% overlap
# learning_rate = 0.001

all_files = list(file_data.keys())
results = []

for test_file in all_files:
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for file_name, dfs in file_data.items():
        for table_name, df in dfs.items():
            num_rows = len(df)

            # If the DataFrame has fewer rows than the window size, skip this file
            if num_rows < window_size:
                continue

            for i in range(0, num_rows - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]

                # Extract the point cloud data (table_points) for this window
                # Since we preprocessed the point clouds, we can directly use them
                feature_window = np.stack(window['table_points'].values)

                label_window = window['label'].mode()[0]  # Most frequent label in the window

                if file_name == test_file:
                    test_features.append(feature_window)
                    test_labels.append(label_window)
                else:
                    train_features.append(feature_window)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train = np.array(train_features)  # Shape: (num_windows, window_size, MAX_POINTS, num_features)
    y_train = np.array(train_labels)    # Shape: (num_windows,)
    X_test = np.array(test_features)    # Shape: (num_windows, window_size, MAX_POINTS, num_features)
    y_test = np.array(test_labels)      # Shape: (num_windows,)

    y_train_onehot = to_categorical(y_train, num_classes=3)
    y_test_onehot = to_categorical(y_test, num_classes=3)

    # Create and train the model
    model = create_pointnet_bilstm_model(time_steps=window_size, num_points=MAX_POINTS, num_features=4)
    history = model.fit(X_train, y_train_onehot, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    plot_loss(history)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')

    # # Predict on the test set
    y_pred = model.predict(X_test)    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    # Ensure the confusion matrix expects all 3 classes even if a class is missing in the test set
    cm_labels = [0, 1, 2]  # Ensure these are ["No Human", "Sitting", "Standing"]

    # Confusion matrix for multi-class
    # cm = confusion_matrix(y_true, y_pred_classes)
    cm = confusion_matrix(y_true, y_pred_classes, labels=cm_labels)
    
    # Store the result
    results.append({
        'test_file': test_file,
        'test_acc': test_acc,
        'confusion_matrix': cm
    })

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Human", "Sitting", "Standing"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()

for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])