import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from keras.optimizers import Adam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.config import MAX_POINTS, FILE_LABEL_MAPPING
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


with open("data/density_dfs.pkl", "rb") as f:
    density_dfs = pickle.load(f)

with open("data/centroid_dfs.pkl", "rb") as f:
    centroid_dfs = pickle.load(f)

with open("data/point_dfs.pkl", "rb") as f:
    point_dfs = pickle.load(f)

# Prepare data for each file separately
file_data = {}

for file_name in FILE_LABEL_MAPPING.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_density = density_dfs[table_name]
        df_centroid = centroid_dfs[table_name]

        # Filter data for the current file
        df_density = df_density[df_density['file'] == file_name]
        df_centroid = df_centroid[df_centroid['file'] == file_name]

        # Merge density and centroid data on time and file columns
        combined_df = pd.merge(df_density, df_centroid, on=['time', 'file'])

        # Assign labels based on file and table name
        label = FILE_LABEL_MAPPING[file_name][table_name]

        if label:
            combined_df['label'] = 1
        else:
            combined_df['label'] = 0

        # Fill NaN values with 0s (padding)
        combined_df[['centroid_x', 'centroid_y', 'centroid_z']] = combined_df[['centroid_x', 'centroid_y', 'centroid_z']].fillna(0)

        file_dfs[table_name] = combined_df

    file_data[file_name] = file_dfs

# Define the LSTM model creation in a function
def create_lstm_model(input_shape, learning_rate):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(units=64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    outputs = layers.Dense(units=1, activation='sigmoid')(x)

    optimizer = Adam(learning_rate=learning_rate)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
window_size = 50  # 50 time steps corresponding to 5 seconds
step_size = 25    # 50% overlap
learning_rate = 0.00005

all_files = list(file_data.keys())
results = []

for test_file in all_files:
    train_features = []
    train_labels = []
    test_features_A, test_labels_A = [], []
    test_features_B, test_labels_B = [], []
    test_features_C, test_labels_C = [], []
    
    for file_name, dfs in file_data.items():
        for table_name, df in dfs.items():
            num_rows = len(df)

            # If the DataFrame has fewer rows than the window size, skip this file
            if num_rows < window_size:
                continue

            for i in range(0, num_rows - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]
                feature_window = window[['density', 'centroid_x', 'centroid_y', 'centroid_z']].values
                label_window = window['label'].max()  # If any label in the window is 1, the window is labeled 1

                if file_name == test_file:
                    if table_name == 'Table A':
                        test_features_A.append(feature_window)
                        test_labels_A.append(label_window)
                    elif table_name == 'Table B':
                        test_features_B.append(feature_window)
                        test_labels_B.append(label_window)
                    elif table_name == 'Table C':
                        test_features_C.append(feature_window)
                        test_labels_C.append(label_window)
                else:
                    train_features.append(feature_window)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train = np.array(train_features)  # Shape: (num_windows, window_size, num_features)
    y_train = np.array(train_labels)    # Shape: (num_windows,)
    # Convert to numpy arrays
    X_test_A = np.array(test_features_A)
    X_test_B = np.array(test_features_B)
    X_test_C = np.array(test_features_C)
    y_test_A = np.array(test_labels_A)
    y_test_B = np.array(test_labels_B)
    y_test_C = np.array(test_labels_C)

    # Create and train the model
    model = create_lstm_model(input_shape=(window_size, X_train.shape[2]), learning_rate=learning_rate)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    # Concatenate the test features and labels for all tables
    X_test_combined = np.concatenate([X_test_A, X_test_B, X_test_C], axis=0)
    y_test_combined = np.concatenate([y_test_A, y_test_B, y_test_C], axis=0)

    # Evaluate the model on the combined dataset
    test_loss_combined, test_acc_combined = model.evaluate(X_test_combined, y_test_combined, verbose=0)

    y_combined_pred = (model.predict(X_test_combined) > 0.5).astype("int32")

    cm = confusion_matrix(y_test_combined, y_combined_pred)

    results.append({
        'test_file': test_file,
        'test_acc': test_acc_combined,
        'confusion_matrix': cm
    })

    # Print the combined accuracy
    print(f'Combined Test Accuracy for {test_file}: {test_acc_combined:.2f}')

    plot_loss(history)

    # Predict for each table separately
    y_pred_A = (model.predict(X_test_A) > 0.5).astype(int)
    y_pred_B = (model.predict(X_test_B) > 0.5).astype(int)
    y_pred_C = (model.predict(X_test_C) > 0.5).astype(int)

    # Now create the confusion matrices for each table
    cm_A = confusion_matrix(y_test_A, y_pred_A, labels=[0, 1])
    cm_B = confusion_matrix(y_test_B, y_pred_B, labels=[0, 1])
    cm_C = confusion_matrix(y_test_C, y_pred_C, labels=[0, 1])

    # Display the confusion matrix for each table
    disp_A = ConfusionMatrixDisplay(confusion_matrix=cm_A, display_labels=['Table A 0', 'Table A 1'])
    disp_B = ConfusionMatrixDisplay(confusion_matrix=cm_B, display_labels=['Table B 0', 'Table B 1'])
    disp_C = ConfusionMatrixDisplay(confusion_matrix=cm_C, display_labels=['Table C 0', 'Table C 1'])

    disp_A.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Table A - {test_file}')
    plt.show()

    disp_B.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Table B - {test_file}')
    plt.show()

    disp_C.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Table C - {test_file}')
    plt.show()

for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])