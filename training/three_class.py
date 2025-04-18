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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


# Define learning rate scheduler
def lr_schedule(epoch, lr):
    decay_rate = 0.9
    decay_step = 5
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Constants
window_size = 50
step_size = 25
initial_lr = 0.005

# Process point_dfs to ensure all table_points have consistent shape
for table_name, df in point_dfs.items():
    for index, row in df.iterrows():
        points = row['table_points']
        if len(points) == 0:
            df.at[index, 'table_points'] = np.zeros((MAX_POINTS, 4))
        else:
            points = np.array(points)
            if len(points) < MAX_POINTS:
                padded_points = np.pad(points, ((0, MAX_POINTS - len(points)), (0, 0)), mode='constant')
                df.at[index, 'table_points'] = padded_points
            else:
                df.at[index, 'table_points'] = points[:MAX_POINTS]

# Combine the data for raw points, density, and centroid coordinates
file_data = {}
for file_name in FILE_LABEL_MAPPING.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_points = point_dfs[table_name]
        df_density = density_dfs[table_name]
        df_centroid = centroid_dfs[table_name]

        # Filter data for the current file
        df_points = df_points[df_points['file'] == file_name]
        df_density = df_density[df_density['file'] == file_name]
        df_centroid = df_centroid[df_centroid['file'] == file_name]

        # Merge point, density, and centroid data
        combined_df = pd.merge(df_points, df_density[['time', 'density']], on='time', how='left')
        combined_df = pd.merge(combined_df, df_centroid[['time', 'centroid_x', 'centroid_y', 'centroid_z']], on='time', how='left')

        # Assign labels based on file and table name
        combined_df['label'] = FILE_LABEL_MAPPING[file_name][table_name]

        # Fill NaN values with 0s (padding)
        combined_df[['centroid_x', 'centroid_y', 'centroid_z', 'density']] = combined_df[['centroid_x', 'centroid_y', 'centroid_z', 'density']].fillna(0)

        file_dfs[table_name] = combined_df
    file_data[file_name] = file_dfs

# Define PointNet block
def pointnet_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1, input_shape[1]))(inputs)
    x = layers.Conv2D(64, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=1, activation='relu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    model = models.Model(inputs, x)
    return model

# Model including derived features
def create_hybrid_pointnet_bilstm_model(time_steps, num_points, num_features, num_derived_features, initial_lr):
    # PointNet block for raw mmWave data
    point_input = layers.Input(shape=(time_steps, num_points, num_features))
    pointnet = pointnet_block((num_points, num_features))
    spatial_features = layers.TimeDistributed(pointnet)(point_input)

    # Dense layers to process derived features (density and centroid coordinates)
    derived_input = layers.Input(shape=(time_steps, num_derived_features))
    derived_features = layers.TimeDistributed(layers.Dense(16, activation='relu'))(derived_input)
    derived_features = layers.TimeDistributed(layers.Dense(32, activation='relu'))(derived_features)

    # Combine spatial features from PointNet and derived features
    combined_features = layers.concatenate([spatial_features, derived_features], axis=-1)

    # LSTM and Attention for temporal dependencies
    lstm_out = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(combined_features)
    attention = layers.Attention()([lstm_out, lstm_out])
    pooled_out = layers.GlobalMaxPooling1D()(attention)

    # Fully connected layers before output
    dense_out = layers.Dense(32, activation='relu')(pooled_out)
    outputs = layers.Dense(3, activation='softmax')(dense_out)

    # Compile the model
    model = models.Model(inputs=[point_input, derived_input], outputs=outputs)
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation and training
scheduler = callbacks.LearningRateScheduler(lr_schedule)
all_files = list(file_data.keys())
results = []

for test_file in all_files:
    train_features, train_derived, train_labels = [], [], []
    test_features, test_derived, test_labels = [], [], []

    for file_name, dfs in file_data.items():
        for table_name, df in dfs.items():
            num_rows = len(df)
            if num_rows < window_size:
                continue

            for i in range(0, num_rows - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]
                feature_window = np.stack(window['table_points'].values)
                density_features = window[['density']].values
                centroid_features = window[['centroid_x', 'centroid_y', 'centroid_z']].values
                derived_features = np.hstack([density_features, centroid_features])
                
                label_window = window['label'].mode()[0]

                if file_name == test_file:
                    test_features.append(feature_window)
                    test_derived.append(derived_features)
                    test_labels.append(label_window)
                else:
                    train_features.append(feature_window)
                    train_derived.append(derived_features)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train = np.array(train_features)
    X_train_derived = np.array(train_derived)
    y_train = to_categorical(train_labels, num_classes=3)

    X_test = np.array(test_features)
    X_test_derived = np.array(test_derived)
    y_test = to_categorical(test_labels, num_classes=3)

    # Model training
    model = create_hybrid_pointnet_bilstm_model(window_size, MAX_POINTS, 4, 4, initial_lr)
    history = model.fit([X_train, X_train_derived], y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1, callbacks=[scheduler])
    
    plot_loss(history)

    # Model evaluation
    test_loss, test_acc = model.evaluate([X_test, X_test_derived], y_test)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')
    
    y_pred = np.argmax(model.predict([X_test, X_test_derived]), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    results.append({
        'test_file': test_file,
        'test_acc': test_acc,
        'confusion_matrix': cm
    })

    # Confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Human", "Sitting", "Standing"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()
    
# Summary of results
for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

# Average accuracy across cross-validation
average_accuracy = np.mean([result['test_acc'] for result in results])
print(f"Average Accuracy: {average_accuracy:.2f}")
