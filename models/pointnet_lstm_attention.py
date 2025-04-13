import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Prepare data for each file separately
file_data = {}

for file_name in file_label_mapping.keys():
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
        label = file_label_mapping[file_name][table_name]

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
    
    # tablewise_true_labels = {'Table A': [], 'Table B': [], 'Table C': []}  # To store true labels for each table
    # tablewise_pred_labels = {'Table A': [], 'Table B': [], 'Table C': []}  # To store predicted labels for each table

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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_loss(history):
    # Extract loss values from the training process
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Number of epochs (assuming you know how many epochs were run)
    epochs = range(1, len(train_loss) + 1)

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Add title and labels
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set x-axis to show only integer values (epoch numbers)
    plt.xticks(epochs)  # Set x-ticks to be exactly the epoch numbers (1, 2, 3, ..., 20)

    plt.legend()
    plt.show()

# Prepare data for each file separately
file_data = {}

for file_name in file_label_mapping.keys():
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
        combined_df['label'] = file_label_mapping[file_name][table_name]

        # # Drop rows with None values in centroid coordinates
        # combined_df.dropna(subset=['centroid_x', 'centroid_y', 'centroid_z'], inplace=True)
        # padding
        combined_df[['centroid_x', 'centroid_y', 'centroid_z']] = combined_df[['centroid_x', 'centroid_y', 'centroid_z']].fillna(0)

        file_dfs[table_name] = combined_df

    file_data[file_name] = file_dfs


# Define the LSTM model creation in a function
def create_multiclass_lstm_model(input_shape, learning_rate):
    inputs = layers.Input(shape=input_shape)
    # x = layers.LSTM(units=32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.LSTM(units=64, return_sequences=True)(inputs)
    # x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    outputs = layers.Dense(units=3, activation='softmax')(x)  # Correctly connect the output layer

    optimizer = Adam(learning_rate=learning_rate)
    
    model = models.Model(inputs=inputs, outputs=outputs)  # Pass the output tensor here
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
window_size = 50  # 50 time steps corresponding to 5 seconds
step_size = 25    # 50% overlap
learning_rate = 0.000005

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
                feature_window = window[['density', 'centroid_x', 'centroid_y', 'centroid_z']].values
                label_window = window['label'].mode()[0]  # Most frequent label in the window

                if file_name == test_file:
                    test_features.append(feature_window)
                    test_labels.append(label_window)
                else:
                    train_features.append(feature_window)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train = np.array(train_features)  # Shape: (num_windows, window_size, num_features)
    y_train = np.array(train_labels)    # Shape: (num_windows,)
    X_test = np.array(test_features)    # Shape: (num_windows, window_size, num_features)
    y_test = np.array(test_labels)      # Shape: (num_windows,)

    y_train_onehot = to_categorical(y_train, num_classes=3)
    y_test_onehot = to_categorical(y_test, num_classes=3)

    # Create and train the model
    model = create_multiclass_lstm_model(input_shape=(window_size, X_train.shape[2]), learning_rate=learning_rate)
    history = model.fit(X_train, y_train_onehot, epochs=20, batch_size=16, validation_split=0.2, verbose=1)
    

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')

    plot_loss(history)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class labels
    y_true = np.argmax(y_test_onehot, axis=1)   # True labels (one-hot decoded)

    # Define the full set of labels you want to display (ensure this matches your class indices)
    all_labels = [0, 1, 2]  # Corresponding to "No Human", "Sitting", "Standing"
    display_labels = ["No Human", "Sitting", "Standing"]

    # Confusion matrix for multi-class
    cm = confusion_matrix(y_true, y_pred_classes, labels=all_labels)
    
    # Store the result
    results.append({
        'test_file': test_file,
        'test_acc': test_acc,
        'confusion_matrix': cm
    })

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()

# Optionally, summarize the results across all cross-validation runs
for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

for file_name in file_label_mapping.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_points = point_dfs[table_name]

        # Filter data for the current file
        df_points = df_points[df_points['file'] == file_name]

        # Assign labels based on file and table name
        df_points['label'] = file_label_mapping[file_name][table_name]

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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Process point_dfs to ensure all table_points have consistent shape
for table_name, df in point_dfs.items():
    for index, row in df.iterrows():
        points = row['table_points']
        if len(points) == 0:
            df.at[index, 'table_points'] = np.zeros((MAX_POINTS, 4))  # 4 features now (x, y, z, v)
        else:
            points = np.array(points)
            if len(points) < MAX_POINTS:
                padded_points = np.pad(points, ((0, MAX_POINTS - len(points)), (0, 0)), mode='constant')
                df.at[index, 'table_points'] = padded_points
            else:
                df.at[index, 'table_points'] = points[:MAX_POINTS]

# Prepare data for each file separately
file_data = {}
for file_name in file_label_mapping.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_points = point_dfs[table_name]
        df_points = df_points[df_points['file'] == file_name]
        # df_points['label'] = file_label_mapping[file_name][table_name]
        df_points.loc[:, 'label'] = file_label_mapping[file_name][table_name]
        file_dfs[table_name] = df_points
    file_data[file_name] = file_dfs

# PointNet block for spatial coordinates (x, y, z)
def pointnet_spatial_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1, input_shape[1]))(inputs)
    x = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=(1, 1), activation='relu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    return models.Model(inputs, x)

# Velocity block using Conv1D
def velocity_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=1, activation='relu')(inputs)
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    return models.Model(inputs, x)

# Full model with spatial and velocity processing
def create_pointnet_velocity_model(time_steps, num_points, spatial_dim=3, velocity_dim=1):
    spatial_input = layers.Input(shape=(time_steps, num_points, spatial_dim))
    velocity_input = layers.Input(shape=(time_steps, num_points, velocity_dim))
    spatial_pointnet = pointnet_spatial_block((num_points, spatial_dim))
    spatial_features = layers.TimeDistributed(spatial_pointnet)(spatial_input)
    velocity_net = velocity_block((num_points, velocity_dim))
    velocity_features = layers.TimeDistributed(velocity_net)(velocity_input)
    combined_features = layers.Concatenate()([spatial_features, velocity_features])
    lstm_out = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(combined_features)
    attention = layers.Attention()([lstm_out, lstm_out])
    pooled_out = layers.GlobalMaxPooling1D()(attention)
    dense_out = layers.Dense(units=32, activation='relu')(pooled_out)
    outputs = layers.Dense(units=3, activation='softmax')(dense_out)
    model = models.Model(inputs=[spatial_input, velocity_input], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
window_size = 50
step_size = 25
all_files = list(file_data.keys())
results = []

for test_file in all_files:
    train_features_spatial, train_features_velocity = [], []
    train_labels, test_features_spatial, test_features_velocity = [], [], []
    test_labels = []

    for file_name, dfs in file_data.items():
        for table_name, df in dfs.items():
            num_rows = len(df)
            if num_rows < window_size:
                continue
            for i in range(0, num_rows - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]
                point_cloud_data = np.stack(window['table_points'].values)
                feature_window_spatial = point_cloud_data[:, :, :3]  # Spatial coordinates
                feature_window_velocity = point_cloud_data[:, :, 3:4]  # Velocity feature
                label_window = window['label'].mode()[0]
                if file_name == test_file:
                    test_features_spatial.append(feature_window_spatial)
                    test_features_velocity.append(feature_window_velocity)
                    test_labels.append(label_window)
                else:
                    train_features_spatial.append(feature_window_spatial)
                    train_features_velocity.append(feature_window_velocity)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train_spatial = np.array(train_features_spatial)
    X_train_velocity = np.array(train_features_velocity)
    y_train = np.array(train_labels)
    X_test_spatial = np.array(test_features_spatial)
    X_test_velocity = np.array(test_features_velocity)
    y_test = np.array(test_labels)
    y_train_onehot = to_categorical(y_train, num_classes=3)
    y_test_onehot = to_categorical(y_test, num_classes=3)

    # Create and train the model
    model = create_pointnet_velocity_model(time_steps=window_size, num_points=MAX_POINTS, spatial_dim=3, velocity_dim=1)
    history = model.fit([X_train_spatial, X_train_velocity], y_train_onehot, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    plot_loss(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate([X_test_spatial, X_test_velocity], y_test_onehot)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')

    # Predictions and confusion matrix
    y_pred = model.predict([X_test_spatial, X_test_velocity])    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)
    cm_labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred_classes, labels=cm_labels)
    results.append({'test_file': test_file, 'test_acc': test_acc, 'confusion_matrix': cm})

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Human", "Sitting", "Standing"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()

# Summary of results
for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])


import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Prepare data for each file separately
file_data = {}

for file_name in file_label_mapping.keys():
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
        label = file_label_mapping[file_name][table_name]

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
    
    # tablewise_true_labels = {'Table A': [], 'Table B': [], 'Table C': []}  # To store true labels for each table
    # tablewise_pred_labels = {'Table A': [], 'Table B': [], 'Table C': []}  # To store predicted labels for each table

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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_loss(history):
    # Extract loss values from the training process
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Number of epochs (assuming you know how many epochs were run)
    epochs = range(1, len(train_loss) + 1)

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Add title and labels
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set x-axis to show only integer values (epoch numbers)
    plt.xticks(epochs)  # Set x-ticks to be exactly the epoch numbers (1, 2, 3, ..., 20)

    plt.legend()
    plt.show()

# Prepare data for each file separately
file_data = {}

for file_name in file_label_mapping.keys():
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
        combined_df['label'] = file_label_mapping[file_name][table_name]

        # # Drop rows with None values in centroid coordinates
        # combined_df.dropna(subset=['centroid_x', 'centroid_y', 'centroid_z'], inplace=True)
        # padding
        combined_df[['centroid_x', 'centroid_y', 'centroid_z']] = combined_df[['centroid_x', 'centroid_y', 'centroid_z']].fillna(0)

        file_dfs[table_name] = combined_df

    file_data[file_name] = file_dfs


# Define the LSTM model creation in a function
def create_multiclass_lstm_model(input_shape, learning_rate):
    inputs = layers.Input(shape=input_shape)
    # x = layers.LSTM(units=32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.LSTM(units=64, return_sequences=True)(inputs)
    # x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    outputs = layers.Dense(units=3, activation='softmax')(x)  # Correctly connect the output layer

    optimizer = Adam(learning_rate=learning_rate)
    
    model = models.Model(inputs=inputs, outputs=outputs)  # Pass the output tensor here
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
window_size = 50  # 50 time steps corresponding to 5 seconds
step_size = 25    # 50% overlap
learning_rate = 0.000005

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
                feature_window = window[['density', 'centroid_x', 'centroid_y', 'centroid_z']].values
                label_window = window['label'].mode()[0]  # Most frequent label in the window

                if file_name == test_file:
                    test_features.append(feature_window)
                    test_labels.append(label_window)
                else:
                    train_features.append(feature_window)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train = np.array(train_features)  # Shape: (num_windows, window_size, num_features)
    y_train = np.array(train_labels)    # Shape: (num_windows,)
    X_test = np.array(test_features)    # Shape: (num_windows, window_size, num_features)
    y_test = np.array(test_labels)      # Shape: (num_windows,)

    y_train_onehot = to_categorical(y_train, num_classes=3)
    y_test_onehot = to_categorical(y_test, num_classes=3)

    # Create and train the model
    model = create_multiclass_lstm_model(input_shape=(window_size, X_train.shape[2]), learning_rate=learning_rate)
    history = model.fit(X_train, y_train_onehot, epochs=20, batch_size=16, validation_split=0.2, verbose=1)
    

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')

    plot_loss(history)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class labels
    y_true = np.argmax(y_test_onehot, axis=1)   # True labels (one-hot decoded)

    # Define the full set of labels you want to display (ensure this matches your class indices)
    all_labels = [0, 1, 2]  # Corresponding to "No Human", "Sitting", "Standing"
    display_labels = ["No Human", "Sitting", "Standing"]

    # Confusion matrix for multi-class
    cm = confusion_matrix(y_true, y_pred_classes, labels=all_labels)
    
    # Store the result
    results.append({
        'test_file': test_file,
        'test_acc': test_acc,
        'confusion_matrix': cm
    })

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()

# Optionally, summarize the results across all cross-validation runs
for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the learning rate scheduler function
def lr_schedule(epoch, lr):
    decay_rate = 0.9
    decay_step = 5
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

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

file_data = {}
for file_name in file_label_mapping.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_points = point_dfs[table_name]
        df_points = df_points[df_points['file'] == file_name]
        df_points['label'] = file_label_mapping[file_name][table_name]
        file_dfs[table_name] = df_points
    file_data[file_name] = file_dfs

def pointnet_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1, input_shape[1]))(inputs)
    x = layers.Conv2D(64, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=1, activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=1, activation='relu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    model = models.Model(inputs, x)
    return model

def create_pointnet_bilstm_model(time_steps, num_points, num_features, initial_lr):
    point_input = layers.Input(shape=(time_steps, num_points, num_features))
    pointnet = pointnet_block((num_points, num_features))
    spatial_features = layers.TimeDistributed(pointnet)(point_input)
    lstm_out = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(spatial_features)
    attention = layers.Attention()([lstm_out, lstm_out])
    pooled_out = layers.GlobalMaxPooling1D()(attention)
    dense_out = layers.Dense(units=32, activation='relu')(pooled_out)
    outputs = layers.Dense(units=3, activation='softmax')(dense_out)

    model = models.Model(inputs=point_input, outputs=outputs)
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

window_size = 50
step_size = 25
initial_lr = 0.001
scheduler = callbacks.LearningRateScheduler(lr_schedule)

all_files = list(file_data.keys())
results = []

for test_file in all_files:
    train_features, train_labels, test_features, test_labels = [], [], [], []

    for file_name, dfs in file_data.items():
        for table_name, df in dfs.items():
            num_rows = len(df)
            if num_rows < window_size:
                continue

            for i in range(0, num_rows - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]
                feature_window = np.stack(window['table_points'].values)
                label_window = window['label'].mode()[0]

                if file_name == test_file:
                    test_features.append(feature_window)
                    test_labels.append(label_window)
                else:
                    train_features.append(feature_window)
                    train_labels.append(label_window)

    X_train = np.array(train_features)
    y_train = to_categorical(train_labels, num_classes=3)
    X_test = np.array(test_features)
    y_test = to_categorical(test_labels, num_classes=3)

    model = create_pointnet_bilstm_model(window_size, MAX_POINTS, 4, initial_lr)
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1, callbacks=[scheduler])
    
    plot_loss(history)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    results.append({
        'test_file': test_file,
        'test_acc': test_acc,
        'confusion_matrix': cm
    })

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Human", "Sitting", "Standing"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()
    
for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

average_accuracy = np.mean([result['test_acc'] for result in results])
print(f"Average Accuracy: {average_accuracy:.2f}")


import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the learning rate scheduler function
def lr_schedule(epoch, lr):
    decay_rate = 0.9
    decay_step = 5
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Process point_dfs to ensure all table_points have consistent shape
for table_name, df in point_dfs.items():
    for index, row in df.iterrows():
        points = row['table_points']
        if len(points) == 0:
            df.at[index, 'table_points'] = np.zeros((MAX_POINTS, 4))  # 4 features now (x, y, z, v)
        else:
            points = np.array(points)
            if len(points) < MAX_POINTS:
                padded_points = np.pad(points, ((0, MAX_POINTS - len(points)), (0, 0)), mode='constant')
                df.at[index, 'table_points'] = padded_points
            else:
                df.at[index, 'table_points'] = points[:MAX_POINTS]

# Prepare data for each file separately
file_data = {}
for file_name in file_label_mapping.keys():
    file_dfs = {}
    for table_name in ['Table A', 'Table B', 'Table C']:
        df_points = point_dfs[table_name]
        df_points = df_points[df_points['file'] == file_name]
        # df_points['label'] = file_label_mapping[file_name][table_name]
        df_points.loc[:, 'label'] = file_label_mapping[file_name][table_name]
        file_dfs[table_name] = df_points
    file_data[file_name] = file_dfs

# PointNet block for spatial coordinates (x, y, z)
def pointnet_spatial_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1, input_shape[1]))(inputs)
    x = layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=(1, 1), activation='relu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    return models.Model(inputs, x)

# Velocity block using Conv1D
def velocity_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=1, activation='relu')(inputs)
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    return models.Model(inputs, x)

# Full model with spatial and velocity processing
def create_pointnet_velocity_model(time_steps, num_points, spatial_dim=3, velocity_dim=1, initial_lr=0.001):
    spatial_input = layers.Input(shape=(time_steps, num_points, spatial_dim))
    velocity_input = layers.Input(shape=(time_steps, num_points, velocity_dim))
    spatial_pointnet = pointnet_spatial_block((num_points, spatial_dim))
    spatial_features = layers.TimeDistributed(spatial_pointnet)(spatial_input)
    velocity_net = velocity_block((num_points, velocity_dim))
    velocity_features = layers.TimeDistributed(velocity_net)(velocity_input)
    combined_features = layers.Concatenate()([spatial_features, velocity_features])
    lstm_out = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(combined_features)
    attention = layers.Attention()([lstm_out, lstm_out])
    pooled_out = layers.GlobalMaxPooling1D()(attention)
    dense_out = layers.Dense(units=32, activation='relu')(pooled_out)
    outputs = layers.Dense(units=3, activation='softmax')(dense_out)
    model = models.Model(inputs=[spatial_input, velocity_input], outputs=outputs)
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
window_size = 50
step_size = 25
all_files = list(file_data.keys())
results = []
initial_lr = 0.001
scheduler = callbacks.LearningRateScheduler(lr_schedule)

for test_file in all_files:
    train_features_spatial, train_features_velocity = [], []
    train_labels, test_features_spatial, test_features_velocity = [], [], []
    test_labels = []

    for file_name, dfs in file_data.items():
        for table_name, df in dfs.items():
            num_rows = len(df)
            if num_rows < window_size:
                continue
            for i in range(0, num_rows - window_size + 1, step_size):
                window = df.iloc[i:i+window_size]
                point_cloud_data = np.stack(window['table_points'].values)
                feature_window_spatial = point_cloud_data[:, :, :3]  # Spatial coordinates
                feature_window_velocity = point_cloud_data[:, :, 3:4]  # Velocity feature
                label_window = window['label'].mode()[0]
                if file_name == test_file:
                    test_features_spatial.append(feature_window_spatial)
                    test_features_velocity.append(feature_window_velocity)
                    test_labels.append(label_window)
                else:
                    train_features_spatial.append(feature_window_spatial)
                    train_features_velocity.append(feature_window_velocity)
                    train_labels.append(label_window)

    # Convert to numpy arrays for model input
    X_train_spatial = np.array(train_features_spatial)
    X_train_velocity = np.array(train_features_velocity)
    y_train = np.array(train_labels)
    X_test_spatial = np.array(test_features_spatial)
    X_test_velocity = np.array(test_features_velocity)
    y_test = np.array(test_labels)
    y_train_onehot = to_categorical(y_train, num_classes=3)
    y_test_onehot = to_categorical(y_test, num_classes=3)

    # Create and train the model
    model = create_pointnet_velocity_model(time_steps=window_size, num_points=MAX_POINTS, spatial_dim=3, velocity_dim=1, initial_lr=initial_lr)
    history = model.fit([X_train_spatial, X_train_velocity], y_train_onehot, epochs=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=[scheduler])
    
    plot_loss(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate([X_test_spatial, X_test_velocity], y_test_onehot)
    print(f'Test Accuracy for {test_file}: {test_acc:.2f}')

    # Predictions and confusion matrix
    y_pred = model.predict([X_test_spatial, X_test_velocity])    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)
    cm_labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred_classes, labels=cm_labels)
    results.append({'test_file': test_file, 'test_acc': test_acc, 'confusion_matrix': cm})

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Human", "Sitting", "Standing"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Test Confusion Matrix for {test_file}')
    plt.show()

# Summary of results
for result in results:
    print(f"Test file: {result['test_file']}, Test Accuracy: {result['test_acc']:.2f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

average_accuracy = np.mean([result['test_acc'] for result in results])
print(f"Average Accuracy: {average_accuracy:.2f}")

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

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
for file_name in file_label_mapping.keys():
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
        combined_df['label'] = file_label_mapping[file_name][table_name]

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
