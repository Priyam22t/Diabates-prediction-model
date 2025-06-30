import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define the Neural Network in TensorFlow
def DiabetesNN():
    inputs = tf.keras.layers.Input(shape=(8,))
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)  # Dropout layer to reduce overfitting
    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare dataset from the specified local path
def load_data(path="D:/python project/IDP/diabetes.csv"):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(path, header=None, names=columns)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target

    # Standardize features (important for neural networks)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_test, y_test), scaler

# Function to input new diabetes data
def input_data():
    print("Enter the following values for the diabetes prediction model:")
    data = []
    data.append(float(input("Pregnancies: ")))
    data.append(float(input("Glucose: ")))
    data.append(float(input("Blood Pressure: ")))
    data.append(float(input("Skin Thickness: ")))
    data.append(float(input("Insulin: ")))
    data.append(float(input("BMI: ")))
    data.append(float(input("Diabetes Pedigree Function: ")))
    data.append(float(input("Age: ")))

    return np.array(data).reshape(1, -1)  # Reshape to match input shape

# Train function for a single client
def train_client(model, train_data, epochs=5):
    X_train, y_train = train_data
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model.get_weights()

# Helper function to average the models' weights
def average_weights(models):
    avg_weights = []
    for weights_list_tuple in zip(*models):
        avg_weights.append(np.mean(weights_list_tuple, axis=0))
    return avg_weights

# Federated training process
def federated_learning(global_model, clients_data, rounds=20):
    for round in range(rounds):
        local_models = []

        # Train clients
        for client_data in clients_data:
            local_model = DiabetesNN()
            local_model.set_weights(global_model.get_weights())
            local_model_weights = train_client(local_model, client_data)
            local_models.append(local_model_weights)

        # Aggregating the models (manual averaging)
        global_weights = average_weights(local_models)
        global_model.set_weights(global_weights)

        print(f"Round {round + 1} completed.")

    return global_model

# Function to calculate and print accuracy for each client and combined data
def evaluate_clients(models, clients_data):
    accuracies = []
    for i, (model, client_data) in enumerate(zip(models, clients_data)):
        X_client, y_client = client_data
        predictions = np.round(model.predict(X_client))
        acc = accuracy_score(y_client, predictions)
        accuracies.append(acc)
        print(f"Client {i + 1} Accuracy: {acc * 100:.2f}%")
    return accuracies

# Main script
if __name__ == "__main__":
    # Load data for clients from the specified dataset
    (X_train, y_train), (X_test, y_test), scaler = load_data("D:/python project/IDP/diabetes.csv")

    # Split data among clients (3 clients in this case)
    client1_data = (X_train[:200], y_train[:200])
    client2_data = (X_train[200:400], y_train[200:400])
    client3_data = (X_train[400:], y_train[400:])
    clients_data = [client1_data, client2_data, client3_data]

    # Initialize models for each client for non-federated training and evaluation
    client_models = [DiabetesNN() for _ in clients_data]

    # Train and evaluate each client model individually (non-federated)
    print("\n--- Non-Federated Training & Evaluation ---")
    for model, client_data in zip(client_models, clients_data):
        train_client(model, client_data, epochs=5)

    # Evaluate each client model individually (non-federated)
    evaluate_clients(client_models, clients_data)

    # Combined accuracy for non-federated models on the test set
    combined_model = DiabetesNN()
    combined_model.set_weights(average_weights([model.get_weights() for model in client_models]))
    combined_predictions = np.round(combined_model.predict(X_test))
    combined_acc = accuracy_score(y_test, combined_predictions)
    print(f"Combined Non-Federated Test Accuracy: {combined_acc * 100:.2f}%")

    # Federated learning
    print("\n--- Federated Learning Training & Evaluation ---")
    global_model = DiabetesNN()
    trained_global_model = federated_learning(global_model, clients_data, rounds=20)

    # Test the global model after federated learning
    predictions = trained_global_model.predict(X_test)
    rounded_predictions = np.round(predictions)
    global_acc = accuracy_score(y_test, rounded_predictions)
    print(f"Federated Learning Test Accuracy: {global_acc * 100:.2f}%")

    # Input new data for prediction
    new_data = input_data()
    scaled_data = scaler.transform(new_data)  # Scale the new input with the same scaler
    raw_prediction = trained_global_model.predict(scaled_data)
    print(f"Raw Prediction (sigmoid output): {raw_prediction[0][0]}")
    rounded_prediction = np.round(raw_prediction)
    print(f"Predicted Diabetes Risk: {rounded_prediction[0][0] * 100:.2f}%")

    # Save the global model in the recommended Keras format
    trained_global_model.save("models/federated_diabetes_model.keras")

    import matplotlib.pyplot as plt

    # Placeholder for non-federated and federated accuracies
    non_federated_accuracy = combined_acc * 100  # Combined non-federated accuracy
    federated_accuracy = global_acc * 100  # Federated learning test accuracy

    # Create the bar chart
    methods = ['Non-Federated', 'Federated']
    accuracies = [non_federated_accuracy, federated_accuracy]

    plt.bar(methods, accuracies, color=['blue', 'red'])
    plt.ylim(0, 100)  # Set y-axis limit
    plt.title('Accuracy Comparison: Non-Federated vs Federated')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Training Method')

    # Annotate the bars with accuracy values
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    # Show the graph
    plt.show()

