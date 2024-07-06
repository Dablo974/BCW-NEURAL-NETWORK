import numpy as np
import time
import os
import json

# Fonction pour diviser les données en ensembles d'entraînement et de test
# Arguments:
# - X: les données d'entrée
# - y: les étiquettes correspondantes
# - test_size: la proportion des données à utiliser pour le test
# - random_state: la graine aléatoire pour la reproductibilité
def train_test_split(X, y, test_size=0.2, random_state=None):
    # Si un état aléatoire est fourni, utiliser cette seed
    if random_state is not None:
        np.random.seed(random_state)
    else:
        # Sinon, utiliser l'heure actuelle comme seed
        np.random.seed(int(time.time()))

    # Nombre total d'échantillons
    num_samples = X.shape[0]
    # Générer des indices pour les échantillons
    indices = np.arange(num_samples)
    # Mélanger les indices
    np.random.shuffle(indices)

    # Calculer le nombre d'échantillons de test
    test_samples = int(num_samples * test_size)

    # Séparer les indices en indices de test et d'entraînement
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Classe pour le modèle de classification
# Arguments du constructeur:
# - input_dim: la dimension d'entrée des données
# - hidden_dim: la dimension de la couche cachée
# - learning_rate: le taux d'apprentissage
# - epochs: le nombre d'époques d'entraînement
class ClassificationModel:
    def __init__(self, input_dim, hidden_dim=32, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim  # Dimension d'entrée
        self.hidden_dim = hidden_dim  # Dimension cachée
        self.output_dim = 1  # Classification binaire (0 pour bénin, 1 pour malin)
        self.learning_rate = learning_rate  # Taux d'apprentissage
        self.epochs = epochs  # Nombre d'époques

        # Initialiser les poids et les biais avec une graine aléatoire basée sur l'heure actuelle
        np.random.seed(int(time.time()))
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2. / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2. / self.hidden_dim)
        self.b2 = np.zeros((1, self.output_dim))

    # Fonction d'activation sigmoïde
    # Arguments:
    # - x: la valeur d'entrée
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Propagation avant
    # Arguments:
    # - X: les données d'entrée
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1  # Calculer les valeurs de la première couche
        self.A1 = self.sigmoid(self.Z1)  # Appliquer la fonction d'activation sigmoïde
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Calculer les valeurs de la deuxième couche
        self.A2 = self.sigmoid(self.Z2)  # Appliquer la fonction d'activation sigmoïde
        return self.A2

    # Rétropropagation
    # Arguments:
    # - X: les données d'entrée
    # - y: les étiquettes correspondantes
    # - output: la sortie du modèle après la propagation avant
    def backward(self, X, y, output):
        m = X.shape[0]  # Nombre d'échantillons
        dZ2 = output - y  # Calculer l'erreur de sortie
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)  # Calculer le gradient des poids de la deuxième couche
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # Calculer le gradient des biais de la deuxième couche
        dZ1 = np.dot(dZ2, self.W2.T) * self.A1 * (1 - self.A1)  # Calculer l'erreur de la première couche
        dW1 = (1 / m) * np.dot(X.T, dZ1)  # Calculer le gradient des poids de la première couche
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # Calculer le gradient des biais de la première couche

        # Mettre à jour les poids et les biais
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Entraînement du modèle
    # Arguments:
    # - X: les données d'entrée
    # - y: les étiquettes correspondantes
    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)  # Propagation avant
            self.backward(X, y, output)  # Rétropropagation

    # Prédiction
    # Arguments:
    # - X: les données d'entrée
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)  # Retourner 1 si la sortie est supérieure à 0.5, sinon 0

    # Évaluation du modèle
    # Arguments:
    # - X_test: les données de test
    # - y_test: les étiquettes de test
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)  # Prédire les étiquettes pour l'ensemble de test
        accuracy = np.mean(predictions == y_test)  # Calculer la précision
        precision = np.sum((predictions == 1) & (y_test == 1)) / np.sum(predictions == 1)  # Calculer la précision
        recall = np.sum((predictions == 1) & (y_test == 1)) / np.sum(y_test == 1)  # Calculer le rappel
        f1_score = 2 * precision * recall / (precision + recall)  # Calculer le score F1
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        # Enregistrer les résultats dans un fichier
        file_path = 'files.txt'
        if not os.path.exists(file_path):
            print("Creating file")
            open(file_path, 'w').close()  # Créer le fichier s'il n'existe pas
        
        with open(file_path, 'w') as f:
            print("Writing to file")
            f.write("Test Set and Predicted Results:\n")
            for i in range(len(X_test)):
                f.write(f"y_test: {y_test[i]}, prediction: {predictions[i]}\n")
            f.write("\nEvaluation Results:\n")
            for metric, value in results.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        return results
    
    def get_weights_and_biases(self):
        """Extract weights and biases from the model."""
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist()
        }

    def save_weights_and_biases(self, filename='my-neural-network-app/public/model_params.json'):
        """Save weights and biases to a JSON file."""
        params = self.get_weights_and_biases()
        with open(filename, 'w') as f:
            json.dump(params, f)