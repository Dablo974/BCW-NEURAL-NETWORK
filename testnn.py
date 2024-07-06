import os
import numpy as np
import pandas as pd
import pickle as pkl
from NN import ClassificationModel, train_test_split

def student_model_train():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    save_folder = "./models/"

    print("Loading the dataset")
    wdbc_data = pd.read_csv("data/wdbc.data", header=None)
    X = wdbc_data.iloc[:, 2:].values
    y = wdbc_data.iloc[:, 1].values

    print("Encoding target labels")
    y = np.where(y == 'M', 1, 0).reshape(-1, 1)

    print("Splitting the dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Removed random_state

    # print("Training and saving the Classification model")
    # input_dim = X.shape[1]
    # classif_model = ClassificationModel(input_dim, hidden_dim=32, learning_rate=0.01, epochs=1000)
    # classif_model.train(X_train, y_train)
    
    # with open(os.path.join(save_folder, 'classif.pkl'), 'wb') as f:
    #     pkl.dump(obj=classif_model, file=f)

    # return X_test, y_test

    print("Training and saving the Classification model")
    input_dim = X.shape[1]
    classif_model = ClassificationModel(input_dim, hidden_dim=32, learning_rate=0.01, epochs=1000)
    classif_model.train(X_train, y_train)
    
    # Save the model
    with open(os.path.join(save_folder, 'classif.pkl'), 'wb') as f:
        pkl.dump(obj=classif_model, file=f)

    # Save weights and biases
    classif_model.save_weights_and_biases('my-neural-network-app/public/model_params.json')

    return X_test, y_test

if __name__ == "__main__":
    # Run the training
    X_test, y_test = student_model_train()

    # Evaluate the model
    print("-----------------------------------------------")
    with open('models/classif.pkl', 'rb') as file:
        classif_model = pkl.load(file)
        evaluation_results = classif_model.evaluate(X_test, y_test)
        print("Classification Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric.capitalize()}: {value:.4f}")