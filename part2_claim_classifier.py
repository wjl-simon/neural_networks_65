import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pickle




#build the model
class Net(nn.Module):

    def __init__(self, l1=1024, l2=256, l3=64):
        super().__init__()
        self.linear1 = nn.Linear(in_features=9, out_features=l1, bias=True)
        self.linear2 = nn.Linear(in_features=l1, out_features=l2, bias=True)
        self.linear3 = nn.Linear(in_features=l2, out_features=l3, bias=True)
        self.linear4 = nn.Linear(in_features=l3, out_features=1, bias=True)


    def forward(self, X_raw):
        out = self.linear1(X_raw)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = torch.sigmoid(self.linear4(out))
        return out


class ClaimClassifier():

    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.model = Net()
        self.scalar = None
        # Hyperpt5aremeters
        #self.set_hyperparameters()


    def set_hyperparameters(self, lr, batch_size, epoch):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        if self.scalar is None:
            # Initialise scalar
            self.scalar = MinMaxScaler()
            # Find min and max value for each property
            self.scalar.fit(X_raw)
        # YOUR CLEAN DATA AS A NUMPY ARRAY
        return self.scalar.transform(X_raw)

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        The trained model

        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE

        # Preprocess data
        X_clean = self._preprocessor(X_raw)
        # Define loss function and optimiser
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Create dataset
        x_train = torch.tensor(X_clean, dtype=torch.float)
        y_train = torch.tensor(y_raw, dtype=torch.float)
        train_set = TensorDataset(x_train, y_train)
        # Define data loader
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        # Train model
        for i in range(self.epoch):
            running_loss = 0.0
            for inputs, label in train_loader:
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = F.binary_cross_entropy(output, torch.unsqueeze(label, dim=1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Iteration: %d, Loss: %.5f." %(i + 1, running_loss / len(train_loader.dataset)))

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE
        X_clean = self._preprocessor(X_raw)
        X_clean = torch.tensor(X_clean, dtype=torch.float)
        results = torch.round(self.model(X_clean)).detach().numpy()
        return results # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self, X_raw, y_raw):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        X_raw_tensor = torch.tensor(X_raw)
        predict = self.predict(X_raw_tensor)


        # Plot ROC
        fpr, tpr, thresholds = roc_curve(y_raw, predict)
        #self.plot_roc_curve(fpr, tpr)

        # Return accuracy and auroc
        return accuracy_score(predict, y_raw), roc_auc_score(y_raw, predict)


    
    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


    def plot_roc_curve(self, fpr, tpr):
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()


def load_model():
# Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model



def ClaimClassifierHyperParameterSearch(X_raw, y_raw):  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """
    # Search space
    lrs = [1e-5, 1e-4, 1e-3]
    batch_sizes = [32, 64]
    epochs = [10, 100]
    # Save best parameters
    best_lr = None
    best_batch_size = None
    best_epoch = None
    best_accuracy = 0
    # Grid search
    for lr in lrs:
        for batch_size in batch_sizes:
            for epoch in epochs:

                # Evaluate model using K-fold cross validation
                k = 10
                kf = KFold(n_splits=k)
                total_accuracy = 0
                for train_index, test_index in kf.split(X_raw):
                    X_train, X_val = X_raw[train_index], X_raw[test_index]
                    y_train, y_val = y_raw[train_index], y_raw[test_index]

                    claimClassifier = ClaimClassifier()
                    claimClassifier.set_hyperparameters(lr=lr, batch_size=batch_size, epoch=epoch)
                    # Train model
                    claimClassifier.fit(X_train, y_train)
                    total_accuracy += claimClassifier.evaluate_architecture(X_val, y_val)
                average_accuracy = total_accuracy / k

                print(f"lr: {lr}. bs: {batch_size}. epoch: {epoch}. accuracy: {average_accuracy}")
                # Update Hyperparemeters
                if average_accuracy > best_accuracy:
                    best_lr = lr
                    best_batch_size = batch_size
                    best_epoch = epoch
                    best_accuracy = average_accuracy
    # Return best Hyperparemeters
    return best_lr, best_batch_size, best_epoch


def main():
    # Read raw data from csv file
    df = pd.read_csv("part2_training_data.csv")
    # Drop column claim_amount for obvious reason
    df = df.drop("claim_amount", 1)
    # Take all rows with claim_made == 1
    ones = df.loc[df["made_claim"] == 1]
    # Duplicate data by 10 times
    ones = pd.concat([ones] * 9, ignore_index=True)
    # Concat ones to original data
    df = pd.concat([df, ones], ignore_index=True)
    # # Shuffle data
    df = shuffle(df)
    # Split data into training (80%) and test set (20%)
    train, test = train_test_split(df, test_size=0.2)
    # Split data into inputs and labels
    X_train, y_train = train.iloc[:,:-1].values, train.iloc[:,-1].values
    X_test, y_test = test.iloc[:,:-1].values, test.iloc[:,-1].values
    # Find best Hyperparemeters
    #lr, batch_size, epoch = ClaimClassifierHyperParameterSearch(X_train, y_train)
    # From KFold, Best parameters lr=0.001, bs=64 epochs=100
    # Create classifier
    claimClassifier = ClaimClassifier()
    claimClassifier.set_hyperparameters(lr=0.001, batch_size=64, epoch=100)
    # Train classifier
    claimClassifier.fit(X_train, y_train)
    # Evaluate classifier
    accuracy, auroc = claimClassifier.evaluate_architecture(X_test, y_test)
    print(f"Model accuracy: {accuracy}. Model auroc: {auroc}")
    # Save model
    claimClassifier.save_model()
    # Load model
    #claimClassifier2 = load_model()
    #accuracy2 = claimClassifier2.evaluate_architecture(X_test, y_test)
    #print(f"accuracy of loaded model: {accuracy2}")


if __name__ == '__main__':
    main()
    # Read raw data from csv file
    # df = pd.read_csv("part2_training_data.csv")
    # # Drop column claim_amount for obvious reason
    # df = df.drop("claim_amount", 1)
    # # Take all rows with claim_made == 1
    # ones = df.loc[df["made_claim"] == 1]
    # # Duplicate data by 10 times
    # ones = pd.concat([ones] * 9, ignore_index=True)
    # # Concat ones to original data
    # df = pd.concat([df, ones], ignore_index=True)
    # # # Shuffle data
    # df = shuffle(df)
    # # Split data into training (80%) and test set (20%)
    # train, test = train_test_split(df, test_size=0.2)
    # # Split data into inputs and labels
    # X_train, y_train = train.iloc[:,:-1].values, train.iloc[:,-1].values
    # X_test, y_test = test.iloc[:,:-1].values, test.iloc[:,-1].values

    # claimClassifier = load_model()
    # predict_res = claimClassifier.predict(X_test)
    # print(predict_res.dtype)
    # print(predict_res.shape)
    # print(predict_res)

