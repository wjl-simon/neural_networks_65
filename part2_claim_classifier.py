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
            # print("Iteration: %d, Loss: %.5f." %(i + 1, running_loss / len(train_loader.dataset)))

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
        results = torch.round(self.model(X_clean))
        return results # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self, X_raw, y_raw):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        predict = self.predict(X_raw)
        predict = predict.squeeze(1).detach().numpy()
        return accuracy_score(predict, y_raw)
        # Preprocess data
        # X_clean = self._preprocessor(X_raw)
        # # Create dataset
        # x_test = torch.tensor(X_clean, dtype=torch.float)
        # y_test = torch.tensor(y_raw, dtype=torch.float)
        # test_set = TensorDataset(x_test, y_test)
        # # Define data loader
        # test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        # correct = 0
        # with torch.no_grad():
        #     for inputs, targets in test_loader:
        #         outputs = self.model(inputs)
        #         predict = torch.squeeze(outputs.round())
        #         correct += predict.eq(targets).sum().item()
        # # Return test accuracy
        # return correct / len(test_loader.dataset)


        # pred = predict.cpu().numpy()
        # y = targets.cpu().numpy()





        #
        #
        #
        #
        #

        #
        # # k = 10
        # k_scores = []
        # # kf = KFold(n_splits=k)
        # # kf.get_n_splits(X_raw)
        # # for train_index, test_index in kf.split(X_raw):
        # #     X_train, X_test = X_raw[train_index], X_raw[test_index]
        # #     y_train, y_test = y_raw[train_index], y_raw[test_index]
        # #
        # #     model = self.fit(X_train, y_train)
        # #
        # #   proba = model.predict_proba(self._preprocessor(X_test))
        # auc = roc_auc_score(y, pred)
        # fpr, tpr, thresholds = roc_curve(y, pred)
        # self.plot_roc_curve(fpr, tpr)
        # k_scores.append(auc)
        # print("Average AUC: " + str(np.mean(k_scores)))
        #
        #








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



# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION

# def ClaimClassifierHyperParameterSearch(X_raw, y_raw):  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
#     """Performs a hyper-parameter for fine-tuning the classifier.
#
#     Implement a function that performs a hyper-parameter search for your
#     architecture as implemented in the ClaimClassifier class.
#
#     The function should return your optimised hyper-parameters.
#     """
#
#     param_grid = {
#         'l1': (16, 32, 64, 128),
#         'l2': (16, 32, 64, 128),
#         'ac1': ['relu', 'tanh'],
#         'ac2': ['relu', 'tanh'],
#         'epoch_size': [8, 12, 16, 20],
#         'batch_size': [16, 32, 64, 128]
#     }
#
#     best_auc = 0
#     best_hyperparameter = []
#     num_interation = 50
#     for i in range(num_interation):
#
#         hyperparameter_lst = []
#         l1 = random.sample(param_grid['l1'], 1)[0]
#         l2 = random.sample(param_grid['l2'], 1)[0]
#         ac1 = random.sample(param_grid['ac1'], 1)[0]
#         ac2 = random.sample(param_grid['ac2'], 1)[0]
#         epoch_size = random.sample(param_grid['epoch_size'], 1)[0]
#         batch_size = random.sample(param_grid['batch_size'], 1)[0]
#
#         hyperparameter_lst.append(l1)
#         hyperparameter_lst.append(l2)
#         hyperparameter_lst.append(ac1)
#         hyperparameter_lst.append(ac2)
#         hyperparameter_lst.append(epoch_size)
#         hyperparameter_lst.append(batch_size)
#
#         k = 3
#         k_scores = []
#         kf = KFold(n_splits=k)
#         kf.get_n_splits(X_raw)
#
#         for train_index, test_index in kf.split(X_raw):
#             X_train, X_test = X_raw[train_index], X_raw[test_index]
#             y_train, y_test = y_raw[train_index], y_raw[test_index]
#
#             scaler = MinMaxScaler()
#             scaler.fit(X_train)
#             X_train = scaler.transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = Sequential()
#             model.add(Dense(l1, input_shape=(9,)))
#             model.add(Activation(ac1))
#             model.add(Dense(l2))
#             model.add(Activation(ac2))
#             model.add(Dense(1))
#             model.add(Activation('sigmoid'))
#             model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#             model.fit(X_train, y_train, epochs=epoch_size, batch_size=batch_size, verbose=1)
#
#             proba = model.predict_proba(X_test)
#             auc = roc_auc_score(y_test, proba)
#             print(auc)
#             k_scores.append(auc)
#
#         avg_auc = np.mean(k_scores)
#         if avg_auc > best_auc:
#             best_auc = avg_auc
#             best_hyperparameter = hyperparameter_lst
#
#         print("auc")
#         print(best_auc)
#         return best_hyperparameter
#
#
#     # c.evaluate_architecture()
#
#     # Return the chosen hyper parameters

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
                k = 5
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
    claimClassifier.set_hyperparameters(lr=0.001, batch_size=4, epoch=100)
    # Train classifier
    claimClassifier.fit(X_train, y_train)
    # Evaluate classifier
    accuracy = claimClassifier.evaluate_architecture(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    # Save model
    claimClassifier.save_model()
    # Load model
    #claimClassifier2 = load_model()
    #accuracy2 = claimClassifier2.evaluate_architecture(X_test, y_test)
    #print(f"accuracy of loaded model: {accuracy2}")


if __name__ == '__main__':
    main()
