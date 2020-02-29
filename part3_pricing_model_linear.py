from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

from sklearn import preprocessing
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier



class Net(nn.Module):
    '''
    Nueral network: copying from part2 but we've slightly changed it
    '''
    
    def __init__(self):
        # since we will use a label binariser, the num of features is unknown due 
        # to the one-hot representation (unless u mannually count them), we configurate
        # the network architecture in another function
        super().__init__()
        self.linear1 = None
        self.linear2 = None
        self.linear3 = None
        self.linear4 = None
        
    def forward(self, X_raw):
        # PLEASE call setNetwork before doing forward()
        out = self.linear1(X_raw)
        out = F.leaky_relu(out)
        out = self.linear2(out)
        out = F.leaky_relu(out)
        out = self.linear3(out)
        out = F.leaky_relu(out)
        out = torch.softmax(self.linear4(out)) # softmax gives prob
        return out

    # set (modify) the network architecture
    def setNetwork(self,input_size,l1=1024, l2=256, l3=64):
        self.linear1 = nn.Linear(in_features=input_size, out_features=l1, bias=True)
        self.linear2 = nn.Linear(in_features=l1, out_features=l2, bias=True)
        self.linear3 = nn.Linear(in_features=l2, out_features=l3, bias=True)
        self.linear4 = nn.Linear(in_features=l3, out_features=1, bias=True)


# class for part 3
class PricingModelLinear():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False, batch_size = 200):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities

        self.batch_size = batch_size
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = Net() # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw,heading):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        
        heading: python list
            a list of names of the features in the data set

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================

        assert X_raw.shape[1] == len(heading), "Wrong input dimension!"

        # convert the x_raw as pandas dataframe
        dat = pd.DataFrame(X_raw)
        dat.columns = heading

        # dat.columns = ['id_policy', 'pol_bonus', 'pol_coverage', 'pol_duration', \
        #     'pol_sit_duration', 'pol_pay_freq','pol_payd' , 'pol_usage', \
        #         'pol_insee_code', 'drv_drv2', 'drv_age1', 'drv_age2', 'drv_sex1',\
        #             'drv_sex2', 'drv_age_lic1', 'drv_age_lic2', 'vh_age', 'vh_cyl', \
        #                 'vh_din','vh_fuel', 'vh_make','vh_model', 'vh_sale_begin', \
        #                     'vh_sale_end', 'vh_speed', 'vh_type', 'vh_value', \
        #                         'vh_weight', 'town_mean_altitude', \
        #                             'town_surface_area', 'population', 'commune_code', \
        #                                 'canton_code','city_district_code', \
        #                                     'regional_department_code']


        # drop the useless features
        dat.drop(columns=["id_policy"],axis=1)
        
        # turn the YES/NO into binary
        dat.pol_payd.map(dict(Yes=1, No=0))
        dat.drv_drv2.map(dict(Yes=1, No=0))

        ##############################################################
        # Handeling features with non-numeric values
        ##############################################################
        
        non_num_feature_names = ["pol_coverage","pol_pay_freq", \
            "pol_usage","drv_sex1","drv_sex2","vh_fuel","vh_make",\
                "vh_model","vh_type"]
        # non-numeric features
        non_num_feature = dat.drop(columns=non_num_feature_names,axis=1)

        # label binarizer
        lb = preprocessing.LabelBinarizer()
        # each element is a one-hot-vectorised feature
        vector_set = [] 
        for i in range(len(non_num_feature_names)):
            # select a feature (a column)
            data = non_num_feature.loc[:,non_num_feature_names[i]]
            # binarise: each element has N rows, each row is a one-hot 
            # vector, where N is the num of data points
            vectors = lb.fit_transform(data.to_numpy())
            vector_set.append(vectors)

        ##############################################################
        # Handeling (normalise) features with non-numeric values
        ##############################################################

        data = dat.to_numpy()
        # scaler/normaliser
        normaliser = preprocessing.MinMaxScaler()
        # clean data
        X = normaliser.fit_transform(data)
        

        ##############################################################
        # Merge the processed features into a clean training set
        ##############################################################
        for feature in vector_set:
            X.append(feature,axis=1)
        
        ##############################################################
        # Reconfigurate the input size of the base_classifier
        ##############################################################
        self.base_classifier.setNetwork(X.shape[1])

        return X



    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # define the optimiser
        optimizer = optim.Adam(self.model.parameters())

        # parsing numpy arrays into tensors
        x_train = torch.tensor(X_clean, dtype=torch.float)
        y_train = torch.tensor(y_raw, dtype=torch.float) # y_raw is clean
        train_set = TensorDataset(x_train, y_train)
        # define data loader
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # epoch number
        epoch_num = y_train.shape[0] // self.batch_size
        # training
        for epoch in range(self.epoch_num):
            for inputs, label in train_loader:
                optimizer.zero_grad()
                # output from the network
                pred = self.base_classifier(inputs)
                loss = F.binary_cross_entropy(pred, torch.unsqueeze(label, dim=1))
                # backprop
                loss.backward()
                optimizer.step()

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        return self.base_classifier



    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        X_clean = torch.tensor(X_clean, dtype=torch.float)
        probability = self.base_classifier(X_clean)

        return  # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model




if __name__ == '__main__':
    # Read raw data from csv file
    df = pd.read_csv("part3_training_data.csv")
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