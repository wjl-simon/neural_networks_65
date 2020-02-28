from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ClaimClassifier from part2_claim_classifier.py


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier



# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
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
        self.base_classifier = ClaimClassifier() # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        # loading dataset
        dat = pd.read_csv("part2_training_data.csv")
        # drop the useless
        dat.drop(columns=["id_policy, claim_amount"])

        # get the label of the traing set
        y = dat.drop(columns=["made_claim"]).to_numpy()

        ###############################################################
        # features with string values (including binary YES/NO)
        ###############################################################
        
        # get the name of features with string values
        str_features_name = ["pol_coverage","pol_pay_freq","pol_payd", \
            "pol_usage","drv_drv2","drv_sex1","drv_sex2","vh_fuel", \
                "vh_make","vh_model","vh_type"]
        
        # get the raw data of features with string values
        str_features = dat.drop(columns=str_features_name)

        # label binarizer     
        lb = preprocessing.LabelBinarizer() 
        
        # produce one-hot row vectors for each data point
        # each element of this list is a set of one-hot row numpy vectors
        vector_set = []

        for i in range(len(str_features_name)):
            # 1. find the uniuqe values (the support)
            # select the corresponding column
            temp1 = str_features.loc[:,str_features_name[i]]
            # find the uniques values (suppory)
            support = temp1.unique().to_numpy()

            # 2. # one-hot vectors for the i-th feature 
            vectors = lb.transform(temp1.to_numpy())
            # save them
            vector_set.append(vectors)
        

        ###############################################################
        # features with numeric values
        ###############################################################
        # numeric feature vectors into ndarrays
        x = dat.to_numpy()
        normaliser = preprocessing.MinMaxScaler()
        # normalised into between 0 and 1
        x_scaled = normaliser.fit_transform(x)


        ###############################################################
        # merge the two parts
        ###############################################################
        X = x_scaled
        for part in vector_set:
            X.append(part, axis=1)

        X.append(y, axis=1) # merge the label

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

