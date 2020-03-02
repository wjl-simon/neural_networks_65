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
from torch.utils.data import TensorDataset, DataLoader

#import random
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


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
    
    def __init__(self,input_size=22,l1=512, l2=128, l3=32):
        # since we will use a label binariser, the num of features is unknown due 
        # to the one-hot representation (unless u mannually count them), we configurate
        # the network architecture in another function
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=l1, bias=True)
        self.linear2 = nn.Linear(in_features=l1, out_features=l2, bias=True)
        self.linear3 = nn.Linear(in_features=l2, out_features=l3, bias=True)
        self.linear4 = nn.Linear(in_features=l3, out_features=1, bias=True)
        
    def forward(self, x_train):
        out = self.linear1(x_train)
        out = F.leaky_relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.leaky_relu(out)
        out = torch.sigmoid(self.linear4(out))
        return out    


class FreqClassifier(Net):
    def __init__(self, batch_size, epoch_num):
        super().__init__()
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.model = Net()
        self.classes_ = [0,1]

    def fit(self,X_clean,y_raw):
        # define the optimiser
        optimizer = optim.Adam(self.model.parameters())

        # parsing numpy arrays into tensors
        x_train = torch.tensor(X_clean, dtype=torch.float)
        y_train = torch.tensor(y_raw, dtype=torch.float) # y_raw is clean
        train_set = TensorDataset(x_train, y_train)
        # define data loader
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # epoch number
        # training
        for epoch in range(self.epoch_num):
            running_loss = 0.0
            for inputs, label in train_loader:
                optimizer.zero_grad()
                # output from the network
                pred = self.model(inputs)
                loss = F.binary_cross_entropy(pred, torch.unsqueeze(label, dim=1))
                # backprop
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Iteration: %d, Loss: %.5f." %(epoch + 1, running_loss / len(train_loader.dataset)))

        return self


    def predict_proba(self,X_clean):
        """predict probability function.

        Gives the predicted probability of claiming

        Parameters
        ----------
        X_clean : ndarray
            An array, this is the processed clean test set feature

        Returns
        -------
        out: ndarray
            A 1d np array of the predicted probability of claiming.
        """
        X_test = torch.tensor(X_clean, dtype=torch.float)
        out = np.zeros((X_test.shape[0],2))
        for i in range(len(out)):
            out[i,1] = self.model(X_test[i])
        
        out[:,0] = 1- out[:,1]

        return out


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=True,batch_size=200, epoch_num=100):
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
        self.base_classifier = FreqClassifier(batch_size,epoch_num) # ADD YOUR BASE CLASSIFIER HERE


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
        heading = ['id_policy', 'pol_bonus', 'pol_coverage', 'pol_duration', \
            'pol_sit_duration', 'pol_pay_freq','pol_payd' , 'pol_usage', \
                'pol_insee_code', 'drv_drv2', 'drv_age1', 'drv_age2', 'drv_sex1',\
                    'drv_sex2', 'drv_age_lic1', 'drv_age_lic2', 'vh_age', 'vh_cyl', \
                        'vh_din','vh_fuel', 'vh_make','vh_model', 'vh_sale_begin', \
                            'vh_sale_end', 'vh_speed', 'vh_type', 'vh_value', \
                                'vh_weight', 'town_mean_altitude', \
                                    'town_surface_area', 'population', 'commune_code', \
                                        'canton_code','city_district_code', \
                                            'regional_department_code']

        assert X_raw.shape[1] == len(heading), "Wrong input dimension!"

        # convert the x_raw as pandas dataframe
        dat = pd.DataFrame(X_raw)
        dat.columns = heading

        # drop the useless features
        drop_list = ['id_policy', 'pol_insee_code','drv_age2', 'drv_sex2', 'vh_make', \
            'vh_model','vh_type','town_mean_altitude', 'town_surface_area', \
                'commune_code', 'canton_code', 'city_district_code', \
                    'regional_department_code']
        dat.drop(columns=drop_list,axis=1, inplace=True)

        # dealing with empty cells
        dat.replace(np.nan, -1, regex=True, inplace=True)

        
        # turn the YES/NO into binary
        dat.pol_payd.replace(to_replace=['No','Yes'], value=[0, 1],inplace=True)
        dat.drv_drv2.replace(to_replace=['No','Yes'], value=[0, 1],inplace=True)
        dat.drv_sex1.replace(to_replace=['F','M'], value=[0, 1],inplace=True)
        #dat.drv_sex2.replace(to_replace=['F','M'], value=[0, 1],inplace=True)

        ##############################################################
        # Handeling features with non-numeric values
        ##############################################################
        
        # non_num_feature_names = ['pol_coverage','pol_pay_freq', 'pol_usage', \
        #     'vh_fuel']
        # # non-numeric features
        # non_num_feature = dat[non_num_feature_names]
        # dat.drop(columns=non_num_feature_names, axis=1,inplace=True)
        
        # # print(dat.dtypes)
        # # print(non_num_feature.dtypes)

        # # label binarizer
        # lb = preprocessing.LabelBinarizer()
        # # each element is a one-hot-vectorised feature
        # vector_set = [] 
        # for i in range(len(non_num_feature_names)):
        #     # select a feature (a column)
        #     data = non_num_feature[non_num_feature_names[i]]

        #     # binarise: each element has N rows, each row is a one-hot 
        #     # vector, where N is the num of data points
        #     vectors = lb.fit_transform(data.values)
        #     vector_set.append(vectors)

        
        # with credit https://pbpython.com/categorical-encoding.html
        # a dict for translating the strings into numeric values
        str2num_dict = {}
        str2num_dict["pol_coverage"] = {"Maxi":0,'Maxi':1, 'Median2':2, 'Median1':3,
        'Mini':4}
        str2num_dict["pol_pay_freq"] = {"Yearly":0, "Monthly":1, 'Biannual':2, 
        "Quarterly":3}
        str2num_dict["pol_usage"] = {"WorkPrivate":0,"Professional":1,"AllTrips":2,"Retired":3}
        str2num_dict["vh_fuel"] = {"Diesel":0, "Gasoline":1, "Hybrid":2}

        dat.replace(str2num_dict, inplace=True)

        ##############################################################
        # Normalise numeric values
        ##############################################################

        #handel the missing cells and those outliers with non-numeric values
        # dat['pol_insee_code'] = pd.to_numeric(dat['pol_insee_code'],errors='coerce')
        # dat['regional_department_code'] = pd.to_numeric(dat['regional_department_code'],errors='coerce')

        # dat.replace(np.nan, -1, regex=True, inplace=True)

        data = dat.values
        # scaler/normaliser
        normaliser = preprocessing.MinMaxScaler()
        # clean data
        X = normaliser.fit_transform(data)
        
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
        self.y_std = np.std(claims_raw[nnz]) # std of claim amount
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)


        ##############################################################
        # Reconfigurate the input size of the base_classifier
        ##############################################################
        #self.base_classifier.model.setNetwork(X_clean.shape[1])

        
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
        X_raw : pd.dataFrame
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

        X_clean = self._preprocessor(X_raw.values)
        
        # return probabilities for the positive class (label 1)
        score = self.base_classifier.predict_proba(X_clean)

        return score[:,1]


    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : pd.dataFrame
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

        # Guassian noise N~(0,y_std)
        noise = np.random.normal(self.y_mean*2,self.y_std/5,X_raw.shape[0])
        
        #return self.predict_claim_probability(X_raw) * self.y_means
        #price =  self.predict_claim_probability(X_raw) * 2 * self.y_mean + noise
        price = self.predict_claim_probability(X_raw) * 3 * self.y_mean + noise
        
        return price

    

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
    # get raw claim amount
    claims_raw = df['claim_amount']
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
    
    #traaining
    X_train, y_train = train.iloc[:,:-1].values, train.iloc[:,-1].values

    # testing
    X_test_raw, y_test_raw = test.iloc[:,:-1], test.iloc[:,-1]

    # instantiate a model
    pricePredictor = PricingModel(epoch_num = 200)

    # training
    pricePredictor.fit(X_train, y_train, claims_raw)

    # save the model
    print('Saving the model')
    pricePredictor.save_model()

    # get the predicted claim probability
    # X_test_clean = pricePredictor._preprocessor(X_test_raw)
    # freq_predict = pricePredictor.base_classifier.predict_proba(X_test_clean)

    # load model
    classifier = load_model()

    res1 = classifier.predict_premium(X_test_raw)
    res2 = classifier.predict_claim_probability(X_test_raw)

    print('the predicted price on test set is {}'.format(res1))
    print('the predicted prob of claiming on test set is{}'.format(res2))

    print('The confusion matrix for the claiming is')
    print(confusion_matrix(y_test_raw, np.round(res2)))
    
    # roc-auc on the frequency model
    print('The ROC-AUC is {}'.format(roc_auc_score(y_test_raw.values, res2)))

    # average amount of price is:
    average_price = np.mean(res1)
    print('Average price given by the price odel is {}'. \
        format(average_price))

   
