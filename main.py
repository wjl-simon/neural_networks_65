from claimClassifier import ClaimClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def main():
    # Read raw data from csv file
    df = pd.read_csv("part2_training_data.csv")
    # Drop column claim_amount for obvious reason
    df = df.drop("claim_amount", 1)

    train, test = train_test_split(df, test_size=0.2)

    X_train, y_train = train.iloc[:,:-1].values, train.iloc[:,-1].values
    X_test, y_test = test.iloc[:,:-1].values, test.iloc[:,-1].values

    # Create classifier
    claimClassifier = ClaimClassifier()
    claimClassifier.fit(X_train, y_train)

    accuracy = claimClassifier.evaluate_architecture(X_test, y_test)
    print(accuracy)


if __name__ == '__main__':
    main()
