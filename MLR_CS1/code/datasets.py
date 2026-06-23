import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :6]
    y = df.iloc[:, 6:]
    # print(X.head)
    # print(y.head)
    # print(df.head)
    return X, y


def prepare_dataset(file_path, test_size=0.2):
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_dataset("robot_kinematics_normalized_dataset.csv")
    X_train, X_test, y_train, y_test = prepare_dataset(
        "robot_kinematics_normalized_dataset.csv"
    )
    # print(X_train.head)
    # print(X_test.head)
    # print(y_train.head)
    # print(y_test.head)
