import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)
X_testing= test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

X_scaler  = MinMaxScaler(feature_range=(0,1))
Y_scaler  = MinMaxScaler(feature_range=(0,1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)
