import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess():
    dataset = pd.read_csv('D:/Work/Datasets/loan_data_set.csv')

    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()
    min_max_scalar = MinMaxScaler()

    dataset = dataset.drop(columns=['Loan_ID', 'Gender'])
    dataset.insert(2, 'TotalIncome', 0)

    for index in range(0,len(dataset['Self_Employed'])):
        if str(dataset['Self_Employed'][index]) == 'nan':
            dataset['Self_Employed'][index] = 'Not Applicable'
        if str(dataset['Married'][index]) == 'nan':
            dataset['Married'][index] = 'No'
        if str(dataset['Credit_History'][index]) == 'nan':
            dataset['Credit_History'][index] = 0
        if str(dataset['Loan_Amount_Term'][index]) == 'nan':
            dataset['Loan_Amount_Term'][index] = 480
        if str(dataset['LoanAmount'][index]) == 'nan':
            dataset['LoanAmount'][index] = 100
        dataset['Loan_Amount_Term'][index] = dataset['Loan_Amount_Term'][index] / 12
        dataset['TotalIncome'][index] = dataset['ApplicantIncome'][index] + dataset['CoapplicantIncome'][index]

    dataset['Property_Area'] = label_encoder.fit_transform(dataset['Property_Area'])
    dataset['Loan_Status'] = label_encoder.fit_transform(dataset['Loan_Status'])
    dataset['Self_Employed'] = label_encoder.fit_transform(dataset['Self_Employed'])
    dataset['Education'] = label_encoder.fit_transform(dataset['Education'])

    married = one_hot_encoder.fit_transform(np.array(dataset['Married']).reshape(-1, 1))
    dataset['Married'] = married.indices
    total_income = min_max_scalar.fit_transform(np.array(dataset['TotalIncome']).reshape(-1, 1))
    dataset['TotalIncome'] = np.concatenate(total_income)
    loan_amount = min_max_scalar.fit_transform(np.array(dataset['LoanAmount']).reshape(-1, 1))
    dataset['LoanAmount'] = np.concatenate(loan_amount)

    dataset = dataset.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

    y = dataset['Loan_Status']
    X = dataset.drop(columns=['Loan_Status'])

    return train_test_split(X, y, test_size = 0.08, random_state = 42)
