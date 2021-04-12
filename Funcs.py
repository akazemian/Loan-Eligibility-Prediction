import pandas as pd
import numpy as np 

# return numerical columns
def numFeat(data):
    num_feats = data.dtypes[data.dtypes != 'object'].index.tolist()
    return data[num_feats]

# return categorical columns
def catFeat(data):
    cat_feats = data.dtypes[data.dtypes == 'object'].index.tolist()
    return data[cat_feats].drop(columns='Loan_ID')

# Impute missing values for categorical columns using mode
def getImputedCat(df):
    cols = df.columns.tolist()

    for col in cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# Impute missing values in numerivcal columns using mode, except for loan amount where mean is used 
def getImputedNum(df):
    cols = df.columns.tolist()

    for col in cols:
        if col == 'LoanAmount':
            df[col] = df[col].fillna(df[col].mean())
        #elif col == 'Credit_History':
        #    notNull = len(df[df[col] == 1.0]) + len(df[df[col] == 0.0])
        #    p1 = len(df[df[col] == 1.0])/notNull
        #    p2 = len(df[df[col] == 0.0])/notNull
         #   df[col] = df[col].fillna(pd.Series(np.random.choice([1.0,0.0], p=[p1, p2], size=len(df))))
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# Create a new feature that determines applicant income based on number of dependents 
def getIncomeByDep(ApplicantIncome,Dependents):
    if Dependents == 0:
        return ApplicantIncome
    elif Dependents == 1:
        return ApplicantIncome/2
    elif Dependents == 2:
        return ApplicantIncome/3
    else:
        return ApplicantIncome/4

# Create other new features  
def getFeatures(df):
    #log Loan
    df['logLoan'] = np.log(df.LoanAmount)

    # Total Income
    df['totalIncome'] = df.ApplicantIncome + df.CoapplicantIncome

    # Log transforming applicant income
    df['logApplicantIncome'] = np.log(df['ApplicantIncome'])

    df['LoanAmountTerm'] = df.LoanAmount/df.Loan_Amount_Term
    df['LoanbyIncome'] = df.totalIncome/df.LoanAmount
    
    return df

# return final categorical columns used
def getFinalCat(df):
    return df[['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area']]

# return final numerical columns used
def getFinalNum(df):
    return df[['Credit_History','Loan_Amount_Term','LoanAmount', 'totalIncome', 'ApplicantIncome', 'CoapplicantIncome','LoanAmountTerm','LoanbyIncome']]
       
class ToDenseTransformer():

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self