# Introduction to Data Science with Python
# MTA - Spring 2021-2022.
# Final Home Exercise.


import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data


class DataPreprocessor(object):

    """
    This class is a mandatory API. More about its structure - few lines below.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages.
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non descriptive columns
    3 ...

    The test data is unavailable when building the ML pipeline, thus it is necessary to determine the
    preprocessing steps and values on the train set and apply them on the test set.


    *** Mandatory structure ***
    The ***fields*** are ***not*** mandatory
    The ***methods***  - "fit" and "transform" - are ***required***.

    You're more than welcome to use sklearn.pipeline for the "heavy lifting" of the preprocessing tasks, but it is not an obligation.
    Any class that implements the methods "fit" and "transform", with the required inputs & outps will be accepted.
    Even if "fit" performs no taksks at all.
    """

    def __init__(self):
      self.transformer: Pipeline = None

    def fit(self, dataset_df):

        """
        Input:
        dataset_df: the training data loaded from the csv file as a dataframe containing only the features
        (not the target - see the main function).

        Output:
        None

        Functionality:
        Based on all the provided training data, this method learns with which values to fill the NA's,
        how to scale the features, how to encode categorical variables etc.

        *** This method will be called exactly once during evaluation. See the main section for details ***


        Note that implementation below is a boilerplate code which performs very basic categorical and numerical fields
        preprocessing.

        """

        # This section can be hard-coded
        # There are more - what else?
        numerical_columns = ['Transportation expense', 'Height','Son','Residence Distance','Service time', 'Weight','Pet']
        categorical_columns = list(set(dataset_df.columns) - set(numerical_columns))

        self.Residence_Distance=dataset_df['Residence Distance'].mean()
        self.Service_time = dataset_df['Service time'].mean()
        self.Education = dataset_df['Education'].value_counts().idxmax()
        self.Son = dataset_df['Son'].mean()
        self.Smoker = dataset_df['Smoker'].value_counts().idxmax()
        self.Pet = dataset_df['Pet'].median()
        self.Weight = dataset_df['Weight'].mean()
        self.Height = dataset_df['Height'].mean()
        self.Season = dataset_df['Season'].value_counts().idxmax()
        self.Drinker = dataset_df['Drinker'].value_counts().idxmax()
        self.Age_Group = dataset_df['Age Group'].value_counts().idxmax()

        # Handling Numerical Fields
        num_pipeline = Pipeline([
          ('imputer', SimpleImputer(strategy="mean")),
            ('std_scaler', StandardScaler()),
        ])

        # Handling Categorical Fields
        categorical_transformer = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')

        cat_pip =Pipeline([
          ('imputer', SimpleImputer(strategy="most_frequent"))
        ])

        cat_pipeline = Pipeline([
          ('1hot', categorical_transformer)
        ])

        preprocessor = ColumnTransformer(
          transformers=[
            ("dropId", 'drop', 'ID'),
            ("num", num_pipeline, numerical_columns),
            ("cat1", cat_pip, categorical_columns),
            ("cat", cat_pipeline, categorical_columns),
          ])

        self.transformer = Pipeline(steps=[
          ("preprocessor", preprocessor)
        ])

        self.transformer.fit(dataset_df)

    def transform(self, df):

        """
        Input:
        df:  *any* data similarly structured to the train data (dataset_df input of "fit")

        Output:
        A processed dataframe or ndarray containing only the input features (X).

        It should maintain the same row order as the input.
        Note that the labels vector (y) should not exist in the returned ndarray object or dataframe.


        Functionality:
        Based on the information learned in the "fit" method, apply the required transformations to the passed data (df)

        """

        df = df.drop(['ID'], axis=1)

        df['Residence Distance'].fillna(self.Residence_Distance, inplace=True)
        df['Service time'].fillna(self.Service_time, inplace=True)
        df['Son'].fillna(self.Son, inplace=True)
        df['Smoker'].fillna(self.Smoker, inplace=True)
        df['Pet'].fillna(self.Pet, inplace=True)
        df['Weight'].fillna(self.Weight, inplace=True)
        df['Height'].fillna(self.Height, inplace=True)
        df['Season'].fillna(self.Season, inplace=True)
        df['Drinker'].fillna(self.Drinker, inplace=True)
        df['Age Group'].fillna(self.Age_Group, inplace=True)
        df['Education'].fillna(self.Education, inplace=True)

        df = pd.get_dummies(df, prefix=["Season", "Age Group"], columns=["Season", "Age Group"])
        df["Bin_Smoker"] = np.where(df['Smoker'] == 'Yes', 1, 0)
        df["Bin_Drinker"] = np.where(df['Drinker'] == 'Yes', 1, 0)
        df = df.drop(["Smoker", "Drinker"], axis=1)
        df['Weight'] = df['Weight']/df['Weight'].max()

        df['education'] = df['Education'].map({'High school': 0, 'Postgraduate': 1, 'Graduate': 1, 'Phd': 1})
        df = df.drop(['Education'], axis=1)

        for col in ['Residence Distance']:
            df['log1p' + col] = np.log1p(df[col])

        df = df.drop(['Residence Distance', 'Service time', 'Height', 'Transportation expense', 'Son'], axis=1)
        return df


def train_model(processed_X, y):
    """
    This function gets the data after the pre-processing stage  - after running DataPreprocessor.transform on it,
    a vector of labels, and returns a trained model.

    Input:
    processed_X (ndarray or dataframe): the data after the pre-processing stage
    y: a vector of labels

    Output:
    model: an object with a "predict" method, which accepts the ***pre-processed*** data and outputs the prediction


    """

    model = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=8)
    model.fit(processed_X, y)

    return model
