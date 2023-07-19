from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from folktables import ACSDataSource, adult_filter, BasicProblem
import pandas as pd
import numpy as np
import os

ACSIncome_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}


def get_acs_pums(dataset, name, states=None,  resample=False):
    data_path = 'data/sampled_{}.csv'.format(name)

    def resample_():
        data_source = ACSDataSource(
            survey_year='2018', horizon='1-Year', survey='person')
        ca_data = data_source.get_data(states=states,  download=True)

        # ca_features, ca_labels, _ = ACSIncome.df_to_pandas(
        #    ca_data, categories=ACSIncome_categories, dummies=True)
        ca_data = ca_data.sample(n=130000, random_state=1)
        print(len(ca_data))
        ca_data.to_csv(data_path, index=False)
        return ca_data

    if resample or os.path.isfile(data_path) is False:
        ca_data = resample_()
    else:
        ca_data = pd.read_csv(data_path, index_col=False)


def get_adult(rseed=0, states=None, sensitive_attrib='', resample=False, base_dir="./"):
    data_path = '{}data/sampled_new_adult.csv'.format(base_dir)

    def resample_newadult():
        data_source = ACSDataSource(
            survey_year='2018', horizon='1-Year', survey='person')
        ca_data = data_source.get_data(states=states,  download=True)

        # ca_features, ca_labels, _ = ACSIncome.df_to_pandas(
        #    ca_data, categories=ACSIncome_categories, dummies=True)
        ca_data = ca_data.sample(n=130000, random_state=1)
        print(len(ca_data))
        ca_data.to_csv(data_path, index=False)
        return ca_data
    ACSEmployment = BasicProblem(
        features=[
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            # 'SEX',
            'RAC1P',
        ],
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group='SEX',
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    if resample or os.path.isfile(data_path) is False:
        ca_data = resample_newadult()
    else:
        ca_data = pd.read_csv(data_path, index_col=False)
        
    X, y, s = ACSEmployment.df_to_pandas(
        ca_data, categories=ACSIncome_categories, dummies=True)
    s[s == 2] = 0 
    y = np.squeeze(y)
    s = np.squeeze(s)
    X = X.values
 

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        X, y, s, test_size=0.2, random_state=1, stratify=y)

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)
 
    return (X_d1, y_d1.values, s_d1.values), (X_d2, y_d2.values, s_d2.values)


def get_acs_employment(states=None, resample=False):
    ACSEmployment = BasicProblem(
        features=[
            'AGEP',
            'SCHL',
            'MAR',
            'RELP',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            # 'SEX',
            'RAC1P',
        ],
        target='ESR',
        target_transform=lambda x: x == 1,
        group='SEX',
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data_path = 'data/sampled_ac_employment.csv'

    def resample_():
        data_source = ACSDataSource(
            survey_year='2018', horizon='1-Year', survey='person')
        ca_data = data_source.get_data(states=states,  download=True)
 
        ca_data = ca_data.sample(n=130000, random_state=1)
        print(len(ca_data))
        ca_data.to_csv(data_path, index=False)
        return ca_data
    if resample or os.path.isfile(data_path) is False:
        ca_data = resample_()
    else:
        ca_data = pd.read_csv(data_path, index_col=False)
    X, y, s = ACSEmployment.df_to_pandas(
        ca_data, categories=ACSIncome_categories, dummies=True)
    s[s == 2] = 0
    y = y * 1 

    y = np.squeeze(y)
    s = np.squeeze(s)
    X = X.values
 

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        X, y, s, test_size=0.2, random_state=1, stratify=y)

    scaler1 = StandardScaler()
    X_d1 = scaler1.fit_transform(X_d1)
    scaler2 = StandardScaler()
    X_d2 = scaler2.fit_transform(X_d2)
 
    return (X_d1, y_d1.values, s_d1.values), (X_d2, y_d2.values, s_d2.values)


def get_old_adult(include_y_in_x=False, base_dir="./"):
    #
    def get_data(df, include_y_in_x=include_y_in_x):
        if 'gender_ Male' in df.columns:
            S = df['gender_ Male'].values
            X = df.drop('gender_ Male', axis=1)
        if not include_y_in_x:
            X = df.drop(['outcome_ >50K', 'gender_ Male'], axis=1).values
        else:
            X = df.values
            
        y = df['outcome_ >50K'].values

        return X, y, S

    data1 = pd.read_csv(
        '{}preprocessing/adult.data1.csv'.format(base_dir), index_col=False)
    data2 = pd.read_csv(
        '{}preprocessing/adult.data2.csv'.format(base_dir), index_col=False)
    data1 = data1.sample(frac=1)
    return get_data(data1), get_data(data2)


def get_compas_race(base_dir="./"):

    return get_compas(sensitive_attrib="African_American", base_dir=base_dir)


def get_compas(sensitive_attrib="Female", base_dir="./"):
    data = pd.read_csv("{}preprocessing/compas.csv".format(base_dir))

    X = data.drop(["Two_yr_Recidivism", sensitive_attrib], axis=1).values
    y = data["Two_yr_Recidivism"].values
    s = data[sensitive_attrib].values

    X_d1, X_d2, y_d1, y_d2, s_d1, s_d2 = train_test_split(
        X, y, s, test_size=0.20, random_state=1)

    return (X_d1, y_d1, s_d1), (X_d2, y_d2, s_d2)

 
