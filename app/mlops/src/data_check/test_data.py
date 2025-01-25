import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = ['Sex', 'Age', 'Tenure', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD',
       'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF',
       'ActBal_OVD', 'ActBal_CC', 'ActBal_CL', 'VolumeCred', 'VolumeCred_CA',
       'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA',
       'VolumeDebCash_Card', 'VolumeDebCashless_Card',
       'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA',
       'TransactionsDebCash_Card', 'TransactionsDebCashless_Card',
       'TransactionsDeb_PaymentOrder', 'Sale_MF', 'Sale_CC', 'Sale_CL',
       'Revenue_MF', 'Revenue_CC', 'Revenue_CL']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_sex(data):

    known_names = ["M", "F"]

    sex = set(data['Sex'].unique())

    # Unordered check
    assert set(known_names) == set(sex)


def test_proper_range(data: pd.DataFrame):
    """
    Test proper range for Age and Tenure
    """
    idx = data['Age'].between(0, 110) & data['Tenure'].between(0, 1320)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['Sex'].value_counts().sort_index()
    dist2 = ref_data['Sex'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    """
    Test that the number of rows in the dataset is within a certain range
    """
    assert 100 < data.shape[0] < 1615

def test_age_range(data, min_age, max_age):
    """
    Test that the age range is within a certain range
    """
    assert min_age <= data['Age'].min()
    assert max_age >= data['Age'].max()

def test_tenure_range(data, min_tenure, max_tenure):
    """
    Test that the tenure range is within a certain range
    """
    assert min_tenure <= data['Tenure'].min()
    assert max_tenure >= data['Tenure'].max()