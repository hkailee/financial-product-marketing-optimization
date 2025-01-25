"""
Feature selection
"""

from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from kneed import KneeLocator
from sklearn.feature_selection import mutual_info_regression

shap.initjs() #initialize javascript to enable visualizations


# To obtain mutual information scores
def make_mi_scores_regression(
        X: pd.DataFrame, 
        y: pd.DataFrame) -> pd.Series:
    """to obtain mutual information scores for regression

    Args:
        X (pd.DataFrame): features
        y (pd.DataFrame): label

    Returns:
        pd.Series: mutual information scores
    """    
    X = X.copy()
    for colname in X.select_dtypes(["object", "int64"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(
        scores: pd.Series,
        n_top_features: int=20) -> Any:
    """plot mutual information scores for all features

    Args:
        scores (pd.Series): mutual information scores

    Returns:
        Any: output a figure in jupyter notebook
    """    
    scores = scores.sort_values(ascending=True)[-n_top_features:]
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.figure(figsize=(6, len(scores)*0.25))
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
def drop_uninformative(
        df: pd.DataFrame,                
        mi_scores: pd.Series, 
        threshold: float=0.0) -> pd.DataFrame:
    """to drop uninformative features based on mutual information score

    Args:
        df (pd.DataFrame): feature dataframe
        mi_scores (pd.Series): mutual information scores
        threshold (float, optional): minimal mutual information score to drop features. 
                                        Defaults to 0.0.

    Returns:
        pd.DataFrame: features selected with mutual information score
    """    
    return df.loc[:, mi_scores > threshold]



def find_optimal_elbow_features(df: pd.DataFrame, column: str, sensitivity: float = 1.0) -> int:
    """Find top n features based on elbow point sensitivity

    Args:
        df (pd.DataFrame): feature importance values with feature names as index and weighing methods as columns 
        column (str): weighing method of choice
        sensitivity (float, optional): . Defaults to 1.0.

    Returns:
        int: top n optimal features based on elbow sensitivity setting
    """   
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.sort_values(by=[column], ascending=False, inplace=True)
    idx_kn = KneeLocator(range(1, len(df)+1), df[column], 
                         curve='convex', 
                         direction='decreasing',
                         S=sensitivity).knee
    
    return idx_kn


# Shapley Additive Explanation - Coaliation game theory to get more faithful importance scores 
# for features
class tree_shap_collector():
    '''
    The tree_shap_collector class is used to obtain tree shap importance scores for a list of features. The class has the following methods and attributes:
    Attributes:
        X_train: A pandas DataFrame containing the features for the training set.
        y_train: A pandas Series containing the target values for the training set.
        X_test: A pandas DataFrame containing the features for the testing set.
        model: An instance of a scikit-learn regression model.
        model_name: A string indicating the name of the regression model.
        features: A list of strings containing the names of the features.
    Methods:

    __init__(self, X_train, y_train, X_test, model, model_name): Initializes the class instance by calculating the shap values 
    and global importances for each feature. The TreeExplainer class from the shap library is used to calculate shap values. 
    The global importances for each feature are obtained by taking the average of the absolute shap values for each feature.
    explainer_summary_plot(self, max_display: int = 20): Generates a summary plot of the shap values for each feature. 
    The summary plot shows the most important features and their corresponding shap values in a bar or dot plot.
    explainer_dependence_plot(self, top_n: int = 2): Generates a partial dependence plot with shap values for the most important features. 
    The partial dependence plot shows the relationship between the feature and the target variable, while also showing the corresponding shap values.
    model_shap_importances(self): Returns a pandas DataFrame containing the global importances for each feature. The features are sorted in descending order of importance.
    The tree_shap_collector class uses the shap library to calculate shap values for a given regression model. 
    The class calculates global importances for each feature and generates summary and partial dependence plots 
    to help visualize the importance of each feature. The class provides a method to return 
    a pandas DataFrame containing the global importances for each feature, sorted by importance.    
    '''
    def __init__(self, X_train, y_train, X_test, model, model_name):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model.fit(self.X_train, self.y_train)
        self.model_name = model_name
        self.features = X_train.columns

        # calculate shap value
        self.explainer = shap.TreeExplainer(self.model, check_additivity=False, feature_perturbation='interventional')
        self.shap_values = self.explainer.shap_values(self.X_test)

        # Shap importance
        # Get mean absolute SHAP values for each feature
        shap_importance = np.mean(np.abs(self.shap_values[0]), axis=0)  # Assuming a single class for simplicity

        # Get feature names and sort by importance
        feature_names = X_train.columns
        self._features_ranked = pd.DataFrame({'feature_name': feature_names, 'importance': shap_importance})
        self._features_ranked.sort_values('importance', ascending=False, inplace=True)

    def explainer_summary_plot(self, max_display: int = 20) -> Any:
        """summary in both bar and dot plots

        Args:
            max_display (int, optional): maximum number of most important features . Defaults to 20.

        Returns:
            Any: bar and dot plots
        """        
        shap.summary_plot(self.shap_values, self.X_test, max_display=max_display, plot_type="bar")
        shap.summary_plot(self.shap_values, self.X_test, max_display=max_display, plot_type="dot")

    def explainer_dependence_plot(self, top_n: int = 2) -> Any:
        """partial dependence plot with shap

        Args:
            top_n (int, optional): maximum number of most important features. Defaults to 2.

        Returns:
            Any: partial dependence plots
        """        
        n_subfigures = top_n
        n_cols = ceil(n_subfigures**0.5)
        n_rows = int(ceil(n_subfigures / n_cols))

        fig, axes = plt.subplots(nrows=n_cols, ncols=n_rows, figsize=(20, 14))
        axes = axes.ravel()

        for fig_count, feat in enumerate(self._features_ranked[:top_n].index):

            shap.dependence_plot(feat, 
                                 self.shap_values, 
                                 self.X_test, 
                                 interaction_index=None, 
                                 ax=axes[fig_count], 
                                 show=False)

        plt.tight_layout()
        plt.show()

    @property
    def model_shap_importances(self) -> pd.DataFrame:
        return self._features_ranked



# Shapley Additive Explanation - Coaliation game theory to get more faithful importance scores 
# for features
class kernel_shap_collector():
    '''
    The `kernel_shap_collector` class is used to obtain kernel shap importance scores for a list of features. 
    The class has the following methods and attributes:

    Attributes:
    - `X_train`: A pandas DataFrame containing the features for the training set.
    - `y_train`: A pandas Series containing the target values for the training set.
    - `X_test`: A pandas DataFrame containing the features for the testing set.
    - `model`: An instance of a scikit-learn regression model.
    - `model_name`: A string indicating the name of the regression model.
    - `features`: A list of strings containing the names of the features.

    Methods:
    - `__init__(self, X_train, y_train, X_test, model, model_name)`: Initializes the class instance by calculating 
    the shap values and global importances for each feature. The `KernelExplainer` class from the `shap` library 
    is used to calculate kernel shap values. The global importances for each feature are obtained by 
    taking the average of the absolute shap values for each feature.
    - `explainer_summary_plot(self, max_display: int = 20)`: Generates a summary plot of the shap values for each feature. 
    The summary plot shows the most important features and their corresponding shap values in a bar or dot plot.
    - `explainer_dependence_plot(self, top_n: int = 2)`: Generates a partial dependence plot with shap values 
    for the most important features. The partial dependence plot shows the relationship between 
    the feature and the target variable, while also showing the corresponding shap values.
    - `model_shap_importances(self)`: Returns a pandas DataFrame containing the global importances
    for each feature. The features are sorted in descending order of importance.

    The `kernel_shap_collector` class uses the `shap` library to calculate kernel shap values for a given regression model. 
    The class calculates global importances for each feature and generates summary and partial dependence plots 
    to help visualize the importance of each feature. The class provides a method to return 
    a pandas DataFrame containing the global importances for each feature, sorted by importance.
    '''
    def __init__(self, X_train, y_train, X_test, model, model_name):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model.fit(self.X_train, self.y_train)
        self.model_name = model_name
        self.features = X_train.columns

        # calculate shap value
        # summarize the background as 100 samples
        # background = shap.sample(self.X_train, 100)
        self.explainer = shap.KernelExplainer(self.model.predict, self.X_train) #background
        self.shap_values = self.explainer.shap_values(self.X_test)

        # Shap importance
        # Get mean absolute SHAP values for each feature
        shap_importance = np.mean(np.abs(self.shap_values[0]), axis=0)  # Assuming a single class for simplicity

        # Get feature names and sort by importance
        feature_names = X_train.columns
        self._features_ranked = pd.DataFrame({'feature_name': feature_names, 'importance': shap_importance})
        self._features_ranked.sort_values('importance', ascending=False, inplace=True)
        
    def explainer_summary_plot(self, max_display: int = 20) -> Any:
        """summary in both bar and dot plots

        Args:
            max_display (int, optional): maximum number of most important features . Defaults to 20.

        Returns:
            Any: bar and dot plots
        """        
        shap.summary_plot(self.shap_values, self.X_test, max_display=max_display, plot_type="bar")
        shap.summary_plot(self.shap_values, self.X_test, max_display=max_display, plot_type="dot")

    def explainer_dependence_plot(self, top_n: int = 2) -> Any:
        """partial dependence plot with shap

        Args:
            top_n (int, optional): maximum number of most important features. Defaults to 2.

        Returns:
            Any: partial dependence plots
        """        
        n_subfigures = top_n
        n_cols = ceil(n_subfigures**0.5)
        n_rows = int(ceil(n_subfigures / n_cols))

        fig, axes = plt.subplots(nrows=n_cols, ncols=n_rows, figsize=(20, 14))
        axes = axes.ravel()

        for fig_count, feat in enumerate(self._features_ranked[:top_n].index):

            shap.dependence_plot(feat, 
                                 self.shap_values, 
                                 self.X_test, 
                                 interaction_index=None, 
                                 ax=axes[fig_count], 
                                 show=False)

        plt.tight_layout()
        plt.show()

    @property
    def model_shap_importances(self) -> pd.DataFrame:
        '''
        Property that returns the SHAP importances of the model features.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the SHAP importances of the model features.

        '''        
        return self._features_ranked
