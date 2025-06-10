# Function and modules for the supervised learning models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

# Function and modules for data analysis and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import SelectKBest
from sklearn.feature_selection import f_regression, chi2

# Function and modlues for deep learning models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM

# Function and modules for time series models
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# Function and modules for data preparation and visualization
import pandas as pd
import numpy as np
from matplotlib import pyplot 
import pandas_datareader.web as web
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf

true_alpha = 0.1
true_beta = 0.1
true_sigma0 = 0.2

risk_free_rate = 0.05

def option_vol_from_surface(moneyness, time_to_maturity):
    return true_sigma0 + true_alpha * time_to_maturity + true_beta * np.square(moneyness - 1)

def call_option_price(moneyness, time_to_maturity, option_val):
    d1 = (np.log(1/moneyness) + (risk_free_rate+np.square(option_vol))*time_to_maturity)/(option_val*np.sqrt(time_to_maturity))
    d2 = (np.log(1/moneyness)+(risk_free_rate-np.square(time_to_maturity))*time_to_maturity) / (option_val*np.sqrt(time_to_maturity))
    N_d1 = np.linalg.norm(d1)
    N_d2 = np.linalg.norm(d2)
    return N_d1 - moneyness * np.exp(-risk_free_rate*time_to_maturity) * N_d2