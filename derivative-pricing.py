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
from sklearn.feature_selection import SelectKBest
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
from pandas_datareader import data as web
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
    moneyness = np.clip(moneyness, 1e-5, None)  # Prevent log(0) or negative
    time_to_maturity = np.clip(time_to_maturity, 1e-5, None)  # Prevent division by zero

    d1 = (np.log(1/moneyness) + (risk_free_rate + 0.5 * np.square(option_val)) * time_to_maturity) / (option_val * np.sqrt(time_to_maturity))
    d2 = d1 - option_val * np.sqrt(time_to_maturity)

    N_d1 = 0.5 * (1 + np.math.erf(d1 / np.sqrt(2)))
    N_d2 = 0.5 * (1 + np.math.erf(d2 / np.sqrt(2)))

    return N_d1 - moneyness * np.exp(-risk_free_rate * time_to_maturity) * N_d2


N = 10000

Ks = 1+0.25*np.random.randn(N)
Ts = np.random.random(N)
Sigmas = np.array([option_vol_from_surface(k, t) for k, t in zip(Ks, Ts)])
Ps = np.array([call_option_price(k, t, sig) for k, t, sig in zip(Ks, Ts, Sigmas)])

Y = Ps
X = np.concatenate((Ks.reshape(-1, 1), Ts.reshape(-1, 1), Sigmas.reshape(-1, 1)), axis=1)
dataset = pd.DataFrame(np.concatenate([Y.reshape(-1, 1), X], axis=1), columns=['Price', 'Moneyness', 'Time', 'Vol'])

pyplot.figure(figsize=(15, 15))
scatter_matrix(dataset, figsize=(12, 12))
pyplot.show()

valid_indices = ~np.isnan(Ps) & np.all(np.isfinite(X), axis=1)
X = X[valid_indices]
Y = Ps[valid_indices]

bestfeatures = SelectKBest(score_func=f_regression, k='all')
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(['Moneyness', 'Time', 'Vol'])
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
featureScores.nlargest(10, 'Score').set_index('Specs')

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

models.append(('MLP', MLPRegressor()))

models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR', GradientBoostingRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

param_grid = {
    'hidden_layer_sizes': [(20,), (50,), (20, 20), (20, 30, 20)]
}
model = MLPRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(model, param_grid, cv=kfold, scoring=scoring)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, std, param))

model_tuned = MLPRegressor(hidden_layer_sizes=(20, 30, 20))
model_tuned.fit(X_train, Y_train)

predictions = model_tuned.predict(X_test)
print(mean_squared_error(Y_test, predictions))