from datetime import datetime
from zipline.utils.factory import load_bars_from_yahoo
import pytz
import talib
import numpy as np

# Set the start and end dates
start = datetime(2000, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2016, 4, 30, 0, 0, 0, 0, pytz.utc)
index = {'SPX': '^GSPC'}
stock = None

time_period = 30     # Techinical Indicators time period
pred_window = 1      # prediction window in days
train_ratio = 0.7    # training records ratio
fold_number = 5      # How many folder

# Load data from Yahoo Finance
data = load_bars_from_yahoo(indexes=index, stocks=stock, start=start, end=end)
print data[index.keys()[0]].describe()


# Exploratory analysis
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

sname = index.keys()[0]
vol = data[sname][['volume']]
bars = data[sname][['open', 'high', 'low', 'close', 'price']]

ax = bars.plot(figsize=(8,6),  title='(a) 2001-2016 Historical Price of S&P 500 Index')
ax.set_ylabel("Value in US Dollar")
plt.savefig('sp500_price.pdf')
plt.figure()
ax = vol.plot(figsize=(8,6), title='(b) 2001-2016 Historical Volume of S&P 500 Index')
ax.set_ylabel("Numer of shares")
plt.savefig('sp500_volume.pdf')

# Calcuate the std of daily price variation
daily_var = bars['price'].iloc[1:].values-bars['price'].iloc[:-1].values
print np.std(daily_var)
# In[6]:

# Calcualte Techinical Indicators

# AD
ti_AD = talib.AD(np.array(bars['high']), np.array(bars['low']),                  np.array(bars['close']), np.array(vol['volume']))

# ADX
ti_ADX = talib.ADX(bars['high'].values, bars['low'].values,                    bars['close'].values, time_period)

# AROONOSC
ti_AROONOSC = talib.AROONOSC(bars['high'].values, bars['low'].values,                             time_period)

# ATR
ti_ATR = talib.ATR(bars['high'].values, bars['low'].values,                    bars['close'].values, time_period)

# AVGPRICE
ti_AVGPRICE = talib.AVGPRICE(bars['open'].values, bars['high'].values,                              bars['low'].values, bars['close'].values)

# CMO
ti_CMO = talib.CMO(bars['price'].values, time_period)

# DEMA
ti_DEMA = talib.DEMA(bars['price'].values, time_period)

# DX
ti_DX = talib.DX(bars['high'].values, bars['low'].values,                  bars['close'].values, time_period)

# EMA
ti_EMA = talib.EMA(bars['price'].values, time_period)

# MACDEXT
ti_MACD, ti_MACDSIGNAL, ti_MACDHIST = talib.MACDFIX(bars['price'].values)

# MEDPRICE
ti_MEDPRICE = talib.MEDPRICE(bars['high'].values, bars['low'].values)

# MFI
ti_MFI = talib.MFI(bars['high'].values, bars['low'].values,\
    bars['close'].values, vol['volume'].values, time_period)

# MINUS_DI
ti_MINUS_DI = talib.MINUS_DI(bars['high'].values, bars['low'].values,\
 bars['close'].values, time_period)

# MOM
ti_MOM = talib.MOM(bars['price'].values, time_period)

# OBV
ti_OBV = talib.OBV(bars['price'].values, vol['volume'].values)

# PLUS_DI
ti_PLUS_DI = talib.PLUS_DI(bars['high'].values, bars['low'].values,\
 bars['close'].values, time_period)

# ROCP
ti_ROCP = talib.ROCP(bars['price'].values, time_period)

# ROCR
ti_ROCR = talib.ROCR(bars['price'].values, time_period)

# RSI
ti_RSI = talib.RSI(bars['price'].values, time_period)

# SMA
ti_SMA = talib.SMA(bars['price'].values, time_period)

# STOCHRSI
ti_FASTK, ti_FASTD = talib.STOCHRSI(bars['price'].values, time_period)

# TRANGE
ti_TRANGE = talib.TRANGE(bars['high'].values, bars['low'].values,\
                         bars['close'].values)

# TYPPRICE
ti_TYPPRICE = talib.TYPPRICE(bars['high'].values, bars['low'].values,\
                             bars['close'].values)

# WCLPRICE
ti_WCLPRICE = talib.WCLPRICE(bars['high'].values, bars['low'].values,\
                             bars['close'].values)

# WILLR
ti_WILLR = talib.WILLR(bars['high'].values, bars['low'].values,\
                       bars['close'].values, time_period)

# WMA
ti_WMA = talib.WMA(bars['price'].values, time_period)

# Combine to dataframe
ti = pd.DataFrame({
        'AD': ti_AD,
        'ADX': ti_ADX,
        'AROONOSC': ti_AROONOSC,
        'ATR': ti_ATR,
        'AVGPRICE': ti_AVGPRICE,
        'CMO': ti_CMO,
        'DEMA': ti_DEMA,
        'DX': ti_DX,
        'EMA': ti_EMA,
        'MACD': ti_MACD,
        'MACDSIGNAL': ti_MACDSIGNAL,
        'MACDHIST': ti_MACDHIST,
        'MEDPRICE': ti_MEDPRICE,
        'MFI': ti_MFI,
        'MINUS_DI': ti_MINUS_DI,
        'MOM': ti_MOM,
        'OBV': ti_OBV,
        'PLUS_DI': ti_PLUS_DI,
        'ROCP': ti_ROCP,
        'ROCR': ti_ROCR,
        'RSI': ti_RSI,
        'SMA': ti_SMA,
        'FASTK': ti_FASTK,
        'FASTD': ti_FASTD,
        'TRANGE': ti_TRANGE,
        'TYPPRICE': ti_TYPPRICE,
        'WCLPRICE': ti_WCLPRICE,
        'WMA': ti_WMA,
        'WILLR': ti_WILLR
    }, index=bars.index)

# Transform indicators to [-1, 1]
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

max_abs_scaler = preprocessing.MaxAbsScaler()
ti_nonan = ti.iloc[2*time_period:, :]
ti_scaled = ti_nonan.apply(max_abs_scaler.fit_transform)

# For visulization purpose, we only plot five indicators.
plt_data = ti_scaled[['AD', 'AVGPRICE', 'DEMA', 'OBV', 'RSI']]
ax = plt_data.plot(figsize=(10, 8), title='Five Scaled Technical Indicators'
              )
ax.legend(loc='best')
ax.set_ylabel('Normalized Value of Technical Indicators')
plt.savefig('indicators.pdf')


# In[7]:

# Split model data into training and testing

Y = bars['price'][2*time_period+pred_window:]
X = ti_scaled.iloc[:len(Y), :]

# Split data into train and test
train_len = int(len(Y)*train_ratio)
Y_train = Y[0:train_len]
Y_test = Y[train_len:]
X_train = X.iloc[:train_len, :]
X_test = X.iloc[train_len:, :]

# Linear Regression model
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

# Use statmodels
X_cons = sm.add_constant(X_train)
clf = sm.OLS(Y_train.values, X_cons.values).fit()
MAE_LR_train = mean_absolute_error(Y_train.values, clf.predict(X_cons.values))

pred_LR = clf.predict(sm.add_constant(X_test).values)
MAE_LR_test = mean_absolute_error(Y_test.values, pred_LR)
print 'The MAE of Linear Regression model on testing dataset is {}'.format(MAE_LR_test)

# Show residual of trainning data: should be randomly distributed without certain pattern
plt.figure(figsize=(9,6))
plt.scatter(range(0, len(Y_train)), Y_train.values - clf.predict(X_cons.values))
plt.xlabel('Time Index')
plt.ylabel('Residual')
plt.xlim([0, 2700])
plt.title('Residual of Linear Regression Model on Training Dataset')
plt.savefig('LinearRegression_residual.pdf')

# Goodness-of-fit
from sklearn.metrics import r2_score

R2 = r2_score(Y_train.values, clf.predict(X_cons.values))
print 'Coefficient of determination is', R2
print clf.summary()


# In[15]:

# Define cross-validation function for time series prediction
def performTimeSeriesCV(X_train, y_train, number_folds, model_function):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the ML algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.
    Reference: http://francescopochetti.com/pythonic-cross-validation-time-series-pandas-scikit-learn/
    """

    # k is the size of each fold. It is computed dividing the number of
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    #print 'Size of each fold: ', k

    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    accuracies = np.zeros(number_folds-1)
    train_accuracy = np.zeros(number_folds-1)
    # model_dict = {}
    # loop from the first 2 folds to the total number of folds
    for i in range(2, number_folds + 1):
        #print ''

        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second),
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i

        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        #print 'Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i)

        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # ....
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        #print 'Size of train + test: ', X.shape # the size of the dataframe is going to be k*i

        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))

        # folds used to train the model
        X_trainFolds = X[:index]
        y_trainFolds = y[:index]

        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]

        # i starts from 2 so the zeroth element in accuracies array is i-2.  y
        model_fit = model_function.fit(X_trainFolds, y_trainFolds)
        model_pred = model_fit.predict(X_testFold)
        accuracies[i-2] = mean_absolute_error(y_testFold, model_pred)
        train_accuracy[i-2] = mean_absolute_error(y_trainFolds, model_fit.predict(X_trainFolds))
        #print 'Mean Absolute Error on fold ' + str(i) + ': ', accuracies[i-2]

    # the function returns the mean of the accuracy on the n-1 folds
    return (accuracies.mean(), train_accuracy.mean())


# In[66]:

# LASSO regression

parameter = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]   # for alpha
tmp_test = []
tmp_train = []
for n in parameter:
    clf = linear_model.Lasso(alpha=n)
    tmp1, tmp2 = performTimeSeriesCV(X_train, Y_train, fold_number, clf)
    tmp_test.append(tmp1)
    tmp_train.append(tmp2)
clf = linear_model.Lasso(alpha=parameter[np.argmin(tmp_test)]).fit(X_train, Y_train)
MAE_Lasso_train = mean_absolute_error(Y_train, clf.predict(X_train))
pred_Lasso = clf.predict(X_test)
MAE_Lasso_test = mean_absolute_error(Y_test, pred_Lasso)
print 'The MAE of Lasso Regression model on testing dataset is {}'.format(MAE_Lasso_test)

# plot MAE for cross-validation
plt.figure(figsize=(8, 5))
plt.plot(parameter, tmp_test, '-bo')
plt.plot(parameter, tmp_train, '-r*')
plt.xlabel('Alpha')
plt.ylabel('MAE for Cross-valiation')
plt.title('MAE for Cross-validation Dataset of Lasso')
plt.legend(['Sub-validation', 'Sub-training'], loc=4)
plt.xlim([-1, 105])
plt.savefig('Lasso_CV.pdf')


# In[68]:

# SVM regression
from sklearn import svm

#para1 = ['linear', 'poly', 'rbf']           # for kernel
para2 = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]             # for C
tmp = np.zeros(len(para2))    # for MAE
tmp_test = []
tmp_train = []
for i in range(len(para2)):
    clf = svm.SVR(C=para2[i])
    tmp1, tmp2 = performTimeSeriesCV(X_train, Y_train, fold_number, clf)
    tmp_test.append(tmp1)
    tmp_train.append(tmp2)
clf = svm.SVR(C=para2[np.argmin(tmp_test)]).fit(X_train, Y_train)
MAE_SVM_train = mean_absolute_error(Y_train, clf.predict(X_train))
pred_SVM = clf.predict(X_test)
MAE_SVM_test = mean_absolute_error(Y_test, pred_SVM)
print 'The MAE of SVM Regression model on testing dataset is {}'.format(MAE_SVM_test)

# plot MAE for cross-validation
plt.figure(figsize=(8, 5))
plt.plot(para2, tmp_test, '-bo')
plt.plot(para2, tmp_train, '-r*')
plt.xlabel('C')
plt.ylabel('MAE for Cross-valiation')
plt.title('MAE for Cross-validation Dataset of SVM')
plt.legend(['Sub-validation', 'Sub-training'], loc=1)
plt.xlim([-1, 105])
plt.savefig('SVM_CV.pdf')


# In[89]:

# Combine all model results into a dataframe
pred_test = pd.DataFrame({
        'True': Y_test.values,
        'Linear Regression': pred_LR,
        'Lasso': pred_Lasso,
        'SVM': pred_SVM
    }, index=Y_test.index)
# plot model predictions and true values
ax = pred_test.plot(figsize=(8, 5), title='Prediction comparison for testing dataset')
ax.legend(loc=4)
ax.set_ylabel("Price in US dollars")
plt.savefig('Test_validation.pdf')


# In[90]:

pred_test = pd.DataFrame({
        'True': Y_test.values[-100:],
        'Linear Regression': pred_LR[-100:],
        'Lasso': pred_Lasso[-100:],
    }, index=Y_test.index[-100:])
# plot model predictions and true values
ax = pred_test.plot(figsize=(8, 5), title='Subsmapled prediction comparison for testing dataset')
ax.legend(loc=4)
ax.set_ylabel("Price in US dollars")
plt.savefig('Test_validation_sub.pdf')


# In[80]:

# Combine model errors into a dataframe
MAE_train = pd.DataFrame({
        'MAE train Linear Regression': MAE_LR_train,
        'MAE train Lasso': MAE_Lasso_train,
        'MAE train SVM': MAE_SVM_train
    }, index=range(1))
ax = MAE_train.plot(kind='bar',figsize=(8, 5),
               title='MAE of train data comparison')
ax.legend(loc=2)
plt.xlabel('Model')
plt.ylabel('MAE in US dollar')
plt.ylim([0, 15])
plt.savefig('MAE_train.pdf')


MAE_test = pd.DataFrame({
        'MAE test Linear Regression': MAE_LR_test,
        'MAE test Lasso': MAE_Lasso_test,
        'MAE test SVM': MAE_SVM_test
    }, index=range(1))
MAE_test.plot(kind='bar', figsize=(8, 5),
              title='MAE of test data comparison'
              ).legend(loc=2)
plt.xlabel('Model')
plt.ylabel('MAE in US dollar')
plt.ylim([0, 55])
plt.savefig('MAE_test.pdf')


# In[84]:

print MAE_test
print MAE_train

