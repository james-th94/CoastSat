# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 by JT
FILENAME: Time Series Analysis of Shoreline Position 
"""

# %% BASIC PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose 
import statsmodels.tsa.stattools as stt
import statsmodels.graphics.tsaplots as tsa_graphics
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.api import qqplot
import pylab
from datetime import datetime as dt

# %% USER INPUTS
working_dir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/CoastSat/data/WSE_landsat_1980-01-01_2022-12-31"
data_file = "transect_time_series_edit.csv"
index_column = 1
transect_of_interest = "Transect other_end"
transect_of_interest2 = 'Transect near_WSE'
event_date = '1/01/2021' # m/d/y
first_date = '9/16/1987' # m/d/y

# Plotting
figname = 'Shoreline_3monthlymean' + '_' + transect_of_interest + '.png'
figtitle = 'Shoreline Position Time Series in the lee of the Uniwave200 WEC'
figyaxis = 'Shoreline Position (m)'
figxaxis = 'Time (Year)'

year = dt.today().year
years = 36
x_ticks = [year - years + i + 1 for i in range(years - 1)]
# x_ticks = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
DPI = 600

# Change dir
os.chdir(working_dir)

# %% LOAD DATA AND VIEW DATAFRAME

df = pd.read_csv(data_file, index_col=index_column) # read csv data into dateframe, df, with index set to dates
df.index = pd.to_datetime(df.index) # convert dates to datetime objects in the dataframe
sns.pairplot(df)

# %% PRE-PROCESSING
data = df[transect_of_interest] # create data series of one column
data = data.resample("3MS").mean()
data = data.truncate(before = first_date)
data_ticks = data.truncate(before = first_date).resample("1YS").mean()

# %% 
data2 = df[transect_of_interest2] # create data series of one column
data2 = data2.resample("3MS").mean()
data2 = data2.truncate(before = first_date)

# %% SET TRAIN AND TEST SERIES

train = data.truncate(after=event_date) # create train data for pre-event data
test = data.truncate(before=event_date) # create test data for post-event data


# %% LOWESS MOVING AVERAGE FIT

y = data.values
y2 = data2.values
x = np.linspace(0, stop = len(y)-1, num = len(y))

smoothed = sm.nonparametric.lowess(exog=x, endog=y, frac=0.09, missing = 'drop')
smoothed2 = sm.nonparametric.lowess(exog=x, endog=y2, frac=0.09, missing = 'drop')

# Plot the fit line
fig, ax = pylab.subplots()

ax.scatter(x, y, color = 'blue', s = 5)
ax.scatter(x, y2, color = 'orange', s = 5)
ax.plot(smoothed[:, 0], smoothed[:, 1], c='blue')
ax.plot(smoothed2[:, 0], smoothed2[:, 1], color = 'orange')
ax.axvline(x=len(y) - 4, ymin=0.1, ymax=0.9, color =  'grey')

ax.set_xticklabels([1988, 1993, 1998, 2003, 2008, 2013, 2018])
ax.set_xlabel('Year')
ax.set_ylabel('Distance (m)')
ax.set_title('Shoreline Change near WEC (orange) and other end (blue) using Landsat')

pylab.autoscale(enable=True, axis="x", tight=True)

plt.savefig(figname, dpi = DPI, bbox_inches = 'tight')
print("Change in shoreline since the WEC was installed is approximately {} metres".format(smoothed[-1][1] - smoothed[-5][1]))

# %% TESTING STATIONALITY

p_value_train = stt.adfuller(train.interpolate(), regression= 'ct')[1]
p_value_test = stt.adfuller(test, regression = 'ct')[1] 
if p_value_train < 0.05:
    print("Training data is stationary. Check for seasonal trend only.")
if p_value_test < 0.05:
    print("Testing data is stationary. Check for seasonal trend only.")

# %% LINEAR REGRESSION (STATS MODELS - Train)

y = train.values
x = np.linspace(0, stop = len(y), num = len(y))
x = sm.add_constant(x)

model_train = sm.OLS(y, x, missing = 'drop').fit()
predictions_train = model_train.predict(x)
results_summary_train = model_train.summary()

conf_int = model_train.conf_int(alpha=0.05)
bottom_train = np.multiply(x[:,1],conf_int[1][0]) + conf_int[0][0]
top_train = np.multiply(x[:,1],conf_int[1][1]) + conf_int[0][1]

intercept_train = model_train.params[0]
coef_train = model_train.params[1]
textstr_train = 'y = {}x + {}'.format(coef_train, intercept_train)

# %% LINEAR REGRESSION (STATS MODELS - Test)

y = test.values
x = np.linspace(0, stop = len(y), num = len(y))
x = sm.add_constant(x)

model = sm.OLS(y, x, missing = 'drop').fit()
predictions = model.predict(x)
results_summary = model.summary()

conf_int = model.conf_int(alpha=0.05)
bottom = np.multiply(x[:,1],conf_int[1][0]) + conf_int[0][0]
top = np.multiply(x[:,1],conf_int[1][1]) + conf_int[0][1]

intercept = model.params[0]
coef = model.params[1]

textstr = 'y = {}x + {}'.format(round(np.multiply(coef,12),4), round(intercept,4))

# %% PLOTTING DATA


# Plot data and regression formula
plt.plot(train.index.values, train.values, 'o', markersize = 4, label = 'pre-WEC', color = 'C0')
plt.plot(test.index.values, test.values, 'o', markersize = 4, color = 'C2') #label = 'post-WEC; {}'.format(textstr), 

plt.plot(test.index.values, predictions, color = 'C2')
# plt.plot(test.index.values, top, color = 'C2')
# plt.plot(test.index.values, bottom, color = 'C2')
# plt.fill_between(test.index.values, bottom, top, alpha=0.3, color='C2')

# Change plot preferences
plt.grid(linestyle = '--')
plt.legend()
plt.tight_layout()

# Plot titles
# plt.title(figtitle)
plt.ylabel(figyaxis)
plt.xlabel(figxaxis)
plt.xticks(data_ticks.index.values, x_ticks)
# Show plot
#plt.show()
# plt.savefig(figname, dpi = DPI, bbox_inches = 'tight')

# %% SUMMARY STATS

change_over_14_months = np.multiply(14, np.array([conf_int[0][1], coef, conf_int[1][1]]))

# %% PERIOD DECOMPOSITION FOR TRAINING SET
train = train.interpolate()
decompose = seasonal_decompose(train, model = 'additive', period = 12)
train_trend = decompose.trend
train_seasonal = decompose.seasonal
train_residual = decompose.resid

decompose.plot()

# %% Is there a relationship between each end of the beach?
for i in range(len(data) - 1):
    plt.scatter(data[i], data2[i+1], color = 'blue')
