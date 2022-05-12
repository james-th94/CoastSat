# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 05:19:33 2022
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
from scipy.stats import pearsonr

# %% USER INPUTS
working_dir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/CoastSat/data/WSE_2015-06-23_2022-03-16"
data_file = "transect_time_series_tide_corrected.csv"
index_column = 0
transect_of_interest = "Transect 66"
transect_of_interest2 = "Transect 33"
transect_of_interest3 = "Transect 4"
event_date = '1/01/2021' # m/d/y
first_date = '10/01/2016' # m/d/y

# Plotting
figname = 'Shoreline_monthlymean.png'
figtitle = 'Shoreline Position Time Series in the lee of the Uniwave200 WEC'
figyaxis = 'Shoreline Position (m)'
figxaxis = 'Time (Year)'
x_ticks = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
DPI = 600

# %% LOAD DATA AND PRE-PROCESSING

os.chdir(working_dir)
df = pd.read_csv(data_file, index_col=index_column) # read csv data into dateframe, df, with index set to dates
df.index = pd.to_datetime(df.index) # convert dates to datetime objects in the dataframe
data = df[transect_of_interest] # create data series of one column
data = data.resample("1MS").mean()
data = data.truncate(before = first_date)
data_ticks = data.truncate(before = first_date).resample("1YS").mean()

data2 = df[transect_of_interest2] # create data series of one column
data2 = data2.resample("1MS").mean()
data2 = data2.truncate(before = first_date)

data3 = df[transect_of_interest3] # create data series of one column
data3 = data3.resample("1MS").mean()
data3 = data3.truncate(before = first_date)

# %% Correlation Matrix

a1 = data.values[~np.isnan(data.values)]
a2 = data2.values[~np.isnan(data2.values)]
corr_1 = 1.0
corr_2 = 0
lag = 0
for shift in range(len(a1)-1):
    first_corr = pearsonr(a2[shift:], np.roll(a1,shift)[shift:])
    second_corr = pearsonr(np.roll(a2, shift)[shift:], a1[shift:])
    if first_corr[1] < second_corr[1]:
        corr_temp = first_corr
        j = 1
    else:
        corr_temp = second_corr
        j = 2
    if corr_temp[1] < corr_1:
        corr_1 = corr_temp[1]
        corr_2 = corr_temp[0]
        lag = shift
        first = j
    else:
        pass

print("The most significant correlation between", transect_of_interest, "and", 
      transect_of_interest2, "occurs at a lag of", lag, "month/s.", 
      "The correlation coefficient is:", np.round(corr_2,5), 
      "with a p-value of:", np.round(corr_1,5),
      "from shifting type", j)

# %% SET TRAIN AND TEST SERIES

train = data.truncate(after=event_date) # create train data for pre-event data
test = data.truncate(before=event_date) # create test data for post-event data


# %% LOWESS MOVING AVERAGE FIT

y3 = data3.values
y2 = data2.values
y = data.values
x = np.linspace(0, stop = len(y)-1, num = len(y))
nearest_points = 6

smoothed = sm.nonparametric.lowess(exog=x, endog=y, frac=nearest_points/len(y), missing = 'drop')
smoothed2 = sm.nonparametric.lowess(exog=x, endog=y2, frac=nearest_points/len(y), missing = 'drop')
smoothed3 = sm.nonparametric.lowess(exog=x, endog=y3, frac=nearest_points/len(y), missing = 'drop')

# Plot the fit line
fig, ax = pylab.subplots(figsize = (8, 5.3))

ax.scatter(x, y, color = 'blue', s = 10)
ax.scatter(x, y2, color = 'C1', s = 10)
ax.scatter(x, y3, color = 'C2', s = 10)
ax.plot(smoothed[:, 0], smoothed[:, 1], c="blue", linewidth = 2, label = 'In Lee of Uniwave200 WEC')
ax.plot(smoothed2[:, 0], smoothed2[:, 1], c="C1", linewidth = 2, label = 'Centre of Grassy Beach')
ax.plot(smoothed3[:, 0], smoothed3[:, 1], c="C2", linewidth = 2, label = 'Other End of Grassy Beach')
ax.axhline(y=0, color = 'k', linewidth = 1)
ax.axvline(x=51.5, ymin=0.1, ymax=0.9, color =  'grey', linestyle = '--')
# ax.text(38, -40, 'Uniwave200 Install Date', bbox = dict(facecolor = '1', edgecolor = 'orange', pad = 2.0))

ax.set_yticklabels([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40], fontsize = 14)
ax.set_xticks([3, 15, 27, 39, 51, 63])
ax.set_xticklabels(['2017', '2018', '2019', '2020', '2021', '2022'], fontsize = 14)
ax.set_xlabel('Year', fontsize = 14)
ax.set_ylabel('Shoreline Position Relative to Median (m)', fontsize = 14, wrap = True)
ax.legend(fontsize = 12)

pylab.autoscale(enable=True, axis="x", tight=True)

# plt.savefig(figname, dpi = DPI, bbox_inches = 'tight')
print("Change in shoreline since the WEC was installed is approximately {} metres".format(smoothed3[-1][1] - smoothed3[-14][1]))


# %% LOWESS MOVING AVERAGE FIT WITH 1-MONTH LAG TO OTHER END TRANSECT

y2 = np.roll(data2.values,1)[1:]
y = data.values[1:]
x = np.linspace(0, stop = len(y)-1, num = len(y))

smoothed = sm.nonparametric.lowess(exog=x, endog=y, frac=0.09, missing = 'drop')
smoothed2 = sm.nonparametric.lowess(exog=x, endog=y2, frac=0.09, missing = 'drop')

# Plot the fit line
fig, ax = pylab.subplots()

ax.scatter(x, y, color = 'blue', s = 10)
ax.scatter(x, y2, color = 'grey', s = 10)
ax.plot(smoothed[:, 0], smoothed[:, 1], c="blue", linewidth = 2)
ax.plot(smoothed2[:, 0], smoothed2[:, 1], c="grey", linewidth = 1.5)
ax.axvline(x=50, ymin=0.1, ymax=0.9, color =  'orange', linestyle = '--')
# ax.text(38, -40, 'Uniwave200 Install Date', bbox = dict(facecolor = '1', edgecolor = 'orange', pad = 2.0))

ax.set_xticks([2, 14, 26, 38, 50, 62])
ax.set_xticklabels(['2017', '2018', '2019', '2020', '2021', '2022'])
ax.set_xlabel('Year')
ax.set_ylabel('Shoreline Position (m)')

pylab.autoscale(enable=True, axis="x", tight=True)

plt.savefig(figname, dpi = DPI, bbox_inches = 'tight')
print("Change in shoreline since the WEC was installed is approximately {} metres".format(smoothed[-1][1] - smoothed[-14][1]))

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
plt.plot(test.index.values, test.values, 'o', markersize = 4, label = 'post-WEC; {}'.format(textstr), color = 'C2')

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

