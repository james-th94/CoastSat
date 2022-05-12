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
# import datetime as dt
# import seaborn as sns
import statsmodels.api as sm
# from statsmodels.tsa.seasonal import STL, seasonal_decompose

# %% USER INPUTS
working_dir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/CoastSat/data/WSE_2015-06-23_2022-03-16"
data_file = "transect_time_series_tide_corrected.csv"
index_column = 0
transect_of_interest = "Transect 66"
event_date = '1/10/2021'

# Plotting
figname = 'Shoreline_position_scatter.png'
figtitle = 'Shoreline Position Time Series in the lee of the Uniwave200 WEC'
figyaxis = 'Shoreline Position (m)'
figxaxis = 'Date (year)'

# %% CHANGE DIRECTORY
os.chdir(working_dir)

# %% LOAD DATA AND PRE-PROCESSING

df = pd.read_csv(data_file, index_col=index_column) # read csv data into dateframe, df, with index set to dates
df.index = pd.to_datetime(df.index) # convert dates to datetime objects in the dataframe
data = df[transect_of_interest] # create data series of one column
data = data.resample("5D").mean()

train = data.truncate(after=event_date) # create train data for pre-event data
test = data.truncate(before=event_date) # create test data for post-event data

# %% LINEAR REGRESSION (STATS MODELS - Train)

y = train.values
x = np.linspace(0, stop = len(y) * 5, num = len(y))
x = sm.add_constant(x)

model_train = sm.OLS(y, x, missing = 'drop').fit()
predictions_train = model_train.predict(x)
results_summary_train = model_train.summary()

conf_int = model_train.conf_int(alpha=0.05)
bottom_train = np.multiply(x[:,1],conf_int[1][0]) + conf_int[0][0]
top_train = np.multiply(x[:,1],conf_int[1][1]) + conf_int[0][1]

intercept_train = round(model_train.params[0], 4)
coef_train = round(model_train.params[1], 4)
textstr_train = 'y = {}x + {}'.format(coef_train, intercept_train)

# %% LINEAR REGRESSION (STATS MODELS - Test)

y = test.values
x = np.linspace(0, stop = len(y) * 5, num = len(y))
x = sm.add_constant(x)

model = sm.OLS(y, x, missing = 'drop').fit()
predictions = model.predict(x)
results_summary = model.summary()
print(results_summary)

conf_int = model.conf_int(alpha=0.05)
bottom = np.multiply(x[:,1],conf_int[1][0]) + conf_int[0][0]
top = np.multiply(x[:,1],conf_int[1][1]) + conf_int[0][1]

intercept = round(model.params[0], 4)
coef = round(model.params[1], 4)

textstr = 'y = {}x + {}'.format(coef, intercept)

# %% PLOTTING DATA


# Plot data and regression formula
plt.plot(train.index.values, train.values, 'o', markersize = 4, label = 'pre-WEC', color = 'C0')
plt.plot([], [], ' ', label=textstr_train)

plt.plot(test.index.values, test.values, 'o', markersize = 4, label = 'post-WEC', color = 'C2')
plt.plot([], [], ' ', label=textstr)

plt.plot(test.index.values, predictions, color = 'C2')
plt.plot(test.index.values, top, color = 'C2')
plt.plot(test.index.values, bottom, color = 'C2')
plt.fill_between(test.index.values, bottom, top, alpha=0.3, color='C2')

#plt.text(np.min(test.index.values), np.max(top), textstr, fontsize=10, verticalalignment='top', color = 'C1')

plt.plot(train.index.values, predictions_train, color = 'C0')
plt.plot(train.index.values, top_train, color = 'C0')
plt.plot(train.index.values, bottom_train, color = 'C0')
plt.fill_between(train.index.values, bottom_train, top_train, alpha=0.3, color='C0')

#plt.text(np.min(train.index.values), np.max(top_train) + 10, textstr_train, fontsize=10, verticalalignment='top', color = 'C0')

# Change plot preferences
plt.grid(linestyle = '--')
plt.legend()
plt.tight_layout()

# Plot titles
plt.title(figtitle)
plt.ylabel(figyaxis)
plt.xlabel(figxaxis)
# Show plot
plt.show()
#plt.savefig(figname, dpi = 600, bbox_inches = 'tight')

# %% LINEAR REGRESSION - SEABORN

values = test.values
sns.regplot(x = np.linspace(0, stop = len(values)*5, num = len(values)), y = values, order=1)
plt.xlabel('Days since WEC Install')
plt.ylabel(figyaxis)
myTitle = 'Shoreline Position since the Installation of the Uniwave200 WEC'
plt.title(figtitle, loc = 'center')
#plt.savefig('seaborn_reg.png', dpi = 450, bbox_inches = 'tight')



# %% PLOTTING
textstr = 'y = {}x + {}'.format(coef, intercept)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.plot(predictions)
plt.plot(top, color = 'b')
plt.plot(bottom, color = 'b')
plt.fill_between(range(len(y)), bottom, top, alpha=0.3, color="b")
plt.plot(y, 'o', markersize = 4)
# plot parameters
plt.text(0, np.max(top), textstr, fontsize=10, verticalalignment='top')
plt.xticks(range(0,len(y)+1, 10), range(0, (len(y) + 1 )*5, 50))
plt.legend()
plt.title(figtitle)
plt.grid(linestyle = '--')
plt.ylabel(figyaxis)
plt.xlabel('Days since WEC install')
plt.show()

# %% OLD
def lowess_with_confidence_bounds(
    x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top
# %% OLD

eval_x = x
smoothed, bottom, top = lowess_with_confidence_bounds(
    x, y, eval_x, lowess_kw={"frac": 0.2}
)
# %% OLD
fig, ax = pylab.subplots()
ax.scatter(x, y)
ax.plot(eval_x, smoothed, c="k")
ax.fill_between(eval_x, bottom, top, alpha=0.5, color="b")
pylab.autoscale(enable=True, axis="x", tight=True)



# %% OLD
smoothed = sm.nonparametric.lowess(exog=x, endog=y, frac=0.15, missing = 'drop')

# Plot the fit line
fig, ax = pylab.subplots()

ax.scatter(x, y)
ax.plot(smoothed[:, 0], smoothed[:, 1], c="k")
pylab.autoscale(enable=True, axis="x", tight=True)


# %% OLD

series = pd.Series(y, index = pd.date_range('2015-12-28', periods = len(y), freq = "5D"), 
                   name = 'shoreline_pos')
series.describe()
# %% OLD

decompose_result_mult = seasonal_decompose(series, model="additive")

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot();

# %% OLD

stl = STL(series, seasonal=37)
res = stl.fit()
fig = res.plot()