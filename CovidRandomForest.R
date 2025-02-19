#####################################################################################################################################################################################
# TIME SERIES ANALYSIS USING RANDOM FOREST REGRESSION
# COVID 19 Data
# Adapted from https://petolau.github.io/Regression-trees-for-forecasting-time-series-in-R/
#####################################################################################################################################################################################

### Import Libraries
library(readr)
library(tidyverse)
library(tsibble)
library(randomForest)
library(rpart)
library(forecast)
library(feather)
library(rpart.plot)
library(ggforce)
library(lubridate)

### Import Data
data_pre = read.csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

### Select Columns
data_pre <- select(data_pre, c("new_cases", "date", "location", "continent"))

#### Convert Character Columns to Factors
data_pre <- as.data.frame(unclass(data_pre),
                          stringsAsFactors = TRUE)

#### Remove NA and extraneous data rows
data_pre <- data_pre[!(is.na(data_pre$continent) | data_pre$continent == ""), ]

### Separate By Country
data_pre_can <- subset(data_pre, data_pre['location']  == 'Canada')
### Remove Country and Location columns
data_pre_can <- subset(data_pre_can, select = -c(location, continent))

### Convert Date to Datetime type
data_pre_can$date <- as.Date(data_pre_can$date)
### Create Weekday variable week_num
data_pre_can$week_num = as.numeric(format(data_pre_can$date, format = "%u"))

## Explore the Data
# New Cases
newcases_pl <- ggplot(data_pre_can, aes(x=date, y=new_cases)) +
  geom_line() + scale_x_date(date_breaks = "3 months", date_labels = "%b") +
  xlab("Months") + ylab("New Cases") + ggtitle("New Covid Cases in Canada")+geom_area(fill="lightblue", color="black") + 
  geom_smooth(method = lm, col = "red", se = FALSE)
newcases_pl

#####################################################################################################################################################################################
# PREPROCESSING
#####################################################################################################################################################################################

### Set the time frame
## The time frame will be the past 36 weeks
data_timeframed <- data_pre_can %>%
  filter(data_pre_can$date > data_pre_can[length(data_pre_can$date), 'date'] %m-% days(246), # Exclusive, i.e. 245 days
         data_pre_can$date < data_pre_can[length(data_pre_can$date), 'date'])

### Create global variables
## Parameters used throughout analysis

## Set the period for analysis based on the units of measurement in the raw data
# Contain the unique dates from the raw dataset
n_date <- unique(data_timeframed[ , 'date'])
# Contain the period length as 1 day (48 sequential observations)
day_period <- 1 # # The data is daily, i.e. each observation represents measurements across a one day interval in time

## Define Seasonality Periods
# Weekly
week_period <- day_period * 7 # Period 1 is 24 hours *  7 is 1 week
# Two Day
twoday_period <- day_period * 2 # Period 2 is every other day, i.e. day_period * 2

## Note: these periods are chosen because they measure frequencies that are asynchronous up to 14 periods (the prediction window)

## Note: Rob Hyndman recommends NOT modelling monthly seasonality patterns, given their dynamic nature as a window of time

# Contain a theme for consistent and clear plotting
theme_ts <- theme(panel.border = element_rect(fill = NA, 
                                              colour = "grey10"),
                  panel.background = element_blank(),
                  panel.grid.minor = element_line(colour = "grey85"),
                  panel.grid.major = element_line(colour = "grey85"),
                  panel.grid.major.x = element_line(colour = "grey85"),
                  axis.text = element_text(size = 13, face = "bold"),
                  axis.title = element_text(size = 15, face = "bold"),
                  plot.title = element_text(size = 16, face = "bold"),
                  strip.text = element_text(size = 16, face = "bold"),
                  strip.background = element_rect(colour = "black"),
                  legend.text = element_text(size = 15),
                  legend.title = element_text(size = 16, face = "bold"),
                  legend.background = element_rect(fill = "white"),
                  legend.key = element_rect(fill = "white"))

### Segment data into testing and training data sets
## To train the machine learning algorithm and test predictions for accuracy

# Using the 'date' column of the original data as a reference,
# Contain the dates from 36 weeks ago to 2 weeks ago in the training set (34 weeks)
data_train <- data_timeframed[data_timeframed[ , 'date'] %in% n_date[1:(length(n_date) - 14)], ]
# Contain the dates from 2 weeks ago to present in the test test (2 weeks)
data_test <- data_timeframed[data_timeframed[ , 'date'] %in% n_date[(length(n_date) - 13):length(n_date)] , ]

### Create Time Series Object from Training Data Set
## Impart weekly seasonality to Time Series object (frequency = daily)
data_ts <- ts(data_train$new_cases, frequency = week_period)

### Decompose Time Series into seasonality and trend component waveforms
## Using Loess
# Return 'time.series': a multiple time series with columns seasonal, trend and remainder.

decomp_ts <- stl(data_ts, s.window = week_period, robust = TRUE)$time.series

### Create a dataframe of the values from the original and decomposed time series,indexed by date and type (a factor of 4 levels)
decomp_stl <- data.frame(NewCases = c(data_train$new_cases, as.numeric(decomp_ts)), # 'Load' is value from decomposed time series
                         Date = rep(x = data_train[, 'date'], times = ncol(decomp_ts)+1), # Repeat the 'date_time' values from the training data set by the number of columns in the decomposed time series (3) + 1 (= 4)
                         Type = factor(rep(c("original data", colnames(decomp_ts)), # As a factor, create 'original data' and the column names for each of the number of rows in the decomposed series
                                           each = nrow(decomp_ts)),
                                       levels = c("original data", colnames(decomp_ts)))) # The levels of the factor will be 'original data', seasonal', 'remainder', and 'trend' (4 levels)

### Plot the dataframe of decomposed timeseries data
ggplot(decomp_stl, aes(x = Date, y = NewCases)) +
  geom_line() + 
  facet_grid(Type ~ ., scales = "free_y", switch = "y") + # Facet by type (4 lvl factor), the scales vary across columns and the right hand labels set to the left (switch = y)
  labs(x = "Date", y = NULL,
       title = "Time Series Decomposition by STL") +
  theme_ts

#####################################################################################################################################################################################
# FEATURE ENGINEERING
#####################################################################################################################################################################################

### Seasonality
## Since this is a Non-Linear Regression analysis, it's appropriate to use a fourier approach to deconstructing seasonal patterns
## With Linear Regression, it would be appropriate to create seasonal dummy variables
# Regression trees cannot accommodate trends because they use the training set is used to infer logical rules by which to predict future values.

# The lag of time series can be preprocessed by removing noise or trend for example by STL decomposition method to ensure stability.

# Given the sampling rate of the raw date, we can a priori estimate 2 seasonal fourier terms: daily and weekly

# Contain the seasonal data from the raw data in a multi-seasonal timeseries object
data_multits <- msts(data_train$new_cases, seasonal.periods = week_period)

K <- 2

data_seasonfourier <- fourier(data_multits, 
                              K = K)

#####################################################################################################################################################################################
# ARIMA MODEL
#####################################################################################################################################################################################

data_trend <- ts(decomp_ts[ , 'trend'])

trend_arimafit <- auto.arima(data_trend)

mean_arimafore <- forecast(trend_arimafit, week_period * 2)$mean

arima_data <- data.frame(NewCases = c(decomp_ts[, 'trend'], mean_arimafore),
                         Date = c(data_train$date, data_test$date),
                         Type = c(rep("Real", nrow(data_train)), rep("Forecast", nrow(data_test))))

ggplot(arima_data, aes(Date, NewCases, color = Type)) +
  geom_line(size = 1) +
  geom_point() +
  labs(title = paste(trend_arimafit)) +
  theme_ts

#####################################################################################################################################################################################
# REGRESSION TREE MODELS: RPART AND CTREE
#####################################################################################################################################################################################

### RPART is CART (Classification and Regression Trees)
## The model is created by recursive binary partitioning of the sample space via regression

### Create Global Variables

N <- nrow(data_train) # Number of observations in the training set

lag_window <- N - 1 * week_period # number of obvs / period = days in train set minus lag (1 day)
### In the case of daily data, the window would be the number of observations - 1

# Contain the seasonal and error data, separate from the trend
# i.e. de-trend
load_detrend <- rowSums(decomp_ts[, c('seasonal', 'remainder')])

# Contain the seasonal part of the time series data to be used as a lag feature
lag_seasonal <- decomp_ts[1:(lag_window), 1]

# Contain a function to calculate MAPE (Mean Absolute Percentage Error) for model evaluation
# mape (real, pred)
mape <- function(real, pred){
  return(100 * mean(abs((real - pred)/real)))
}

### Format the Data for Random Forest Regression
## The time series data as a series of vectors (dataframe) cannot be interpreted by ML algorithms, e.g. Random Forest

ts_matrix_train <- data.frame(NewCases = tail(load_detrend, lag_window), # NOTE: the detrend data is used
                              data_seasonfourier[(week_period+1):N, ], # Incorporates fourier estimates from msts
                              Lag = lag_seasonal)

### Growing Regression Trees
## The algorithm will use the lag of the timeseries as a predictor of the time series itself
## Construct an 'if-then' model of the data
## Iterative process: grow a tree, inspect the results, tune the parameters, repeat ...

# Contain the first regression tree
tree1 <- rpart(NewCases ~ ., data = ts_matrix_train)

## Results
# Variable Importance Factor
tree1$variable.importance
# Number of splits
paste("Number of splits: ", tree1$cptable[dim(tree1$cptable)[1], "nsplit"])

rpart.plot(tree1, digits = 2, 
           box.palette = viridis::viridis(10, option = "D", begin = 0.85, end = 0), 
           shadow.col = "grey65", col = "grey99")
## Plotting the tree conveys clearly what the algorithm is doing

## Evaluate the model
## Model vs Actual Data

# Contain Data to Plot
tree1_performance <- data.frame(NewCases = c(ts_matrix_train$NewCases, # Actual training data
                                         predict(tree1)),      # Predicted values
                                Time = rep(1:length(ts_matrix_train$NewCases), 2),
                                Type = rep(c("Real", "RPART"), each = length(ts_matrix_train$NewCases)))

# Plot the Model Estimates vs Actual Data
ggplot(tree1_performance, aes(Time, NewCases, color = Type)) +
  geom_line(size = 0.8, alpha = 0.75) +
  labs(y = "Detrended load", title = "Fitted values from RPART tree") +
  theme_ts

# Mean Absolute Percentage Error
# error of fitted values against real values
mape(ts_matrix_train$NewCases, predict(tree1))

### Preliminary Results: Tree 1
## The model is doing what it should
## The  fit is not great

### Parameter Tuning
## Adjust the Complexity Parameter
## Change the minimum and maximum level of splits

tree2 <- rpart(NewCases ~ ., data = ts_matrix_train,
               control = rpart.control(
                 minsplit = 2,
                 maxdepth = 30,
                 cp = 0.000001
               ))

## Results
# Variable Importance Factor
tree1$variable.importance
# Number of splits
paste("Number of splits: ", tree2$cptable[dim(tree2$cptable)[1], "nsplit"])
# Plot the tree
plot(tree2, compress = TRUE)

## Evaluate the model against the Testing Data

tree2_performance <- data.frame(NewCases = c(ts_matrix_train$NewCases,
                                         predict(tree2)),
                                Time = rep(1:length(ts_matrix_train$NewCases), 2),
                                Type = rep(c("Real", "CART"), each = length(ts_matrix_train$NewCases)))

# Plot the Model Estimates vs Actual Data
ggplot(tree2_performance, aes(Time, NewCases, color = Type)) +
  geom_line(size = 0.8, alpha = 0.75) +
  geom_point() +
  labs(y = "Detrended load", title = "Fitted values from CART") +
  theme_ts

# MAPE
mape(ts_matrix_train$NewCases, predict(tree2))

### Preliminary Results: Tree 2
## The model performs better with added complexity
## The model may be over fitted to the data

### Forecast using the CART Model

## Create Testing Data Matrix
lag_test <- decomp_ts[(lag_window+1):N, 'seasonal']
 
fourier_test <- fourier(data_multits, K = K, h = week_period * 2)

ts_matrix_test <- data.frame(fourier_test,
                             Lag = lag_test)

cart_fcst <- predict(tree2, ts_matrix_test) + mean_arimafore

data_cartfcst <- data.frame(NewCases = c(data_train$new_cases, data_test$new_cases, cart_fcst),
                            Date = c(data_train$date, rep(data_test$date, 2)),
                            Type = c(rep("Train data", nrow(data_train)),
                                     rep("Test data", nrow(data_test)),
                                     rep("Forecast", nrow(data_test))))

ggplot(data_cartfcst, aes(Date, NewCases, color = Type)) +
  geom_line(size = 0.8, alpha = 0.75) +
  facet_zoom(x = Date %in% data_test$date, zoom.size = 1.2) +
  labs(title = "Forecast from CART") +
  theme_ts
