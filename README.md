# Air_Quality_Index
A project for predicting air quality index over time
## Predicting AQI:

One of the most reliable ways to quantify air pollution is by calculating the **Air Quality Index (AQI)**

### **Data Set Description**

AQI dataset from UCI Machine Learning repository

It is multivariable time series data.The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level, within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2)  and were provided by a co-located reference certified analyzer. Missing values are tagged with -200 value.

It has 15 attributes:
<img width="908" height="664" alt="Screenshot 2025-08-08 at 9 48 57 PM" src="https://github.com/user-attachments/assets/e456578c-c8b0-41ee-b8e4-2d2502ae8021" />
<img width="875" height="264" alt="Screenshot 2025-08-08 at 9 47 40 PM" src="https://github.com/user-attachments/assets/f69f205d-0c55-4b9a-ac2f-6455870fb6b1" />





 Libraries used

Source code:

```python
import pandas as pd
import numpy as np
```

| **Library** | **Role in AQI Prediction Pipeline** |
| --- | --- |
| pandas | Load, clean, and manipulate tabular data, structuring data |
| numpy | Perform numerical computations |

 Loading Dataset

we have used google collab for building this project 

```python
# importing the data from csv file into pandas dataframe
air_quality_df = pd.read_csv('/AirQualityUCI.csv') 
#checking whether data is loaded or not 
air_quality_df.head()
```

 Preprocessing Data

This includes structuring of data, removing empty rows and columns 

In our .csv file variables are separated by ‘;’ and few instances have ‘,’ in place of decimal we have fixed that.

Additionally last two columns of dataset had ‘NaN’ value which are unnecessary we removed them.

Our dataset has only entries till index 9356(9357 value containing rows) rest of rows are null.

Another task is filling null values between the dataset, it was mentioned in the UCL ML documentation that null values in dataframe are tagged with ‘-200’. So we cannot directly feed dataframe containing ‘-200’ to ML model as our result will be affected alot, so we fill in the instances where ‘-200’ occurs.

There are various methods such as median value , mean , interpolating value filling, here we have simply chosen mean value substitution. But before calculating mean of each attributes we also need to replace ‘-200’ with NaN. Now we replace NaN with each attributes respective means. 

Here the task of data processing has been completed.

### 4. Forecasting v/s Regression

Now many operations such as forecasting or regression can be performed on processed.

|  | **Forecasting** | **Regression** |
| --- | --- | --- |
| **Objective** | Predict **future values** (usually over time) | Model relationship between **dependent and independent variables** |
| **Focus** | **Time-based prediction** (e.g. next day, next month) | **Relationship modeling** (e.g. how X affects Y) |
| **Input Data** | Time series (ordered over time) | Cross-sectional or time series (not necessarily ordered) |
| **Output** | Future value(s) of a variable | Estimated value of dependent variable |
| **Time Dependency** | Strongly time-dependent | Not necessarily time-dependent |
| **Typical Methods** | ARIMA, Prophet, LSTM, Exponential Smoothing | Linear regression, polynomial regression, logistic regression |
| **Example Use Case** | Forecast next month’s sales | Determine how advertising affects sales |
| **Trend/Cycle Analysis** | Often includes trend, seasonality, cycles | Usually ignores time trends unless explicitly modeled |

We used forecasting in this project and used **FB prophet Model**

## FB PROPHET MODEL

The input to Prophet is always a dataframe with two columns: `ds` and `y`. The `ds` (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The `y` column must be numeric, and represents the measurement we wish to forecast.

Example:

|  | **ds** | **y** |
| --- | --- | --- |
| **0** | 2007-12-10 | 9.590761 |
| **1** | 2007-12-11 | 8.519590 |
| **2** | 2007-12-12 | 8.183677 |
| **3** | 2007-12-13 | 8.072467 |
| **4** | 2007-12-14 | 7.893572 |

<img width="817" height="132" alt="Screenshot 2025-08-08 at 9 50 37 PM" src="https://github.com/user-attachments/assets/bb5e3522-4644-48dd-ac5f-751794104190" />
<img width="829" height="127" alt="Screenshot 2025-08-08 at 9 51 22 PM" src="https://github.com/user-attachments/assets/ffe5e14c-32c1-4ea9-9a64-ba45e5126eb2" />



We fit the model by instantiating a new `Prophet` object. Any settings to the forecasting procedure are passed into the constructor. Then you call its `fit` method and pass in the historical dataframe
<img width="828" height="111" alt="Screenshot 2025-08-08 at 9 52 44 PM" src="https://github.com/user-attachments/assets/f11f3d54-09fb-4c70-aaa6-1d2c35628349" />

Predictions are then made on a dataframe with a column `ds` containing the dates for which a prediction is to be made. You can get a suitable dataframe that extends into the future a specified number of days using the helper method `Prophet.make_future_dataframe`. By default it will also include the dates from the history, so we will see the model fit as well.
<img width="820" height="114" alt="Screenshot 2025-08-08 at 9 53 16 PM" src="https://github.com/user-attachments/assets/d0eadf43-0120-43a7-a575-fccd4765f632" />

|  | **ds** |
| --- | --- |
| **3265** | 2017-01-15 |
| **3266** | 2017-01-16 |
| **3267** | 2017-01-17 |
| **3268** | 2017-01-18 |
| **3269** | 2017-01-19 |

The `predict` method will assign each row in `future` a predicted value which it names `yhat`. If you pass in historical dates, it will provide an in-sample fit. The `forecast` object here is a new dataframe that includes a column `yhat` with the forecast, as well as columns for components and uncertainty intervals
****
<img width="819" height="115" alt="Screenshot 2025-08-08 at 9 54 19 PM" src="https://github.com/user-attachments/assets/73f3aee3-b0c6-499d-9373-0906dabf05cb" />


|  | **ds** | **yhat** | **yhat_lower** | **yhat_upper** |
| --- | --- | --- | --- | --- |
| **3265** | 2017-01-15 | 8.212625 | 7.456310 | 8.959726 |
| **3266** | 2017-01-16 | 8.537635 | 7.842986 | 9.290934 |
| **3267** | 2017-01-17 | 8.325071 | 7.600879 | 9.072006 |
| **3268** | 2017-01-18 | 8.157723 | 7.512052 | 8.924022 |
| **3269** | 2017-01-19 | 8.169677 | 7.412473 | 8.946977 |

You can plot the forecast by calling the `Prophet.plot` method and passing in your forecast dataframe.
<img width="811" height="103" alt="Screenshot 2025-08-08 at 9 54 52 PM" src="https://github.com/user-attachments/assets/41a3eecc-62ae-4d27-93b0-625660b396d4" />


If you want to see the forecast components, you can use the `Prophet.plot_components` method. By default you’ll see the trend, yearly seasonality, and weekly seasonality of the time series. If you include holidays, you’ll see those here, too.

<img width="815" height="86" alt="Screenshot 2025-08-08 at 9 55 37 PM" src="https://github.com/user-attachments/assets/2634e65b-76fa-4b88-90b1-8f30ddbf3889" />
