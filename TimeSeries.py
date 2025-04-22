import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv('/content/Weather Data.csv')

# Convert to datetime
data['Date/Time'] = pd.to_datetime(data['Date/Time'])
data.set_index('Date/Time', inplace=True)

# Plot original temperature
data['Temp_C'].plot(title='Temperature Over Time', figsize=(10, 4))
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.show()

# Resample to daily average
daily_data = data['Temp_C'].resample('D').mean()

# Drop NaN values (in case any day has missing data)
daily_data = daily_data.dropna()

# Decompose with daily period (approx. monthly seasonality)
decomposition = seasonal_decompose(daily_data, model='additive', period=30)
decomposition.plot()
plt.show()
