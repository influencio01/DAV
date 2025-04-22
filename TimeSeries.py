import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv('weather_data.csv')  # Replace with your file name

# Convert 'Date/Time' to datetime format
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

# Set the date column as index
data.set_index('Date/Time', inplace=True)

# Plot original temperature data
data['Temp_C'].plot(title='Temperature Over Time', figsize=(10, 4))
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.show()

# Resample to daily or monthly average temperature
monthly_data = data['Temp_C'].resample('M').mean()

# Decompose the time series (Trend + Seasonality + Residual)
decomposition = seasonal_decompose(monthly_data, model='additive')
decomposition.plot()
plt.show()
