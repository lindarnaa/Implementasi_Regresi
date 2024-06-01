import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Path to the CSV file
file_path = r'C:\xampp\htdocs\metnum-linda ratna/Student_Performance.csv'

# Read the CSV file with delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

# Check the columns in the dataframe to ensure 'NL' and 'NT' columns are present
print("Columns in data:", data.columns)

# Assuming columns in CSV are 'NL' for Number of Exercises and 'NT' for Test Scores
NL = data['NL']
NT = data['NT']

# Linear Regression Model
X = NL.values.reshape(-1, 1)
y = NT.values
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
rms_error_linear = np.sqrt(np.mean((y - y_pred_linear) ** 2))
print("RMS Error (Linear Regression):", rms_error_linear)

# Simple Power Model (Transform to log-log space)
# Avoid log(0) by replacing 0 with a small value (e.g., 1e-10)
NL_non_zero = NL.replace(0, 1e-10)
log_X = np.log(NL_non_zero.values.reshape(-1, 1))
log_y = np.log(y)
power_model = LinearRegression()
power_model.fit(log_X, log_y)
log_y_pred_power = power_model.predict(log_X)
y_pred_power = np.exp(log_y_pred_power)
rms_error_power = np.sqrt(np.mean((y - y_pred_power) ** 2))
print("RMS Error (Power Model):", rms_error_power)

# Plotting the graph
plt.scatter(NL, NT, label='Original Data')
plt.plot(NL, y_pred_linear, color='red', label='Linear Regression')
plt.plot(NL, y_pred_power, color='green', label='Power Model')
plt.xlabel('Number of Exercises (NL)')
plt.ylabel('Test Scores (NT)')
plt.legend()
plt.title('Relationship between Number of Exercises and Test Scores')
plt.show()