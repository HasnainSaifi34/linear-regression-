import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the data into a DataFrame
df = pd.read_csv(r"D:\USER DATA C DRIVE\Documents\CSV\Housing.csv")

# Extracting features (X) and target variable (y)
X = df[['area']]
y = df[['price']]

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
new_area = [[3600]]
predicted_price = model.predict(new_area)
print(f"predicted price from the model {predicted_price}")
# Plot the data points
plt.scatter(X, y, label='Actual Prices', color='blue')

# Plot the regression line
plt.plot(X, predictions, label='Linear Regression', color='red')

# Add labels and title
plt.xlabel('Area (sq. ft)')
plt.ylabel('Price')
plt.title('Home Price Prediction')
# y= mx +b
manualPrediction = model.coef_*3600+model.intercept_

print(f"predicted value manually {manualPrediction}");
# Display the legend
plt.legend()
futurePredictionDataFrame = pd.read_excel(r"D:\USER DATA C DRIVE\Documents\CSV\home price.xlsx");
# Show the plot
futurePrediction=model.predict(futurePredictionDataFrame)
futurePredictionDataFrame['prices']=futurePrediction
#EXPORTING the predictions to an excel file
futurePredictionDataFrame.to_excel("home price.xlsx",index=False)
print(futurePredictionDataFrame)
plt.show()
