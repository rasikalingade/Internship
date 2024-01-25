# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (replace this with your actual dataset)
data = {'Advertising Spend': [200, 400, 600, 800, 1000],
        'Sales': [1200, 1600, 2100, 2500, 3100]}
df = pd.DataFrame(data)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(df[['Advertising Spend']], df['Sales'])

# Function to predict sales with adjustable advertising spend
def predict_sales(ad_spend_change):
    new_ad_spend = df['Advertising Spend'].values.reshape(-1, 1) + ad_spend_change
    predicted_sales = model.predict(new_ad_spend)
    return predicted_sales

# Predict sales with increased advertising spend by 2
predicted_sales_increase = predict_sales(100)

# Predict sales with decreased advertising spend by 2
predicted_sales_decrease = predict_sales(-200)

# Plotting the original and predicted sales
plt.scatter(df['Advertising Spend'], df['Sales'], color='black', label='Original Data')
plt.plot(df['Advertising Spend'], model.predict(df[['Advertising Spend']]), color='blue', linewidth=3, label='Regression Line')

# Plotting the predicted sales with increased advertising spend by 2
plt.scatter(df['Advertising Spend'] + 2, predicted_sales_increase, color='red', label='Increase by 100')

# Plotting the predicted sales with decreased advertising spend by 2
plt.scatter(df['Advertising Spend'] - 2, predicted_sales_decrease, color='green', label='Decrease by 200')

plt.title('Sales Prediction with Advertising Spend Adjustment')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.legend()
plt.show()