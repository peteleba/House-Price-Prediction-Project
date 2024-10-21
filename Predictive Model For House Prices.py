import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('house_prices.csv')

# Step 2: Handle missing values (if any)
df['square_footage'].fillna(df['square_footage'].mean(), inplace=True)
df['bedrooms'].fillna(df['bedrooms'].mean(), inplace=True)

# Step 3: Convert categorical 'location' column into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Step 4: Define the features (X) and target (y)
X = df.drop('price', axis=1)  # Features (everything except price)
y = df['price']  # Target (price)

# Step 5: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 7: Make predictions using the test set
y_pred_lr = lr.predict(X_test)

# Step 8: Evaluate the model's performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Mean Squared Error (Linear Regression): {mse_lr}')

# Step 9: Optional - Visualize the predictions vs actual prices
plt.scatter(y_test, y_pred_lr)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Linear Regression)')
plt.show()
