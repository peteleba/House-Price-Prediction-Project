# House-Price-Prediction-Project
This project focuses on building a machine learning model to predict house prices using various features such as square footage, number of bedrooms, and location. It demonstrates the use of Linear Regression, data preprocessing, and evaluation using Python, Pandas, and Scikit-learn.

# House Price Prediction using Regression Models

## Objective
Build a machine learning model to predict house prices based on features such as square footage, number of bedrooms, and location.

## Project Overview
This project utilizes data preprocessing techniques and linear regression modeling to predict house prices. We handle categorical variables (location) through one-hot encoding and train the model using Scikit-learn's linear regression algorithm. Our goal is to achieve reasonable predictions and assess the model’s performance using Mean Squared Error (MSE).

## Skills and Tools
- **Skills**: Data cleaning, feature engineering, regression modeling
- **Tools**: Python, Pandas, Scikit-learn, Matplotlib

## Dataset
The dataset contains 10 features:
- square_footage
- num_bedrooms
- num_bathrooms
- location_score
- year_built
- garage_size
- num_floors
- distance_city_center
- has_pool
- crime_rate

The target variable is `house_price`.

## Steps Involved
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and splitting data.
2. **Modeling**: Applying linear regression to build a predictive model.
3. **Evaluation**: Using Mean Squared Error (MSE) to measure model performance.
4. **Visualization**: Plotting actual vs predicted prices.

## Results
- **MSE**: Mean Squared Error: 560231.7362342561 A scatter plot comparing actual and predicted prices showed a reasonable correlation.
- The model performed with reasonable accuracy, and future improvements could be made by exploring more complex algorithms.

## How to Run
1. Clone the repository.
2. Install the necessary libraries:
```bash
pip install -r requirements.txt
