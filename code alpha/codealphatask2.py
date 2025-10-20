import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data_url = "https://www.statlearning.com/s/Advertising.csv"
try:
    df = pd.read_csv(data_url, index_col=0) 
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check your internet connection or the data URL.")
    exit()

print(" First 5 rows of Advertising data")
print(df.head())
print("\n")
features = ['TV', 'Radio', 'Newspaper'] 
target = 'Sales'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(" Model Evaluation")
print(f"R-squared (R2): {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print("\n")

coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(" Impact of Advertising on Sales (Model Coefficients)")
print(coefficients)

print("\nInterpretation")
print("An R-squared of (e.g.) 0.897 means the model explains ~89.7% of the variance in sales.")
print("The coefficients show how much 'Sales' (in thousands of units) are expected to increase for each $1,000 spent on an advertising platform, holding other platforms constant.")