# sml


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("Housing.csv")

# Predictor and target
X = df['SqFt_Living']
y = df['Price']

# Create spline term (knot at 2000)
df['SqFt_above2000'] = np.where(df['SqFt_Living'] > 2000,
                                df['SqFt_Living'] - 2000, 0)

# Add constant
X_spline = sm.add_constant(df[['SqFt_Living', 'SqFt_above2000']])

# Fit model
model = sm.OLS(y, X_spline).fit()
print(model.summary())

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['SqFt_Living'], y, color='lightgray', label='Data')

plt.plot(df['SqFt_Living'],
         model.predict(X_spline),
         color='black',
         linewidth=2,
         label='Spline Fit')

plt.axvline(2000, color='red', linestyle='--', label='Knot at 2000 sqft')
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Spline Regression")
plt.legend()
plt.show()
```
