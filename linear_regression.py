import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_dataset.csv', index_col=0)

df = df.sample(frac=1).reset_index(drop=True)

x = df['YearsExperience']
y = df['Salary']

training_size = int(len(df) * 0.8)

X_train = x.iloc[:training_size]
X_test = x.iloc[training_size:]
y_train = y.iloc[:training_size]
y_test = y.iloc[training_size:]

print(len(X_train), len(X_test), len(y_train), len(y_test))

# Mean calculation of X_train and y_train
x_mean = X_train.mean()
y_mean = y_train.mean()

# Calculate m1
numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
denominator = np.sum((X_train - x_mean) ** 2)
m1 = numerator / denominator

# Calculate m0
m0 = y_mean - m1 * x_mean

print(f"Fomrmula: y = {m0:.2f} + {m1:.2f}*X")

y_pred = m0 + m1 * X_test

df_check = pd.DataFrame({'y test': y_test, 'y pred': y_pred})

print(df_check)

mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean((abs(y_test - y_pred)) ** 2)
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")

error_porcentaje = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Error in percentage: {error_porcentaje:.2f}%")

# Gráfico de dispersión regresión lineal
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='orange', label='Testing data')
plt.plot(X_test, y_pred, color='red', label='Linear regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.legend()
plt.show()