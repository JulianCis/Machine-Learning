# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset
df = pd.read_csv('Salary_dataset.csv', index_col=0)

# Shuffling the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Separate X and y
x = df['YearsExperience']
y = df['Salary']

# Set the training size to 80% and test size to 20$
training_size = int(len(df) * 0.8)

# Dividing the dataset into training and testing
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

print(f"Formula: y = {m0:.2f} + {m1:.2f}*X")

# Making the predictions of the testing dataset
y_pred = m0 + m1 * X_test

# Dataframe to compare de testing and the predicted data
df_check = pd.DataFrame({'y test': y_test, 'y pred': y_pred})

print(df_check)

# Calculate the mean squared error and the mean abs error
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean((abs(y_test - y_pred)) ** 2)
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")

# Calculate the error percentage in scale of 1-100
error_porcentaje = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Error in percentage: {error_porcentaje:.2f}%")

# Linear regression graphic
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='orange', label='Testing data')
plt.plot(X_test, y_pred, color='red', label='Linear regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.legend()
plt.show()
