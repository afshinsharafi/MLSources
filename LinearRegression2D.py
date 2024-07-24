import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == '__main__':
    data = pandas.read_csv('data/Salary_Data.csv')
    print(data)
    data.describe()

    x_experience = data.iloc[:,0:1].values
    y = data.iloc[:,1].values
    plt.scatter(x_experience, y)
    plt.xlabel('x_experience')
    plt.ylabel('salary')
    plt.title('Test set prediction Salary')

    model = LinearRegression()
    model.fit(x_experience, y)

    w1 = model.coef_
    w0 = model.intercept_

    print(w0,w1)
    predicted_y = model.predict(x_experience)
    print(predicted_y)
    mse = np.mean((predicted_y - y) ** 2)
    print(f"Mean Squared Error: {mse}")
    plt.plot(x_experience,predicted_y,color='red')
    plt.show()