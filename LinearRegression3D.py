import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == '__main__':
    df = pandas.read_csv('data/Salary_Data.csv')

    x_experience = df["YearsExperience"].to_numpy()
    x_age = df["Age"].to_numpy()
    y = df["Salary"].to_numpy()
    x = np.column_stack((x_experience, x_age))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_age,x_experience,y)
    ax.set_xlabel('x_age')
    ax.set_ylabel('x_experience')
    ax.set_zlabel('salary')


    model = LinearRegression()
    model.fit(x, y)

    w1,w2 = model.coef_
    w0 = model.intercept_

    print(w0,w2)
    predicted_y = model.predict(x)
    print(predicted_y)
    mse = np.mean((predicted_y - y) ** 2)
    print(f"Mean Squared Error: {mse}")
    # plt.scatter(x, y)
    # plt.xlabel('Years')
    # plt.ylabel('Salary')
    plt.plot(x_age,x_experience,predicted_y,color='red')
    plt.show()