import matplotlib.pyplot as plt

X = [[0], [1], [2], [3], [1]]
y = [0, 0, 1, 1, 1]
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=4)
neigh.fit(X, y)

test_x = [[1.5]]
test_y = neigh.predict(test_x)

plt.scatter(X, y,  color='blue')
plt.scatter(test_x, test_y,  color='r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
