import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 8)
y = np.array([0, 34, 45, 56, 78, 123, 156, 200])

# Create matrix to run all samples to our y = mx + b linear function
A = np.vstack([x, np.ones(len(x))]).T
print(A)

# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
m, c = np.linalg.lstsq(A, y)[0]
print(m, c)

# plot everything
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()

