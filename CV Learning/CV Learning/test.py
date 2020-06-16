
import numpy as np

x = np.array([[1,2],[4,3]])
print(x)
y= np.argmax(x, axis=-1)
print(y)