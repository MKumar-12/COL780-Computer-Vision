'''NumPy basics'''

import numpy as np
print(np.__version__)         #chk for module version

# Normal row vector
arr1 = np.array([1,2,3])
print(arr1)

# Normal column vector
arr1_vertical = np.array([[1],[2],[3]])
print(arr1_vertical)

# Defining a 2D matrix
arr2 = np.array([(1,2,3),(4,5,6)])
print(arr2)

# Defining a 3D matrix - adding a z-dimension
arr3 = np.array([[(1,2,3),(4,5,6)],[(1,2,1),(1,1,1)]])
print(arr3)

# in similar fashion multi-dimensional matrices can be formed

# defining matrix with def parameters
arr1 = np.zeros((3,3))    #def values -> float
print(arr1)

arr2 = np.zeros((3,3), dtype = "uint8")    #other dtype values -> int, uint
print(arr2)

# dtype can be used with numpy array defining fn. such as .array(), .zeros(), .ones()

arr3 = np.ones((3,5), dtype = "uint")
print(arr3)

arr4 = np.full((2,4), 8)                  #used when we need to apply a kernal or a mask or a convolution to an image
print(arr4)

#defining identity matrix
arr1 = np.eye(3, dtype = "uint8")
print(arr1)

#defining a matrix with some step value for range [a,b)
arr2 = np.arange(0,10,2)
print(arr2)

#to get some desired no. of points within range [a,b]
arr3 = np.linspace(0,2, 9)
print(arr3)

#random is used when we want to add impurities to image :
# (i.e. to generate noise)
arr = np.random.rand(2,3)               # all values lie in range [0,1)
print(arr)

#to scale the values :
arr = np.random.rand(2,3) * 255            # for an img. - scalar multiplication
arr_int = arr.astype(int)
print(arr_int)

#OR to generate random int matrix :
arr2 = np.random.randint(256, size = (3,3))       #generates int values from [a,b)
print(arr2)

#Array inspection - to know about dim. of image
#     #rows = height(y)    #col = width(x)
arr2 = np.array([(1,2,3),(4,5,6)])
print(arr2.shape)

#Pixels count in img
print(arr2.size)

#to know element tpye of arr
print(arr.dtype)

# Arithmetic operation on arrays
arr1 = np.array([(4,3,9),(5,-2,1)])
arr2 = np.array([(1,2,3),(4,5,6)])

arr_sum = arr1 + arr2                 # can help in increasing image brightness
print(arr_sum)

arr_mul = arr1 * arr2
print(arr_mul)

# to normalize arr : we can use fn. like min() & max()
print(arr_mul.max())