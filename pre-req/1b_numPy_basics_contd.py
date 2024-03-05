import numpy as np
print(np.__version__)

#arrays are 0-indexed in python aswell
arr = np.array([1,2,3])
print(arr[1])

arr2 = np.array([[1,2,3,7],[2,7,1,8]])
print(arr2[:, 2])                         #prints 3rd col

#printing 0th row
# print(arr2[0])
print(arr2[0,:])

#printing specific elements in index range
print(arr2[0, 1:])
print(arr2[0, 1:3])
# for ':', it only half-open, doesnt includes b in range [a,b)

#array reversal
print(arr2)
print(arr2[::-1])

#array reshaping - element count must be same for both
arr2_new = arr2.reshape(4,2)                #used in Deep learning
print(arr2_new)

#to add an element to arr
arr2 = np.append(arr2, 12)
print(arr2)

#deletion of an element requires the position to be specified
arr2 = np.delete(arr2, 9)
print(arr2)

# flatten operation on arrays
arr3 = arr2.ravel()             #returns simple vector - for image processing
print(arr3)

#vs reshape
arr3 = arr2.reshape(1,9)        #creates arr of arr
print(arr3)

#array concatenation
arr1 = [1,2,3]
arr2 = [5,6,9]

arr_new = np.concatenate((arr1,arr2))
print(arr_new)

#transpose
arr = [(1,2,4,5),(7,4,2,8)]
print(np.transpose(arr))

#copy opr. - when performing opr. on image, we require to save a instance of image
arr1 = np.copy(arr)
print(arr1)