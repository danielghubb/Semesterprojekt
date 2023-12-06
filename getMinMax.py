import h5py
import numpy as np

file_path = '/home/kali/Projects/Semesterprojekt/rzp-1_sphere1mm_train_100k.h5'
mins = [float('inf')] * 7
maxs = [float('-inf')] * 7


def getMinMax(file_path):
    h5 = h5py.File(file_path, 'r')
    for i in range (7):
        for group in h5:
            value = np.array(h5[group]['X'][i])
            
            #check max
            if value > maxs[i]:
                maxs[i] = value
                print(f"Max: {value}, Iteration: {i}")

            #check min
            if value < mins[i]:
                mins[i] = value
                print(f"Min: {value}, Iteration: {i}")
                
    h5.close()

getMinMax(file_path)

print(mins)
print(maxs)
