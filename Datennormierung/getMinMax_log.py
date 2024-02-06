import torch
import h5py
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler

#file_path = '/home/kali/Projects/Semesterprojekt/rzp-1_sphere1mm_train_100k.h5'
#file_path = '/vol/fob-vol7/mi21/arendtda/Sempro/rzp-1_sphere1mm_train_100k.h5'


file_path = '/vol/fob-vol7/mi21/arendtda/Sempro/rzp-1_sphere1mm_train_2million.h5'
#file_path = '/vol/fob-vol7/mi21/arendtda/Sempro/rzp-1_sphere1mm_train_2million_bin32.h5'

minsX = [float('inf')] * 7
maxsX = [float('-inf')] * 7
minsY = [float('inf')]
maxsY = [float('-inf')]

ySize = 64

def getMinMaxX(file_path):
    h5 = h5py.File(file_path, 'r')
    for i in range (7):
        for group in h5:
            value = np.array(h5[group]['X'][i])
            
            #check max
            if value > maxsX[i]:
                maxsX[i] = value
                #print(f"Max: {value}, Iteration: {i}")

            #check min
            if value < minsX[i]:
                minsX[i] = value
                #print(f"Min: {value}, Iteration: {i}")
                
    h5.close()

def getMinMaxY(file_path):
    h5 = h5py.File(file_path, 'r')
    for group in h5:
        print(h5[group])

        #Max herausfinden
        stelle = np.argmax(h5[group]['Y'])
        zeile = np.floor(stelle / ySize)
        spalte = stelle % ySize
        value = h5[group]['Y'][int(zeile)][int(spalte)]

        #Min herausfinden
        stelleMin = np.argmin(h5[group]['Y'])
        zeileMin = np.floor(stelleMin / ySize)
        spalteMin = stelleMin % ySize
        valueMin = h5[group]['Y'][int(zeileMin)][int(spalteMin)]

        #check max
        if value > maxsY[0]:
            maxsY[0] = value
            print(f"Max: {value}, Gruppe: {group}")
        
        #check min
        if valueMin < minsY[0]:
            minsY[0] = valueMin
            print(f"Max: {value}, Gruppe: {group}")
                
    h5.close()

def createScaler():
    xMms = MinMaxScaler(feature_range=(0,1))
    yMms = MinMaxScaler(feature_range=(0,1))
    x = [minsX, maxsX]
    y = [minsY * ySize, maxsY * ySize]
    xMms.fit(x)
    yMms.fit(y)

    return xMms, yMms

def minmax_transfer(old_file_path, new_file_path):
    newFile = h5py.File(new_file_path, 'w')
    oldFile = h5py.File(old_file_path, 'r')

    xScaler, yScaler = createScaler()

    for group in oldFile:
        print(group)
        neueGruppe = newFile.create_group(f"{group}")

        x_data = torch.Tensor(oldFile[group]['X'][:7])
        x_data = x_data.reshape(1,-1)

        neueGruppe.create_dataset('X', data = xScaler.transform(x_data), compression="gzip", compression_opts=9)
        neueGruppe.create_dataset('Y', data = yScaler.transform(oldFile[group]['Y']), compression="gzip", compression_opts=9)

    newFile.close()
    oldFile.close()


def log_transfer(old_file_path, new_file_path):
    newFile = h5py.File(new_file_path, 'w')
    oldFile = h5py.File(old_file_path, 'r')

    xScaler, yScaler = createScaler()

    for group in oldFile:
        print(group)
        neueGruppe = newFile.create_group(f"{group}")

        x_data = torch.Tensor(oldFile[group]['X'][:7])
        x_data = x_data.reshape(1,-1)

        neueGruppe.create_dataset('X', data = xScaler.transform(x_data), compression="gzip", compression_opts=9)
        neueGruppe.create_dataset('Y', data = np.where(oldFile[group]['Y'] != 0, np.log2(oldFile[group]['Y']), oldFile[group]['Y']), compression="gzip", compression_opts=9)

    newFile.close()
    oldFile.close()


#getMinMaxX(file_path)
#getMinMaxY(file_path)
#print("Der größte y Wert ist:")
#print(maxsY)
#print("Der kleinste y Wert ist:")
#print(minsY)

minsX = [70, 1.5, 300, 0, 0, 3000, 0.5]
maxsX = [150, 3, 800, 100, 100, 10000, 3]
minsY = [0]
#maxsY = [15526] #auf 100k
maxsY = [17211] #auf 2mio
#maxsY = [33787]  #auf 2mio bin32

#minmax_transfer(file_path,'/vol/fob-vol7/mi21/arendtda/Sempro/normed_data_2mio.h5')
log_transfer(file_path,'/vol/fob-vol7/mi21/arendtda/Sempro/log_normed_data_2mio.h5')
