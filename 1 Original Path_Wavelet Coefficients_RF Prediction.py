
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Ingnore Warning
warnings.filterwarnings("ignore")
#  "Beautify using built-in styles.
plt.style.use('fivethirtyeight')

from sklearn.model_selection import train_test_split
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # Specify default font.
mpl.rcParams['axes.unicode_minus'] = False #  Resolve the issue of negative sign '-' displaying as a square when saving the image.

a00 = r"b00.csv"
y00 = r'y00.csv'

# a00 = r"b01.csv"
# y00 = r'y01.csv'
#
# a00 = r"b02.csv"
# y00 = r'y02.csv'

# a00 = r"aa.csv"
# y00 = r'aa00.csv'

class ParkinsonLoader():
    def __init__(self, path1,path2):
        parkinson_x = pd.read_csv(path1)
        parkinson_dataset_y = pd.read_csv(path2)


        # parkinson_dataset = parkinson_dataset.dropna()
        columns = ['Unnamed: 0']
        parkinson_x = parkinson_x.drop(columns, axis=1)

        columns = ['0']
        parkinson_y = pd.DataFrame(parkinson_dataset_y, columns=columns)

        dataset = parkinson_x.values
        parkinson_x = dataset.astype('float32')

        dataset = parkinson_y.values
        parkinson_y = dataset.astype('float32')

        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(parkinson_x, parkinson_y,test_size=0.1, random_state=67,shuffle=True) #, random_state=5

        self.num_train_data = np.shape(self.X_train)[0]
        self.num_test_data = np.shape(self.X_test)[0]
    def train_xy(self):
        self.x=self.X_train
        self.y=self.y_train
        return self.x, self.y
    def test_xy(self):
        self.x_test = self.X_test
        self.y_test = self.y_test
        return self.x_test, self.y_test


data_loader = ParkinsonLoader(a00,y00)
# print(data_loader.num_train_data)
x, y = data_loader.train_xy()
print(x.shape)
test_x, test_y = data_loader.test_xy()

trainYYY=pd.DataFrame(y)
trainYYY.to_csv("trainY.csv")
trainYYA=pd.DataFrame(test_y)
trainYYA.to_csv("testY.csv")

from sklearn.preprocessing import StandardScaler,MinMaxScaler

scale=MinMaxScaler()
trainX = scale.fit_transform(x)
testX = scale.transform(test_x)

scale1 = MinMaxScaler()
trainY = scale1.fit_transform(y)
test_y = scale1.transform(test_y)

import pywt

#  "2-layer decomposition."

def wavelet(trainX):
    train = []
    for i in range(trainX.shape[0]):
        train1 = trainX[i, :]

        coeffs = pywt.wavedec(train1, 'db2', level=2, axis=0)
        fenliang = np.concatenate(coeffs)  # Use concatenate instead of extend

        train.append(fenliang)

    train = np.array(train)
    return train



trainXX = wavelet(trainX).reshape((trainX.shape[0], -1))
print(trainXX.shape)


testX =wavelet(testX).reshape((testX.shape[0],-1))

num_tree=30

from sklearn import datasets,ensemble
# rf_model = ensemble.RandomForestRegressor(n_estimators=num_tree,max_depth=50,criterion='mse')
rf_model = ensemble.RandomForestRegressor(n_estimators=num_tree, max_depth=50, criterion='squared_error')

rf_model.fit(trainXX,trainY)

pre = rf_model.predict(testX)
pre =pre.reshape((-1,1))

pre_train = np.array([tree.predict(trainXX) for tree in rf_model.estimators_])
print("pre_train:",pre_train.shape)
pre_test = np.array([tree.predict(testX) for tree in rf_model.estimators_])
print("pre_test:",pre_test.shape)

pre_train = pre_train.T
print("pre_train:",pre_train.shape)
pre_test = pre_test.T
print("pre_test:",pre_test.shape)

trainYY=pd.DataFrame(pre_train)
trainYY.to_csv("original_train_tree_pre.csv") #"original" or "primitive"
trainYY=pd.DataFrame(pre_test)
trainYY.to_csv("original_test_tree_pre.csv")







