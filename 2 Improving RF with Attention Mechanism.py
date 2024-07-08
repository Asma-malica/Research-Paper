
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
from sklearn.metrics import mean_squared_error
import math

a1 = r"original_train_tree_pre.csv"
a2 = r"shortest_train_tree_pre.csv"
a3 = r"longest_train_tree_pre.csv"
ytrain= r"trainY.csv"

parkinson_x = pd.read_csv(a1)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
x1 = dataset.astype('float32')

parkinson_x = pd.read_csv(a2)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
x2 = dataset.astype('float32')

parkinson_x = pd.read_csv(a3)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
x3 = dataset.astype('float32')

parkinson_x = pd.read_csv(ytrain)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
ytrain = dataset.astype('float32')


a11 = r"original_test_tree_pre.csv"
a22 = r"shortest_test_tree_pre.csv"
a33 = r"longest_test_tree_pre.csv"
ytest= r"testY.csv"


parkinson_x = pd.read_csv(a11)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
x11 = dataset.astype('float32')

parkinson_x = pd.read_csv(a22)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
x22 = dataset.astype('float32')

parkinson_x = pd.read_csv(a33)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
x33 = dataset.astype('float32')

parkinson_x = pd.read_csv(ytest)
columns = ['Unnamed: 0']
parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
ytest = dataset.astype('float32')

print(x1.shape)
print(x2.shape)
print(x3.shape)

print(x11.shape)
print(x22.shape)
print(x33.shape)

train=np.hstack((x1,x2))
train=np.hstack((train,x3))
print(train.shape)

test=np.hstack((x11,x22))
test=np.hstack((test,x33))
print("test.shape",test.shape)

from sklearn.preprocessing import StandardScaler,MinMaxScaler

scale=MinMaxScaler()
train = scale.fit_transform(train)
test = scale.transform(test)

scale1 = MinMaxScaler()
ytrain = scale1.fit_transform(ytrain)
ytest = scale1.transform(ytest)


num=train.shape[1]

SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[1])
    print("input:",input_dim)
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
    print(a.shape)

    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        print(a.shape)
        a = RepeatVector(inputs.shape[1])(a)
        print(a.shape)
    # a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a], name='attention_mul')

    return output_attention_mul

def model_attention_applied_before_lstm():
    inputs = Input(shape=(num,))

    attention_mul = attention_3d_block(inputs)
    print(attention_mul.shape)
    output1 = Dense(32, activation='relu')(attention_mul)
    output = Dense(1, activation='relu')(output1)     #sigmoid, tanh, relu, LeakyReLU(alpha=0.05)()
    model = Model(inputs=[inputs], outputs=output)
    return model


from keras.callbacks import TensorBoard
model=model_attention_applied_before_lstm()
model.compile(optimizer='adam', loss='mae', metrics=['mse'])
model.fit(train,ytrain, epochs=500, batch_size=16,verbose=2, callbacks=None)

pre1=model.predict(test)
print(pre1.shape)

testPredict = scale1.inverse_transform(pre1)
testY = scale1.inverse_transform(ytest)
# testPredict =pre1
# testY=ytest

trainYY=pd.DataFrame(testPredict)
trainYY.to_csv("original_test.csv") 
# "original"
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:, 0]))
print('Test Score: %.3f RMSE' % (testScore))
test_mape = (np.abs((testY[:,0] - testPredict[:, 0]))).mean()
print('Test Score: %.3f MAE' % (test_mape))
from sklearn.metrics import r2_score
r2 = r2_score(testY, testPredict)
print('Test Score: %.3f R2_score' % (r2))
mape = np.mean(np.abs((testPredict - testY) / testY))
print('Test Score: %.3f mape' % (mape))





