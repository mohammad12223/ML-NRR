Standardization: 

Preprocessing operation on data before feeding to the neural network is very important. In such a data set that ranges of features are significantly different, it seems not to be reasonable to feed those raw features to network. One of the best method to deal with this problem is using normalization process. In this process the data (each column in data) is centered around with zero mean and unit variance. It can be easily done in Python as follows:

mean = data.mean(axis=0)
data -= mean
std = data.std(axis=0)
data /= std

or 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(data)

we found in DNN (deep neural network) that the catalytic activity is mainly governed by seven features: PCA1, electro-negativity of TM (?), atomic number of TM (Z), atomic radius of TM (R in pm), coordination number of TM (N_C), fraction of B atoms among coordinating atoms in TM-B_xC_y-Gr [f_B=x/(x+y)], and number of nitrogen atoms adsorbed on TM (for side-on and end-on mechanisms-N_M=2/1).


Adam optimizer was used in DNN that can be summarized as follows:
 
Initialize m = 0 as the “first moment”, and v = 0 as the “second moment.” Good default settings for the tested machine learning problems are ß1 = 0.9 and ß2 = 0.999. 
e should be sufficiently small and lr_t (0.001) is learning rate. Initialize t = 0. 

Do until stopping criterion is met:

(Get gradients at timestep t): g
m? ß1 · m + (1 - ß1) · g (First update moment) 
v ? ß2 · v + (1 - ß2) · g•g (Second update moment) 
m_t? m/(1 – ß1 ) (Bias correction in first moment) 
v_t ? v/(1 – ß2 ) (Bias correction in second moment) 
?_t = ? - lr_t * m_t / (v(v_t) +e) (Update parameters) 

return ?_t 


It is worth noting you do not need to write a code for Adam Optimizer by using Keras library as follows: 

DNN

import keras
from keras import metrics
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense,Dropout,BatchNormalization
from keras import layers
import matplotlib.pyplot as plt
from keras import models
from keras import optimizers



regressor = Sequential()
# hidden layer
regressor.add(Dense(units =10, activation = ‘tanh',kernel_regularizer=regularizers.l2(0.003), input_dim = X.shape[1]))
# hidden layer
regressor.add(Dense(units =10, activation = 'tanh', kernel_regularizer=regularizers.l2(0.003) ))
# output layer
regressor.add(Dense(units = 1,activation = 'sigmoid'))
#compile DNN
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['acc'])
# split into input (X) and output (Y) variables # Fit the model
history=regressor.fit (X,Y,batch_size = 1, epochs =50, validation_split=0.15)
# save model and architecture to single file
regressor.save('nnr_weights.h5')
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,50 + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



by loading the weights of the trained network you can predict new compounds as follows:


model=load_model('nnr_weights.h5')
print(model.summary())
A=model.predict(X_1)
the summary of the model is as follows:

number of neurons in first layer:10
number of neurons in first layer:10

Total params: 201
Trainable params: 201
Non-trainable params: 0



Adam Optimizer

Some of the hyperparameters used for LightGBM regressor model are as follows:
max_depth =3
random_state=42,
n_estimators = 582 
min_child_weight =3
learning rate = 0.069

