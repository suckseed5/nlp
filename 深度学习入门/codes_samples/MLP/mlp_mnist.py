from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def data_processing():
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()

    img=test_images[0]
    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()

    train_images=train_images.reshape((60000,28*28))
    train_images=train_images.astype('float32')/255

    test_images=test_images.reshape((10000,28*28))
    test_images=test_images.astype('float32')/255

    train_labels=to_categorical(train_labels)
    test_labels=to_categorical(test_labels)

    return train_images,train_labels,test_images,test_labels

def create_model():
    model=models.Sequential()
    model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
    model.add(layers.Dense(10,activation='softmax'))
    model.summary()
  
    return model

def train(model,train_images,train_labels,test_images,test_labels):
    print(train_images.shape)
    print(train_labels.shape)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=10,batch_size=128)
    model.save('mlp.h5')
    test_loss,test_acc=model.evaluate(test_images,test_labels)
    print('loss=',test_loss)
    print('acc=',test_acc)
    

def predict(model,test_images):
    #model=load_model('cnn.h5')
    predictions=model.predict(test_images)

    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))

train_images,train_labels,test_images,test_labels=data_processing()
model=create_model()
train(model,train_images,train_labels,test_images,test_labels)
predict(model,test_images)
