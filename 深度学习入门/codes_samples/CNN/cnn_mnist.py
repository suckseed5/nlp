from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pdb

def data_processing():
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()

    img=test_images[0]
    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()

    train_images=train_images.reshape((60000,28,28,1))
    train_images=train_images.astype('float32')/255

    test_images=test_images.reshape((10000,28,28,1))
    test_images=test_images.astype('float32')/255

    train_labels=to_categorical(train_labels)
    test_labels=to_categorical(test_labels)

    return train_images,train_labels,test_images,test_labels

def create_model():
    model=models.Sequential()
    # 第1层卷积，卷积核大小为3*3，32个卷积核，28*28为待训练图片的大小
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    # 池化层
    model.add(layers.MaxPooling2D((2,2)))
    # 第2层卷积，卷积核大小为3*3，64个卷积核
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    # 池化层
    model.add(layers.MaxPooling2D((2,2)))
    # 第3层卷积，卷积核大小为3*3，64个卷积核
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    # 拉成1维形状
    model.add(layers.Flatten()) #3D->1D
    # 第4层全连接层，64个神经元
    model.add(layers.Dense(64,activation='relu'))
    # 第5层全连接层，10个神经元，sofmax多用于分类
    model.add(layers.Dense(10,activation='softmax'))
    model.summary()
    
    return model

def train(model,train_images,train_labels,test_images,test_labels):
    print(train_images.shape)
    print(train_labels.shape)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=1,batch_size=64)
    model.save('cnn.h5')
    test_loss,test_acc=model.evaluate(test_images,test_labels)
    print('loss=',test_loss)
    print('acc=',test_acc)
    

def predict(model,test_images):
    #model=load_model('cnn.h5')
    predictions=model.predict(test_images)
    pdb.set_trace()
    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))

pdb.set_trace()
train_images,train_labels,test_images,test_labels=data_processing()
model=create_model()
train(model,train_images,train_labels,test_images,test_labels)
predict(model,test_images)
