# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

def build_cnn_model():
    """Constrói e retorna o modelo CNN sequencial."""
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                     activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    
    print("Modelo CNN construído.")
    return model