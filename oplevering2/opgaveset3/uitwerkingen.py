import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    # YOUR CODE HERE
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()


# OPGAVE 1b
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    # YOUR CODE HERE
    max =  np.amax(X)
    return np.divide(X, max)    

# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.
    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    # Make model
    model = keras.Sequential()

    # Flatten converts a n-dimensional tensor into a 1D tensor, 
    # in this case if Input is used, Flatten should be used after to make a 1D tensor:
    # model.add(keras.Input(shape=(28,28)))
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    
    # relu applies the rectified linear unit activation function 
    # returns element wise the max of (x, 0)
    model.add(keras.layers.Dense(128, activation='relu'))
    
    # softmax converts a vector of values to a probability distribution [0,1]
    model.add(keras.layers.Dense(10, activation='softmax'))

    # adam is a gradient descent method
    # sparse_categorical_crossentropy computes loss between labels and predictions, used when there are more than two labels (which are integers)
    # accuracy calculates how often predictions equal labels
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model

def save_model(model):
    # Deze methode ontvangt het model dat je hierboven hebt gemaakt en dat door 
    # het exercise-script wordt getraind, en slaat het op op een locatie op je lokale
    # computer.

    model.save("model-deel2.keras")
