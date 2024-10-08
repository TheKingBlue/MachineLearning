import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    reshaped = np.reshape(nrVector, (20,20), order='F')
    plt.matshow(reshaped)
    plt.show()

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    e = np.exp(-z)
    return np.divide(1, (1+e))

# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m × x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0, ... 0] zijn en als
    # y_i=10, dan is regel i in de matrix [0,0, ... 1] (10 symboliseert in deze dataset dus 9 en niet 0!).
    # In dit geval is de breedte van de matrix 10 (0-9),
    # maar de methode moet werken voor elke waarde van y en m

    # -1 to all to make the matrix 0 based
    y_zerobase = y-1
    # Transpose to go from 5000x1 to 1x5000 (A normal 1D array)
    # (Het lijkt alsof hij in de les de data niet in de jusite vorm pakte)
    cols = y_zerobase.T[0] #[0] to get rid of the array-in-array style
    print(cols, y_zerobase.T)
    rows = [i for i in range(m)]
    data = [1 for _ in range(m)]
    # convert to int, then +1 to width because it is 0 based
    width = int(max(y_zerobase))+1
    y_matrix = csr_matrix((data, (rows, cols)), shape=(m, width)).toarray()
    return y_matrix

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta2, Theta3, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta2 en Theta3. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta2 en Theta3 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg een kolom van enen toe vooraan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: Let op: a1 is de activatie van de eerste laag, dus het dotproduct van
    #       de input-vector met de betreffende Theta-matrix. dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    a1 = np.hstack((np.ones((5000, 1)), X))
    z2 = np.dot(a1, Theta2.T)
    o2 = sigmoid(z2)
    a2 = np.hstack((np.ones((5000, 1)), o2))
    z3 = np.dot(a2, Theta3.T)
    a3 = sigmoid(z3)
    return a3

# ===== deel 2: =====
def compute_cost(Theta2, Theta3, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta2 en Theta3) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een ijle matrix. 
    
    # Call the necessary functions
    m,n = X.shape
    y_matrix = get_y_matrix(y, m)
    pred = predict_number(Theta2, Theta3, X)
    
    # Cost function
    K = -1*y_matrix * np.log(pred) - (1-y_matrix)*np.log(1-pred)
    J = 1/m * np.sum(np.sum(K)) # First sum K, each node into a cost per node. Then sum the outcome to sum all nodes into one total cost.
    return J

# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Controleer dat deze werkt met
    # scalaire waarden en met vectoren.

    g = sigmoid(z)
    sigmoid_grad = g * (1-g)
    return sigmoid_grad[0]

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta2, Theta3, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta2.shape)
    Delta3 = np.zeros(Theta3.shape)
    m,n = X.shape
    # Fetch y_vec
    y_vec = get_y_matrix(y, m)

    a1 = np.hstack((np.ones((5000, 1)), X))
    z2 = np.dot(a1, Theta2.T)
    o2 = sigmoid(z2)
    a2 = np.hstack((np.ones((5000, 1)), o2))
    z3 = np.dot(a2, Theta3.T)
    a3 = sigmoid(z3)

    for k in range(m):
        # Outer layer
        delta3 = a3[k] - y_vec[k]
        # Hidden layer
        # delta2 = np.dot(Theta2.T, delta3) * sigmoid_gradient(z2) -> "shapes (401,25) and (10,) not aligned"
        # delta2 = np.dot(Theta2[k].T, delta3) * sigmoid_gradient(z2) -> "shapes (401,) and (10,) not aligned"
        # delta2 = np.dot(Theta3[k].T, delta3) * sigmoid_gradient(z2) -> "Shapes (26,) and (10,) not aligned"
        delta2 = np.dot(Theta3.T[k], delta3) * sigmoid_gradient(z2) # -> 26 is out of range
        # Update martices Delta2 & Delta3
        Delta2 = Delta2 + np.dot(delta2, a2[k][1:]) # Ignore the first node
        Delta3 = Delta3 + np.dot(delta3, a3[k])

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m
    
    return Delta2_grad, Delta3_grad