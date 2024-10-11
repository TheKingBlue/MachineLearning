import numpy as np
from tensorflow import keras
import tensorflow as tf


def load_model():
    # Deze methode laadt het getrainde model dat je bij de vorige opgavenset heb
    # opgeslagen. 

    model = tf.keras.models.load_model('model-deel2.keras')
    return model

# OPGAVE 1a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    
    confusius_matrix = tf.math.confusion_matrix(labels,pred)
    return confusius_matrix
    

# OPGAVE 1b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
    x = list()
    # tpi = Cii
    tp = [conf[i][i] for i in range(len(labels))]

    # fpi = (som van Cli) - tpi
    fp = [np.sum(conf[:, i])- tp[i] for i in range(len(labels))]

    # (som van Cil) - tpi 
    fn = [np.sum(conf[i, :]) - tp[i] for i in range(len(labels))]

    # (som van Clk) - 
    tn = [np.sum(conf) - tp[i] - fp[i] - fn[i] for i in range(len(labels))]

    c = list(zip(labels, tp, fp, fn, tn))
    # formatted_data = [f"Category: {label}, tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn})" for label, tp, fp, fn, tn in c]

    return c
    

# OPGAVE 1c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    labels, tp, fp, fn, tn = zip(*metrics)

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)
    tn = np.sum(tn)
    print(fn, fp)
    tpr = np.round(np.divide(tp, tp+fn), decimals=4)
    ppv = np.round(np.divide(tp, tp+fp), decimals=4)
    tnr = np.round(np.divide(tn, tn+fp), decimals=4)
    fpr = np.round(np.divide(fp, fp+tn), decimals=4)

    rv = {'tpr':tpr, 'ppv':ppv, 'tnr':tnr, 'fpr':fpr}
    return rv