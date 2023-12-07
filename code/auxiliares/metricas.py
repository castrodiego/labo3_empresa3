import numpy as np

def error_rate(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    #Las predicciones negativas se convierten a 0
    y_pred = np.maximum(y_pred, 0)
    
    dif_abs = sum(abs(y - y_pred))
    suma_real = sum(y)
    
    return round(100*dif_abs/suma_real,2)