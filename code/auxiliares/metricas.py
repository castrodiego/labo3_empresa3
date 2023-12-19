import numpy as np
from IPython.display import Markdown, display


def printmd(string):
    display(Markdown(string))

def error_rate(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    #Las predicciones negativas se convierten a 0
    y_pred = np.maximum(y_pred, 0)
    
    dif_abs = sum(abs(y - y_pred))
    suma_real = sum(y)

    if suma_real > 0:
        er = round(100*dif_abs/suma_real,2)
    else:
        er = 0
        
    return er

def print_error_rate_total_y_cat(df_pred):
    error_rate_total = error_rate(df_pred.tn_real,df_pred.tn_pred)
    printmd("**Error Rate Total:** " + str(error_rate_total))

    for cat1_iter in df_pred.cat1.unique():
        df_pred_cat = df_pred[df_pred.cat1 == cat1_iter]
        
        error_rate_cat = error_rate(df_pred_cat.tn_real,df_pred_cat.tn_pred)
        printmd("**Error Rate por Categoría " + cat1_iter + "**: " + str(error_rate_cat))
