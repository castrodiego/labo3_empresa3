import numpy as np
import config

## Normalizacion
def normalizar_valor(valor,minimo,maximo):
    if np.isnan(minimo):
        return valor #si no hay minimo, devuelvo valor original (no tener minimo significa que no habia datos en train)
    else:
        if (maximo-minimo)==0: 
            return 0
        else:
            return (valor - minimo) / (maximo-minimo)
        
def desnormalizar_valor(valor,minimo,maximo):
    if np.isnan(minimo):
        return valor #si no hay minimo, devuelvo valor original (no tener minimo significa que no habia datos en train)
    else:
        return (valor * (maximo-minimo)) + minimo

## Estandarizacion
def estandarizar_valor(valor,media,desvio):
    if np.isnan(media):
        return valor #si no hay media, devuelvo valor original (no tener media significa que no habia datos en train)
    else:
        if desvio==0: 
            return 0
        else:
            return (valor - media) / desvio

def desestandarizar_valor(valor,media,desvio):
    if np.isnan(media):
        return valor #si no hay media, devuelvo valor original (no tener media significa que no habia datos en train)
    else:
        return (valor * desvio) + media

## Transformacion
def transformar_valor(tipo_transf, valor,var1,var2):
    if tipo_transf=="normalizacion":
        return normalizar_valor(valor,var1,var2)
    elif tipo_transf=="estandarizacion":
        return estandarizar_valor(valor,var1,var2)
    else:
        return valor

def destransformar_valor(tipo_transf,valor,var1,var2):
    if tipo_transf=="normalizacion":
        return desnormalizar_valor(valor,var1,var2)
    elif tipo_transf=="estandarizacion":
        return desestandarizar_valor(valor,var1,var2)
    else:
        return valor

def error_rate_transf(y, y_pred):
    y_pred = np.array(y_pred)
    
    #Destransformo
    df_calculo = GLOBAL_PRODUCT_IDS.copy()
    df_calculo["y"] = y
    df_calculo["y_destransformado"]=df_calculo.apply(lambda row: destransformar_valor(config.TIPO_TRANSF_PARAM,row["y"],row["valor_1"],row["valor_2"]),axis=1)

    df_calculo["y_pred"] = y_pred
    df_calculo["y_pred_destransformado"]=df_calculo.apply(lambda row: destransformar_valor(config.TIPO_TRANSF_PARAM,row["y_pred"],row["valor_1"],row["valor_2"]),axis=1)

    y = df_calculo.y_destransformado
    y_pred = df_calculo.y_pred_destransformado
    
    #Las predicciones negativas se convierten a 0
    y_pred = np.maximum(y_pred, 0)
    
    dif_abs = sum(abs(y - y_pred))
    suma_real = sum(y)
    return round(100*dif_abs/suma_real,2)

