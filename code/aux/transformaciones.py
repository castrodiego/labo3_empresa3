import numpy as np

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
