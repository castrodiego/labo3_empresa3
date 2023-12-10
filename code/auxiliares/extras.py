import numpy as np

def limpiar_prediccion(y_pred):
    #Se ponen como 0 las predicciones negativas
    return  np.maximum(y_pred, 0)

def convertir_categoricas_prod(df_param):
    #Transformo todas las categoricas
    categories = ["product_id","cero_ventas","producto_estrella","plan_precios_cuidados","cat1","cat2","cat3","catastrofe"]
    
    for c in categories: 
        df_param[c] = df_param[c].astype("category")   

    return df_param
