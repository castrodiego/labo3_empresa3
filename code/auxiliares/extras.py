import numpy as np

CATEGORIAS_BASE = ["product_id","cero_ventas","producto_estrella","plan_precios_cuidados","cat1","cat2","cat3","catastrofe"]

def limpiar_prediccion(y_pred):
    #Se ponen como 0 las predicciones negativas
    return  np.maximum(y_pred, 0)

def convertir_categoricas_prod(df_param):
    #Transformo todas las categoricas
    categories = CATEGORIAS_BASE
    cols = df_param.columns
    
    for c in categories: 
        if c in cols:
            df_param[c] = df_param[c].astype("category")   

    return df_param

def convertir_categoricas_prod_cust(df_param):
    #Transformo todas las categoricas
    categories = CATEGORIAS_BASE
    categories.append("prod_cust")
    
    for c in categories: 
        df_param[c] = df_param[c].astype("category")   

    return df_param