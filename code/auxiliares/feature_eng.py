import numpy as np

def crear_target_prod_cust(df_param):
    df_param = df_param.sort_values(by=["prod_cust","periodo"],ascending=True)
    df_param["tn_mas_2"] = df_param.groupby("prod_cust")["tn"].shift(-2)
    return df_param

def crear_target_product(df_param):
    df_param = df_param.sort_values(by=["product_id","periodo"],ascending=True)
    df_param["tn_mas_2"] = df_param.groupby("product_id")["tn"].shift(-2)
    return df_param
    
def crear_nuevos_features(df_param):
    # Diferencia entre lo solicitado y lo entregado
    df_param["dif_cust_request_tn"]=df_param.cust_request_tn-df_param.tn
    df_param["dif_cust_request_tn_porc"]=np.where(df_param.cust_request_tn==0,0,100*df_param.dif_cust_request_tn/df_param.cust_request_tn)
   
    return df_param

def crear_features_temporales(campo,df_param,num_lag_params,familia_features_temporales):
    # Primero y FUNDAMENTAL, ordeno por el campo y periodo
    df_param = df_param.sort_values(by=[campo,"periodo"],ascending=True)

    if campo in ["prod_cust","product_id"]:
        prefijo_tn = ""
    else:
        prefijo_tn = campo + "_"
        
    for i in range(1, num_lag_params + 1):
        if "lags" in familia_features_temporales:
            df_param[campo + "_tn_lag_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].shift(i)
        
        if "delta_lags" in familia_features_temporales:
            df_param[campo + "_tn_delta_lag_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].diff(i)

        if "rolling_std" in familia_features_temporales:
            df_param[campo + "_tn_rolling_std_" + str(i)] = df_param.groupby(campo)[prefijo_tn+"tn"].rolling(window=i+1).std().reset_index(level=0, drop=True)
            
        if "rolling_mean" in familia_features_temporales:
            df_param[campo + "_tn_rolling_mean_" + str(i)] = df_param.groupby(campo)[prefijo_tn +"tn"].rolling(window=i+1).mean().reset_index(level=0, drop=True)
            
        if "rolling_sum" in familia_features_temporales:
            df_param[campo + "_tn_rolling_sum_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].rolling(window=i+1).sum().reset_index(level=0, drop=True)

        if "bollinger_bands" in familia_features_temporales:
            df_param[campo + "_tn_bollinger_band_upper_" + str(i)] = df_param[campo + "_tn_rolling_mean_" + str(i)] + 2*df_param[campo + "_tn_rolling_std_" + str(i)]
            df_param[campo + "_tn_bollinger_band_lower_" + str(i)] = df_param[campo + "_tn_rolling_mean_" + str(i)] - 2*df_param[campo + "_tn_rolling_std_" + str(i)]

    return df_param

