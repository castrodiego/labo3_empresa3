def crear_nuevos_features(df_param):
    # Diferencia entre lo solicitado y lo entregado
    df_param["dif_cust_request_tn"]=df_param.cust_request_tn-df_param.tn
    df_param["dif_cust_request_tn_porc"]=np.where(df_param.cust_request_tn==0,0,100*df_param.dif_cust_request_tn/df_param.cust_request_tn)
   
    return df_param

def crear_features_temporales(campo,df_param,num_lag_params):
    # Primero y FUNDAMENTAL, ordeno por el campo y periodo
    df_param = df_param.sort_values(by=[campo,"periodo"],ascending=True)

    if campo=="prod_cust":
        prefijo_tn = ""
    else:
        prefijo_tn = campo + "_"
    for i in range(1, num_lag_params + 1):
        df_param[campo + "_tn_lag_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].shift(i)
        df_param[campo + "_tn_delta_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].diff(i)
        df_param[campo + "_rolling_std_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].rolling(window=i).std().reset_index(level=0, drop=True)
        df_param[campo + "_rolling_mean_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].rolling(i).mean().reset_index(level=0, drop=True)
        df_param[campo + "_rolling_sum_" + str(i)] = df_param.groupby(campo)[prefijo_tn + "tn"].rolling(window=i).sum().reset_index(level=0, drop=True)

    return df_param