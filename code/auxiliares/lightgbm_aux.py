import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import metricas
import random
import numpy as np
import extras


def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

def semillerio(lgbtrain_all, lgb_params, best_iteration, X, cant_semillas):
    y_pred_list = []
    
    for i in range(0,cant_semillas):
        seed = random.randint(1,99999999999999)
        lgb_params_new = {**lgb_params, **{"seed":seed}}

        final_model = lgb.train(lgb_params_new, lgbtrain_all, num_boost_round=best_iteration)

        y_pred = final_model.predict(X)
        
        y_pred_list.append(y_pred)

    y_pred_semillerio = np.sum(y_pred_list,axis=0)/cant_semillas
    y_pred_semillerio = extras.limpiar_prediccion(y_pred_semillerio)
    return y_pred_semillerio