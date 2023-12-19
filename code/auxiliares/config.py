#################################
############# PERIODOS ##########
#################################
PERIODOS_EXCLUIR=[]#[201908]

PERIODO_INICIO_PARAM=201701 #inclusive
TRAIN_PERIODO_LIMITE_PARAM = 201810 #inclusive
VALIDATE_PERIODO_PARAM = 201812

USAR_HOLDOUT_PARAM=True
TRAIN_ALL_PERIODO_LIMITE_PARAM = 201812 #inclusive
HOLDOUT_PERIODO_PARAM = 201902

MODELO_FINAL_PERIODO_LIMITE_PARAM = 0 #201812
FUTURE_PERIODO_PARAM = 0 #201902

ARCH_PRODUCTOS_PREDICCION_ENTRENAMIENTO="datasets/productos_a_predecir_201902.csv"
ARCH_PRODUCTOS_PREDICCION_FUTURE="" #"datasets/productos_a_predecir_201912.csv"


#################################
######## TRANSFORMACIONES #######
#################################
#TIPO_TRANSF_PARAM = "normalizacion"
#TIPO_TRANSF_PARAM = "estandarizacion"

#################################
###### FEATURE ENGINEERING ######
#################################
NUM_LAGS_PARAM = 5
FAMILIA_FEATURES_TEMP_PARAM =["lags"]#,"delta_lags","rolling_mean","rolling_std","rolling_sum","bollinger_bands"]
AMPLIA_FEATURES_PARAM=[]#["cat1","cat2","cat3"]#,"customer_id","product_id"]


###########################
###### ENTRENAMIENTO ######
###########################
CANT_ITERACIONES_OPT_BAY_PARAM = 10

MAX_BIN_PARAM=255
#MAX_BIN_PARAM=1023

OBJECTIVE_PARAM = "tweedie"
#OBJECTIVE_PARAM = "regression"
#OBJECTIVE_PARAM = "regression_l1"


LEARNING_RATE_LOWER_PARAM=0.01
LEARNING_RATE_UPPER_PARAM=0.3

FEATURE_FRACTION_LOWER_PARAM=0.2
FEATURE_FRACTION_UPPER_PARAM=1.0

MIN_DATA_IN_LEAF_LOWER_PARAM=0
MIN_DATA_IN_LEAF_UPPER_PARAM=8000

NUM_LEAVES_LOWER_PARAM=10
NUM_LEAVES_UPPER_PARAM=1024

L1_UPPER_PARAM=1000.0
L2_UPPER_PARAM=1000.0

CANT_SEMILLAS=4
OFFSET_EVAL_HOLDOUT = 0
CANT_EVAL_HOLDOUT = 5
