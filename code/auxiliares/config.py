#################################
############# PERIODOS ##########
#################################
PERIODO_INICIO_PARAM=201701 #inclusive
TRAIN_PERIODO_LIMITE_PARAM = 201810 #inclusive
VALIDATE_PERIODO_PARAM = 201812
TRAIN_ALL_PERIODO_LIMITE_PARAM = 201812 #inclusive
HOLDOUT_PERIODO_PARAM = 201902

#################################
######## TRANSFORMACIONES #######
#################################
#TIPO_TRANSF_PARAM = "sin_transformacion"
TIPO_TRANSF_PARAM = "normalizacion"
#TIPO_TRANSF_PARAM = "estandarizacion"

#################################
###### FEATURE ENGINEERING ######
#################################
NUM_LAGS_PARAM = 15

###########################
###### ENTRENAMIENTO ######
###########################
CANT_ITERACIONES_OPT_BAY_PARAM = 20

MAX_BIN_PARAM=255
#MAX_BIN_PARAM=1023

OBJECTIVE_PARAM = "tweedie"
#OBJECTIVE_PARAM = "regression"

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
