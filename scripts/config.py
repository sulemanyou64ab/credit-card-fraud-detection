"""
Configuration file for credit card fraud detection analysis
"""

# Model Parameters
RFC_METRIC = 'gini'
NUM_ESTIMATORS = 100
NO_JOBS = 4

# Train/Validation/Test Split
VALID_SIZE = 0.20
TEST_SIZE = 0.20

# Cross-Validation
NUMBER_KFOLDS = 5

# Random State
RANDOM_STATE = 2018

# LightGBM Parameters
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 1000
VERBOSE_EVAL = 50

# Feature Names
TARGET = 'Class'
PREDICTORS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
              'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
              'Amount']

# Paths
PLOTS_DIR = '../plots'
DATA_CACHE_DIR = '../data'
