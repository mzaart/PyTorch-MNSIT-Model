import os


# directory variables

PROJECT_ROOT = os.environ.get('PROJECT_ROOT')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# data set variables

DATA_SET_NAME = 'heart.csv'

NUM_COLS = 14
NUM_FEATURES = NUM_COLS - 1


# data loading variables

DEFAULT_TRAINING_SET_SIZE = 0.6
DEFAULT_VALIDATION_SET_SIZE = 0.2
DEFAULT_TEST_SET_SIZE = 0.2

BATCH_SIZE = 64


# worker variables

NUM_WORKERS = 4


# training variables

NUM_EPOCHS = 100
