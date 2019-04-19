import os


# directory variables

PROJECT_ROOT = os.environ.get('PROJECT_ROOT')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


# data set variables

RAW_DATA_SET_NAME = 'raw.csv'
FILTERED_DATA_SET_NAME = 'filtered.csv'
PROCESSED_DATA_SET_NAME = 'processed.csv'
NUM_COLS = 3073
ASSIGNED_LABELS = ('5', '7')


# data loading variables

DEFAULT_TRAINING_SET_SIZE = 0.6
DEFAULT_VALIDATION_SET_SIZE = 0.2
DEFAULT_TEST_SET_SIZE = 0.2

BATCH_SIZE = 64


# worker variables

NUM_WORKERS = 4
