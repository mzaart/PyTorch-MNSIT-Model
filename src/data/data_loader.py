import os
import pandas as pd
from torch.utils.data import DataLoader
from .data_sets import HeartDataSet, ShuffledDataSet
from .tranforms import MinMaxScaler
from .data_set_samplers import DataSetSamplers
from ..config import DATA_DIR, DATA_SET_NAME, NUM_COLS, BATCH_SIZE, NUM_WORKERS

df = pd.read_csv(os.path.join(DATA_DIR, DATA_SET_NAME))
print('Dataframe shape: ', df.shape)
np_array = df.to_numpy()
transform = MinMaxScaler.from_features(np_array[:, :np_array.shape[1] - 1])

data_set = HeartDataSet(np_array)
data_set = ShuffledDataSet(data_set)
data_set_samplers = DataSetSamplers(data_set)

training_data_loader = DataLoader(data_set, sampler=data_set_samplers.training_sampler,
                                  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
validation_data_loader = DataLoader(data_set, sampler=data_set_samplers.validation_sampler,
                                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_data_loader = DataLoader(data_set, sampler=data_set_samplers.test_sampler,
                              batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
