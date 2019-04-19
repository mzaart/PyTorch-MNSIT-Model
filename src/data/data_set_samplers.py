from torch.utils.data import Subset, RandomSampler
from ..config import DEFAULT_TRAINING_SET_SIZE, DEFAULT_VALIDATION_SET_SIZE, DEFAULT_TEST_SET_SIZE


class DataSetSamplers:

    def __init__(
            self,
            data_set,
            training_set_size=DEFAULT_TRAINING_SET_SIZE,
            validation_set_size=DEFAULT_VALIDATION_SET_SIZE,
            test_set_size=DEFAULT_TEST_SET_SIZE,
            sampler_cls=RandomSampler,
            **kwargs
    ):
        self.data_set = data_set

        self.data_set_size = len(data_set)
        self.training_set_size = training_set_size * self.data_set_size
        self.validation_set_size = validation_set_size * self.data_set_size
        self.test_set_size = test_set_size * self.data_set_size

        self.sampler_cls = sampler_cls
        self.sampler_kwargs = kwargs
        self._training_sampler = None
        self._validation_sampler = None
        self._test_sampler = None

    @property
    def training_sampler(self):
        if not self._training_sampler:
            sample_start = 0
            sample_end = int(self.training_set_size - 1)
            subset = Subset(self.data_set, range(sample_start, sample_end + 1))
            self._training_sampler = self.sampler_cls(subset, **self.sampler_kwargs)
        return self._training_sampler

    @property
    def validation_sampler(self):
        if not self._validation_sampler:
            sample_start = int(self.training_set_size)
            sample_end = int(self.training_set_size + self.validation_set_size - 1)
            subset = Subset(self.data_set, range(sample_start, sample_end + 1))
            self._validation_sampler = self.sampler_cls(subset, **self.sampler_kwargs)
        return self._validation_sampler

    @property
    def test_sampler(self):
        if not self._test_sampler:
            sample_start = int(self.training_set_size + self.validation_set_size)
            sample_end = int(self.data_set_size - 1)
            subset = Subset(self.data_set, range(sample_start, sample_end + 1))
            self._test_sampler = self.sampler_cls(subset, **self.sampler_kwargs)
        return self._test_sampler
