
class ClassMapper:

    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, item):
        x, y = item['x'], item['y']
        if not y in self.class_mapping:
            raise ValueError('Class {} does not exist in mapping'.format(y))
        return x, self.class_mapping[y]
