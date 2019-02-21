class ValueInfo(object):
    def __init__(self, value):
        self.id = id(value)
        self.shape = value.shape
        self.dtype = value.dtype
