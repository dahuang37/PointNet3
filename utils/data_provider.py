import os
import sys



class DataProvider(object):

    def __init__(self, items_to_tensors):
        self._items_to_tensors = items_to_tensors

    def get(self, items):

        return [self._items_to_tensors[item] for item in items]

