import sys
import pandas as pd
import warnings
from contextlib import contextmanager

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
pd.options.display.width = 80

warnings.simplefilter('ignore')

@contextmanager
def show_all_rows(new_max=sys.maxsize):
    old_max = pd.options.display.max_rows
    try:
        pd.options.display.max_rows = new_max 
        yield old_max
    finally:
        pd.options.display.max_rows = old_max

