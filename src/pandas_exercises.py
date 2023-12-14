import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from copy import copy
from dataclasses import dataclass
from typing import Any
from pprint import pformat

@dataclass
class BasicExercise:
    data: Any
    result: Any
    ideal: None
        
    def __repr__(self):
        return f"""
Data
----------------------------------------
{pformat(self.data)}

Result
----------------------------------------
{pformat(self.result)}
"""

patients = pd.read_csv('data/patient-records.csv', 
                       skiprows=1,
                       names=['Patient', 'Visit_Date', 'Weight', 'Height'],
                       parse_dates=['Visit_Date'],
                       dtype={'Height': np.float16, 'Weight': np.float16},
                       index_col='Visit_Date')

verl = pd.read_csv('data/verlegenhuken.csv', 
                    parse_dates=['DATE'], index_col='DATE')

def make_artists():
    artists = pd.read_csv('data/artists.csv', index_col='id')
    genres = artists.genre.str.split(',', expand=True)

    artists['genre1'] = genres[0]
    artists['genre2'] = genres[1]
    artists['genre3'] = genres[2]

    artists['years'] = artists.years.str.replace('–', '-')
    dates = artists.years.str.split(' - ', expand=True)
    pd.to_datetime(dates[1], format="%Y", errors='ignore')

    artists['start'] = pd.to_datetime(dates[0], format="%Y", errors='ignore')
    artists['end'] = pd.to_datetime(dates[1], format="%Y", errors='ignore')
    del artists['genre']
    del artists['years']
    del artists['bio']
    del artists['wikipedia']
    return artists

artists = make_artists()

# Exercise 3.1
data = patients.to_dict(orient='index')
result = pd.DataFrame.from_dict(data, orient='index')
ex3_1 = BasicExercise(data=data, result=result, ideal=patients)

# Exercise 3.2
data = patients.to_dict(orient='records')
result = pd.DataFrame.from_records(data)
ex3_2 = BasicExercise(data=data, result=result, ideal=patients)

# Exercise 3.3
data = patients.to_dict(orient='dict')
result = pd.DataFrame.from_dict(data, orient='columns')
ex3_3 = BasicExercise(data=data, result=result, ideal=patients)

# Exercise 3.4
result = patients.reset_index().set_index(patients.Patient.str.upper())
ex3_4 = BasicExercise(data=patients, result=result, ideal=None)

# Exercise 3.5
data = np.random.randint(10, 100, 16).reshape(4, 4)
result = pd.DataFrame(data, 
                      index=['Alice', 'Bob', 'Carlos', 'Dan'],
                      columns=['key1', 'key2', 'key3', 'key4'])
ex3_5 = BasicExercise(data=data, result=result, ideal=None)

# Exercise 3.6
data = pd.Series(np.random.randint(10,100, 6))
result = pd.DataFrame({'Int16': data.astype(np.int16), 
                       'Float32': data.astype(np.float32),
                       'Complex': data.astype(complex),
                       'String': data.astype(str)}).set_index(data)
ex3_6 = BasicExercise(data=data, result=result, ideal=None)


# Exercise 7.1
result = (verl
    .sort_index()
    .DEWP
    .asfreq('d')
    .rolling(60)
    .mean()
    .loc[::10]
)
ex7_1 = BasicExercise(data=verl, result=result, ideal=None)

# Exercise 7.2
result = verl.groupby(verl.index.month).SLP.agg(['first', 'last'])
result = result['last'] - result['first']
result.index = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
ex7_2 = BasicExercise(data=verl, result=result, ideal=None)

# Exercise 8.1
most_by_nationality = (
    artists.groupby('nationality')
    .max(numeric_only=True)
    .reset_index()
)
result = (most_by_nationality
     .set_index(["nationality", "paintings"])
     .join(artists.set_index(["nationality", "paintings"]))
).loc[:, ["name"]]
ex8_1 = BasicExercise(data=artists, result=result, ideal=None)

# Exercise 8.2
result = (artists
    .set_index(['end', 'start', 'nationality'])
    .sort_index()
    .reset_index()
    .set_index(['end', 'nationality'])
    .loc[:, ['name', 'start']]
)
ex8_2 = BasicExercise(data=artists, result=result, ideal=None)

# Exercise 8.3
result = (artists
    .loc[:, ['name', 'paintings']]
    .drop_duplicates()
    .set_index('name')
    .paintings
)
ex8_3 = BasicExercise(data=artists, result=result, ideal=None)

# Exercise 8.4
dfs = []
for genre in ['genre1', 'genre2', 'genre3']:
    dfs.append(artists.pivot_table(index='name', columns=genre, aggfunc='first'))
result = (pd.concat(dfs, axis=0)
    .stack()
    .reset_index()
    .set_index('name')
    .level_1
    .sort_index()
)
ex8_4 = BasicExercise(data=artists, result=result, ideal=None)

# Cleanup names we don't want to export
del patients
del verl
del data
del dfs
del result
