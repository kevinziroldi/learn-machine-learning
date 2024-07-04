import pandas as pd
import numpy as np

df = pd.read_csv('fatal-police-shootings-data.csv')

df.drop(['id'], 1, inplace=True)

races = []
for i in range(len(df.index)):
    if df.loc[i, 'race'] not in races:
        races.append(df.loc[i, 'race'])

races.remove(np.nan)

deathPerRace = {}

for race in races:
    deathPerRace[race] = 0

for i in range(len(df)):
    for race in races:
        if df.loc[i, 'race'] == race:
            deathPerRace[race] += 1
            break

print(deathPerRace)

print(df.groupby('city').armed.count().sort_values(ascending=False))