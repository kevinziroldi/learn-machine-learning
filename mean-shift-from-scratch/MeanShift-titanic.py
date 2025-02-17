import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('titanic.xls')
original_df = df.copy()

df.drop(['body','name'], 1, inplace=True)

pd.to_numeric(df.columns.values, errors='ignore')
df.fillna(0, inplace=True)

def handle_non_numeric_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {} # sarà tipo {'female':0}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
            
    return df

df = handle_non_numeric_data(df)

df.drop(['boat', 'sex'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df.cluster_group.iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temporary_df = original_df[(original_df['cluster_group']==float(i))] # new temporary df where cluster_group is cluster_group 0
    survival_cluster = temporary_df[(temporary_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temporary_df)
    survival_rates[i] = survival_rate

print(survival_rates)