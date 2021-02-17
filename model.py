
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

combine = pd.read_csv('C:/Users/archi/Documents/combine-proj/combine_data_since_2000_PROCESSED_2018-04-26.csv')
stats = pd.read_csv('C:/Users/archi/Documents/combine-proj/Game_Logs_Runningback.csv')
stats.Name = stats.Name.str.split(', ').map(lambda x : ' '.join(x[::-1]))
#print(combine)
stats = stats.loc[stats['Season']=='Regular Season']
stats = stats.drop('Year',axis=1)
combine = combine.loc[combine['Pos']=='RB']
combine = combine.merge(stats,on='Name',how='left')
#print(combine)
combine = combine.drop(['Pick','Round','Week'],axis=1)
#print(combine)
#print(combine.columns.values.tolist())
#combine = combine.groupby(['Name','Ht','Wt','Forty','Vertical','BenchReps','BroadJump','Cone','Shuttle','Year','AV','Pos'])[["Games Played","Receiving TDs"]].sum()
combine = combine.groupby(['Name','Ht','Wt','Forty','Vertical','BenchReps','BroadJump','Cone','Shuttle','Year','AV','Pos']).sum().reset_index()
#combine = combine.sort_values(by=['AV'], ascending=False)
#print(combine)

#combine.set_axis(combine["Name"],axis="columns")
X = combine.iloc[:, 1:9].values
y = combine.iloc[:, -3].values


"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

from sklearn.metrics import r2_score
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df)
print(r2_score(y_test,y_pred))

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
regressor2 = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)


y_pred = regressor2.predict(X_test)
df2 = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df2)
print(regressor2.score(X_test,y_test))
