mean = data.mean(axis=0)
data -= mean
std = data.std(axis=0)
data /= std

or 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(data)
