import numpy as np
import sys
from sklearn.decomposition import PCA

data = sys.stdin.readlines()[1:]

das = [line.split(",")[0] for line in data]
X = np.genfromtxt(data, delimiter=',')[:, [9,11,12,14,15]]
print(das[1:])
print(X)

pca = PCA(n_components=2)
X_ = pca.fit_transform(X)

for da,x in zip(das,X_):
    print("{},{},{}".format(da,x[0],x[1]))
