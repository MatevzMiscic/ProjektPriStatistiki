import math
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f
import numpy as np

path = os.path.join("podatki", "Piscanci.csv")
data = pd.read_csv(path)


# test, če b) točka deluje ravilno (zgleda, da ja)
#"""
import random
random.seed(1)

data = data[data.DIETA == 1]
rows = []
for i in range(4):
    for _, row in data.iterrows():
        if random.randint(0, 100) < 35:
            continue
        newRow = {"TEZA" : row.TEZA, "STAROST" : row.STAROST, "OSEBEK" : row.OSEBEK, "DIETA" : row.DIETA + i}
        rows.append(newRow)
data = pd.DataFrame(rows)   
#"""

# naloga a)

def LR(X, Y):
    Xt = np.transpose(X)
    return np.matmul(np.linalg.inv(np.matmul(Xt, X)), np.matmul(Xt, Y))
    

# x in y sta seznama enake dolzine, ki dolocata tocke v ravnini
# vrne tupple (a, b), kjer je y = a + b*x premica, ki se najbolje prilega danim tockam
def SLR(x, y):
    X = np.array([[1, val] for val in x])
    Y = np.array([[val] for val in y])
    beta = LR(X, Y)
    return (beta[0][0], beta[1][0])


x = [1, 3, 4, 6]
y = [2, 5, 9, 12]
print(SLR(x, y))


chickenCount = data.OSEBEK.max()
print(chickenCount)

x = [[] for i in range(chickenCount)]
y = [[] for i in range(chickenCount)]
#b = [0 for i in range(chickenCount)]
for _, row in data.iterrows():
    x[row.OSEBEK - 1].append(row.STAROST)
    y[row.OSEBEK - 1].append(row.TEZA)
massgain = 0
for i in range(chickenCount):
    a, b = SLR(x[i], y[i])
    massgain += b
    #print(x[i])
    #print(y[i])
    #print(a, b)
massgain /= chickenCount
print("Piscanec v poprecju pridobi", massgain, "gramov na dan.")

"""
i = 2
fig, ax = plt.subplots()
ax.scatter(x[i], y[i])
a, b = SLR(x[i], y[i])
x2 = 21
ax.plot([0, x2], [a, a + b*x2], color='k')
plt.show()
"""



# naloga b)
n = len(data.index)
dietCount = data.DIETA.max()
p = dietCount + 1
q = 2

# najprej skonstruirajmo matriko X
X = [[0 for j in range(p)] for i in range(n)]
Y = [[0] for i in range(n)]
for i, row in data.iterrows():
    X[i][0] = 1
    X[i][row.DIETA] = row.STAROST
    Y[i] = [row.TEZA]
X = np.array(X)
print(X.shape)
Y = np.array(Y)
print(Y.shape)

#print(LR(X, Y))

Z = [[0, 1] for i in range(p)]
Z[0] = [1, 0]
print(Z)

# vrne ortogonalni projektor na Im(A)
def orthogonalProj(A):
    At = np.transpose(A)
    return np.matmul(np.matmul(A, np.linalg.inv(np.matmul(At, A))), At)

# izracuna kvadrat norme vekrtorja
def norm2(v):
    return np.linalg.norm(v)**2

H = orthogonalProj(X)
K = orthogonalProj(np.matmul(X, Z))

F = (norm2(np.matmul(np.subtract(H, K), Y)) / (p - q)) / (norm2(np.matmul(np.subtract(np.identity(n), H), Y)) / (n - p))
print("Vrednost preizkusne statistike je", F)

for alpha in [0.01, 0.05]:
    c = f.ppf(1 - alpha, dfn=p-q, dfd=n-p)
    print("Stopnja tveganja:", alpha)
    print("Mejna vrednost:", c)
    if(F > c):
        print("Domnevo H0 zavrnemo.")
    else:
        print("Domneve H0 ne zavrnemo.")



