import math
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import lognorm, norm
import numpy as np

path = os.path.join("podatki", "Hobotnice.csv")
data = pd.read_csv(path, names=["X"])

n = data.size

Y = [math.log(x) for _, x in data.X.items()]
data["Y"] = Y

Z = [math.log(x, 10) for _, x in data.X.items()]
data["Z"] = Z


# naloga a)

# Izračunajmo dolžino intervala po modificiranem Freedman-Diaconisovem pravilu
quart1 = data.X.quantile(0.25)
quart3 = data.X.quantile(0.75)
iqr = quart3 - quart1

width = 2.6 * iqr / (n ** (1/3))

print("N = ", n, ", širina po modificiranem Freedman-Diaconisovem pravilu je", width)

alpha = data.X.min()
beta = data.X.max()
binNumber = math.ceil((beta - alpha) / width)
shift = (width * binNumber - (beta - alpha)) / 2
start = alpha - shift

start = 0
bins = [float(start + i * width) for i in range(0, binNumber + 1)]

print(alpha)
print(beta)
print(binNumber)
print(shift)
print(start)
print(bins)

fig1, ax1 = plt.subplots()
ax1.set_title("Histogram z log-normalno gostoto")
ax1.hist(data.X, bins=bins)

# Če so podatki iz datoteke hobotnice porazdeljeni log-normalno, je Y porazdeljen normalo
# Izračunajmo pričakovano vrednost in varianco Y
mu1 = data.Y.mean()
sigma1 = data.Y.std(ddof=0)
print("mi = ", mu1)
print("sigma = ", sigma1)

x = np.linspace(0, 200, 1000)
area = width * n
ax1.plot(x, area * lognorm.pdf(x, s=sigma1, scale=math.exp(mu1)))

fig1.show()






# naloga b)

# Izračunajmo dolžino intervala po modificiranem Freedman-Diaconisovem pravilu
quart1 = data.Z.quantile(0.25)
quart3 = data.Z.quantile(0.75)
iqr = quart3 - quart1

width = 2.6 * iqr / (n ** (1/3))

print("Širina po modificiranem Freedman-Diaconisovem pravilu je", width)

alpha = data.Z.min()
beta = data.Z.max()
binNumber = math.ceil((beta - alpha) / width)
shift = (width * binNumber - (beta - alpha)) / 2
start = alpha - shift

bins = [float(math.pow(10, start + i * width)) for i in range(0, binNumber + 1)]

print(alpha)
print(beta)
print(binNumber)
print(shift)
print(start)
print(bins)

fig2, ax2 = plt.subplots()
ax2.set_title("Histogram z normalno gostoto")
ax2.hist(data.X, bins=bins)
plt.xscale("log")

mu2 = data.Z.mean()
sigma2 = data.Z.std(ddof=0)

x = np.linspace(alpha, beta, 1000)
area = width * n
print(area)
ax2.plot(np.power(10, x), area * norm.pdf(x, loc=mu2, scale=sigma2))

fig2.show()


# naloga c)

fig11, ax11 = plt.subplots()
#plt.xscale('log')
stats.probplot(data.X, plot=plt)
#fig11.show()

def qqplot(data, cdf):
    n = len(data)
    data.sort()
    X = [(k + 1) / (n + 1) for k in range(n)]
    Y = [cdf(x) for x in data]
    return (X, Y)

def qqplot2(data, ppf):
    n = len(data)
    data.sort()
    X = [ppf((k + 1) / (n + 1)) for k in range(n)]
    Y = data
    return (X, Y)

fig3, ax3 = plt.subplots()
ax3.set_title("Primerjalni kvantilni grafikon")
#X, Y = qqplot(data.X.tolist(), lambda x : lognorm.cdf(x, s=sigma1, scale=math.exp(mu1)))

mu0 = data.X.mean()
sigma0 = data.X.std(ddof=0)
#X, Y = qqplot(data.X.tolist(), lambda x : norm.cdf(x, loc=mu0, scale=sigma0))
#X, Y = qqplot2(data.X.tolist(), lambda x : norm.ppf(x, loc=0, scale=1))

X, Y = qqplot2(data.X.tolist(), lambda x : lognorm.ppf(x, s=sigma1, scale=math.exp(mu1)))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log-normalna porazdelitev")
plt.ylabel("porazdelitev dolžin hobotnic")
ax3.scatter(X, Y)

fig3.show()