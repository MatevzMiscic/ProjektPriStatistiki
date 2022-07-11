import random
import math
import pandas as pd
import os.path
import matplotlib.pyplot as plt

random.seed(1)

inf = 10**9
n = 200

# Preberemo podatke
path = os.path.join("podatki", "Kibergrad.csv")
data = pd.read_csv(path)

# naloga a)
children = data["'OTROK'"]
sample = children.sample(n, random_state=random.randint(0, inf))

avg = sample.mean()
print("Povprečje na vzorcu je ", avg, ".", sep="")

# naloga b)

# Vrne vrednost nepristranske cenilnke SE^2_+
def StandardError(n, N, avg, samp):
    SE2 = 0
    for val in samp.items():
        SE2 += (val[1] - avg) ** 2
    SE2 *= (N - n) / (N * n * (n - 1))
    return math.sqrt(SE2)

# Vrne približni interval zaupanja
def ConfidenceInterval(avg, SE):
    return (avg - 1.96*SE, avg + 1.96*SE)

N = children.size
aproxSE = StandardError(n, N, avg, sample)
CI = ConfidenceInterval(avg, aproxSE)
print("Ocena standardne napake je ", aproxSE, " interval zaupanja pa je ", CI, ".", sep="")

# naloga c)
popavg = children.mean()

realSE2 = 0
for val in children.items():
    realSE2 += (val[1] - avg) ** 2
realSE2 *= (N - n) / (n * N * (N - 1))
realSE = math.sqrt(realSE2)

print("Povprečje celotne populacije je ", popavg, ", prava standardna napaka pa ", realSE, ".", sep="")
if CI[0] < popavg and popavg < CI[1]:
    print("Interval zaupanja vsebuje populacijsko povprečje.")
else:
    print("Interval zaupanja ne vsebuje populacijskega povprečja.")

# naloga d)
CIs = [None for _ in range(100)]
CIs[0] = CI

avgs = [None for _ in range(100)]
avgs[0] = avg

# poiščimo intervale zaupanja in preštejmo, koliko jih vsebuje populacijsko povprečje.
count = 0
for i in range(1, 100):
    sample = children.sample(n, random_state=random.randint(0, inf))
    avg = sample.mean()
    avgs[i] = avg
    SE = StandardError(n, N, avg, sample)
    CIs[i] = ConfidenceInterval(avg, SE)
    if CIs[i][0] < popavg and popavg < CIs[i][1]:
        count += 1

print("Od 100 intervalov zaupanja jih ", count, " vsebje populacijsko povprečje.", sep="")

# Prikažimo intervale zaupanja na grafikonu.
fig1, ax1 = plt.subplots()
ax1.set_title("Intervali zaupanja")
for i, ci in enumerate(CIs):
    ax1.plot(ci, (i, i), color='blue')
ax1.plot((popavg, popavg), (0, 100), color="orange")
plt.show()

# naloga e)

# Izračunamo standardni odklon povprečij
"""
avg = 0
for val in avgs:
    avg += val
avg /= 100

SE = 0
for val in avgs:
    SE += (val - avg) ** 2
SE /= 100
SE = math.sqrt(SE)
"""
# Najbolje, da uporabimo že vgrajeno funkcijo iz knjižnice pandas
SE = pd.Series(avgs).std(ddof=0)

print("Standardna napaka vzorca je ", aproxSE, ", standardni odklon pa je ", SE, ".", sep="")

# naloga f)

# Zgeneriramo 100 enostavnih naključnih vzorcev velikosti 800 in poiščemo intervale zaupanja
n = 800
count = 0
for i in range(0, 100):
    sample = children.sample(n, random_state=random.randint(0, inf))
    avg = sample.mean()
    avgs[i] = avg
    SE = StandardError(n, N, avg, sample)
    if i == 0:
        aproxSE = SE
    CIs[i] = ConfidenceInterval(avg, SE)
    if CIs[i][0] < popavg and popavg < CIs[i][1]:
        count += 1

print("Od 100 intervalov zaupanja jih tokrat ", count, " vsebje populacijsko povprečje.", sep="")

# Prikažimo intervale zaupanja na grafikonu
fig2, ax2 = plt.subplots()
ax2.set_title("Intervali zaupanja")
for i, ci in enumerate(CIs):
    ax2.plot(ci, (i, i), color='blue')
ax2.plot((popavg, popavg), (0, 100), color="orange")
plt.show()

# Izračunamo standardni odklon povprečij
"""
avg = 0
for val in avgs:
    avg += val
avg /= 100

SE = 0
for val in avgs:
    SE += (val - avg) ** 2
SE /= 100
SE = math.sqrt(SE)
"""
SE = pd.Series(avgs).std(ddof=0)

print("Standardna napaka vzorca je ", aproxSE, ", standardni odklon pa je ", SE, ".", sep="")




