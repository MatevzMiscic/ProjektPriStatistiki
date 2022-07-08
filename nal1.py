import random
import math
import pandas as pd
import os.path

random.seed(10)

n = 200

path = os.path.join("podatki", "Kibergrad.csv")
data = pd.read_csv(path)

# naloga a)
children = data["'OTROK'"]
sample = children.sample(n, random_state=random.randint(0, 10**6))

avg = sample.mean()
print("Povprečje na vzorcu je", avg, ".")

# naloga b)
def StandardError(n, N, avg, samp):
    SE2 = 0
    for val in samp.items():
        SE2 += (val[1] - avg) ** 2
    SE2 *= (N - n) / (N * n * (n - 1))
    return math.sqrt(SE2)

def ConfidenceInterval(avg, SE):
    return (avg - 1.96*SE, avg + 1.96*SE)

N = children.size
SE = StandardError(n, N, avg, sample)
CI = ConfidenceInterval(avg, SE)
print("Ocena standardne napake je ", SE, "interval zaupanja pa je ", CI, ".")

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

count = 0
for i in range(1, 100):
    sample = children.sample(n, random_state=random.randint(0, 10**6))
    avg = sample.mean()
    avgs[i] = avg
    SE = StandardError(n, N, avg, sample)
    CIs[i] = ConfidenceInterval(avg, SE)
    if CIs[i][0] < popavg and popavg < CIs[i][1]:
        count += 1

print("Od 100 intervalov zaupanja jih ", count, "vsebje populacijsko povprečje.")

# naloga e)
avg = 0
for val in avgs:
    avg += val
avg /= 100
SE = 0
for val in avgs:
    SE += (val - avg) ** 2
SE /= 100
SE = math.sqrt(SE)







