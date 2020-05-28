import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline
dt = .001  # Time step.
T = 10.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

re = np.random.rand() 
a = 0.02
b = 0.2
c = -65 + 15 * re ** 2
d = 8 - 6 * re ** 2


sqrtdt = np.sqrt(dt)

v = np.zeros(n)
u = np.zeros(n)

v[0] = -65
u[0] = b*v[0]



for i in range(n - 1):
    I = 5 * np.random.randn()
    #
    v[i + 1] = v[i] + dt * (0.04 * v[i] ** 2 + 5 * v[i] + 140 - u[i] + I) + sqrtdt * np.random.randn()
    u[i + 1] = u[i] + dt * (a * (b * v[i] - u[i]))

#plt.plot(t,x)
#plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, v, lw=2)
plt.show()