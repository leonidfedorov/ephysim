import numpy as np
import matplotlib.pyplot as plt


#euler maruyama integration setup
dt = .001  # Time step.
T = 10.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

sqrtdt = np.sqrt(dt)

#ref: Hodgkin and Huxley 1952
c_m  =   1.0
g_K  =  36.0
g_Na = 120.0 
g_l  =   0.3
v_K  = - 12.0
v_Na =  115.0 
v_l  = 10.613

#nonlinearities
alpha_n = lambda V: 0.01 * (10.0 - V) / (np.exp((10.0 - V) / 10.0) - 1.0)
beta_n = lambda V: 0.125 * np.exp( (0.0 - V) / 80.0)
alpha_m = lambda V: 0.1 * (25.0 - V) / (np.exp((25.0 - V) / 10.0) - 1.0)
beta_m = lambda V: 4.0 * np.exp( (0.0 - V) / 18.0)
alpha_h = lambda V: 0.07 * np.exp((0.0 - V) / 20.0)
beta_h = lambda V: 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)

#additive current terms
I_K = lambda V, n: g_K  * n**4 * (v_K - V)
I_Na = lambda V, m, h: g_Na * m**3 * h * (v_Na - V)
I_L = lambda V: g_l * (v_l - V)

mu = 4

#initial values
V = np.zeros(n)
M = np.zeros(n)
H = np.zeros(n)
N = np.zeros(n)

V[0] = 0.0
N[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
M[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
H[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))


for i in range(n - 1):
    V[i + 1] = V[i] + dt * (g_K * N[i] ** 4 * (v_K - V[i]) + g_Na * M[i] ** 3 * H[i] * (v_Na - V[i]) + g_l * (v_l - V[i]) + mu) / c_m #+ sqrtdt * np.random.randn()
    N[i + 1] = alpha_n(V[i]) * (1.0 - N[i]) - beta_n(V[i]) * N[i]
    M[i + 1] = alpha_m(V[i]) * (1.0 - M[i]) - beta_m(V[i]) * M[i]
    H[i + 1] = alpha_h(V[i]) * (1.0 - H[i]) - beta_h(V[i]) * H[i]
    print (i)

print(range(n-1))
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, V, lw=2)
plt.show()

