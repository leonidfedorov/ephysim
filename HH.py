import numpy as np
import matplotlib.pyplot as plt


#euler maruyama integration setup
dt = .001  # Time step.
T = 10.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

sqrtdt = np.sqrt(dt)

#TODO: biophysical model constants requires a reference to the specific neuron type)
C_m  =   1.0 
g_Na = 120.0 
g_K  =  36.0
g_L  =   0.3
E_Na =  50.0 
E_K  = -77.0
E_L  = -54.387

#nonlinearities
alpha_m = lambda V: 0.1*(V + 40.0)/(1.0 - np.exp(-(V + 40.0) / 10.0))
beta_m = lambda V: 4.0*np.exp(-(V + 65.0) / 18.0)
alpha_h = lambda V: 0.07*np.exp(-(V + 65.0) / 20.0)
beta_h = lambda V: 1.0/(1.0 + np.exp(-(V + 35.0) / 10.0))
alpha_n = lambda V: 0.01*(V + 55.0)/(1.0 - np.exp(-(V + 55.0) / 10.0))
beta_n = lambda V: 0.125*np.exp(-(V + 65) / 80.0)
I_Na = lambda V, m, h: g_Na * m**3 * h * (V - E_Na)
I_K = lambda V, n: g_K  * n**4 * (V - E_K)
I_L = lambda V: g_L * (V - E_L)

#stochastic terms
I_input = lambda mu: mu + np.random.randn()

#initial values
V = np.zeros(n)
M = np.zeros(n)
H = np.zeros(n)
N = np.zeros(n)

V[0] = -65
M[0] = 0.05
H[0] = 0.6
N[0] = 0.32



for i in range(n - 1):
    V[i + 1] = V[i] + dt * (0. - I_Na(V[i], M[i], H[i]) - I_K(V[i], N[i]) - I_L(V[i])) / C_m + sqrtdt * np.random.randn()
    M[i + 1] = alpha_m(V[i])*(1.0-M[i]) - beta_m(V[i])*M[i]
    H[i + 1] = alpha_h(V[i])*(1.0-M[i]) - beta_h(V[i])*H[i]
    N[i + 1] = alpha_n(V[i])*(1.0-N[i]) - beta_n(V[i])*N[i]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, V, lw=2)
plt.show()

