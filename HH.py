import numpy as np
import matplotlib.pyplot as plt

#state variables
V = []
M = []
H = []
N = []

dt = 0.05
t = np.linspace(0,10,10000)

#constants
mu = 6.0#7 without noise
c_m = 1.0 #microFarad
v_Na = 50 #miliVolt
v_K = -77  #miliVolt
v_l = -54 #miliVolt
g_Na = 120 #mScm-2
g_K = 36 #mScm-2
g_l = 0.03 #mScm-2

#nonlinearities
alpha_N = lambda v: 0.01 * (v + 50)/(1 - np.exp(- (v + 50) / 10))
beta_N = lambda v: 0.125 * np.exp(- (v + 60) / 80)
alpha_M = lambda v: 0.1 * (v + 35)/(1 - np.exp(- (v + 35) / 10))
beta_M = lambda v: 4.0 * np.exp(- 0.0556 * (v + 60))
alpha_H = lambda v: 0.07 * np.exp(- 0.05 * (v + 60))
beta_H = lambda v: 1 / (1 + np.exp(- (0.1) * (v + 30)))

#intitial conditions:
V.append(-60)
M.append(alpha_M(V[0]) / (alpha_M(V[0]) + beta_M(V[0])))
N.append(alpha_N(V[0]) / (alpha_N(V[0]) + beta_N(V[0])))
H.append(alpha_H(V[0]) / (alpha_H(V[0]) + beta_H(V[0])))

sqrtdt = np.sqrt(dt) #Euler-Maruyama

#electrode = []
#electrode.append(30-V[0])

for i in range(1,len(t)):
    M.append(M[i-1] + dt * ((alpha_M(V[i-1]) * (1 - M[i-1])) - beta_M(V[i-1]) * M[i-1]))
    N.append(N[i-1] + dt * ((alpha_N(V[i-1]) * (1 - N[i-1])) - beta_N(V[i-1]) * N[i-1]))
    H.append(H[i-1] + dt * ((alpha_H(V[i-1]) * (1 - H[i-1])) - beta_H(V[i-1]) * H[i-1]))

    #additive current terms
    I_Na = g_Na * H[i-1] * (M[i-1]) ** 3 * (V[i-1] - v_Na)
    I_K = g_K * N[i-1] ** 4 * (V[i-1] - v_K)
    I_l = g_l * (V[i-1] - v_l)    
    I_noise = sqrtdt * np.random.randn()

    V.append(V[i-1] + (dt) * ((1 / c_m) * (mu - (I_Na + I_K + I_l))) + I_noise)
    #electrode.append((sqrtdt * np.random.randn()+30-V[i-1])/(4*np.pi))


fig, ax = plt.subplots(1, 1, figsize=(8,4))
ax.plot(t, electrode, lw=2)
plt.show()