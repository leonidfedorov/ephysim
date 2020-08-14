import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.representations
from scipy.signal import find_peaks

dt = 0.05
t = np.linspace(0, 10, 10000)

#constants
mu = 4.5#7#put 7 when without noise
c_m = 1.0 #microFarad
v_Na = 50 #miliVolt
v_K = -77  #miliVolt
v_l = -54 #miliVolt
g_Na = 120 #mScm-2
g_K = 36 #mScm-2
g_l = 0.03 #mScm-2

#nonlinearities
alpha_N = lambda v: 0.01 * (v + 50) / (1 - np.exp(- (v + 50) / 10))
beta_N = lambda v: 0.125 * np.exp(- (v + 60) / 80)
alpha_M = lambda v: 0.1 * (v + 35) / (1 - np.exp(- (v + 35) / 10))
beta_M = lambda v: 4.0 * np.exp(- 0.0556 * (v + 60))
alpha_H = lambda v: 0.07 * np.exp(- 0.05 * (v + 60))
beta_H = lambda v: 1 / (1 + np.exp(- (0.1) * (v + 30)))


sqrtdt = np.sqrt(dt) #Euler-Maruyama

neurons = np.empty([2, len(t)])
#temporary = []
#run multiple independent compartments
for k in range(0, 2):
    #intitial conditions:
    #state variables
    V = []
    M = []
    H = []
    N = []
#    temporary = []
 #   temporary.append(mu)
    V.append(-60)
    M.append(alpha_M(V[0]) / (alpha_M(V[0]) + beta_M(V[0])))
    N.append(alpha_N(V[0]) / (alpha_N(V[0]) + beta_N(V[0])))
    H.append(alpha_H(V[0]) / (alpha_H(V[0]) + beta_H(V[0])))

    for i in range(1, len(t)):
        M.append(M[i-1] + dt * ((alpha_M(V[i-1]) * (1 - M[i-1])) - beta_M(V[i-1]) * M[i-1]))
        N.append(N[i-1] + dt * ((alpha_N(V[i-1]) * (1 - N[i-1])) - beta_N(V[i-1]) * N[i-1]))
        H.append(H[i-1] + dt * ((alpha_H(V[i-1]) * (1 - H[i-1])) - beta_H(V[i-1]) * H[i-1]))

        #additive current terms
        I_Na = g_Na * H[i-1] * M[i-1] ** 3 * (V[i-1] - v_Na)
        I_K = g_K * N[i-1] ** 4 * (V[i-1] - v_K)
        I_l = g_l * (V[i-1] - v_l)    
        I_noise = sqrtdt * np.random.randn()
        #temporary.append(((1 / c_m) * (mu - (I_Na + I_K + I_l))))
        V.append(V[i-1] + (dt) * ((1 / c_m) * (mu - (I_Na + I_K + I_l))) + I_noise)
    
    neurons[k-1, :] = V

print(np.shape(neurons))
print(max(neurons[0,:]))
print(max(neurons[1,:]))

timesA = (neurons[0, :] > 41) * neurons[0, :]
timesB = (neurons[1, :] > 41) * neurons[1, :]

#timesA2 = np.argwhere(neurons[0, :] > 41)
#timesB2 = np.argwhere(neurons[1, :] > 41)

#windowsA = np.empty([len(timesA2), 70])
#for j in range(0, len(timesA2)):
#    print(timesA2[j][0])
#    leftEnd =  timesA2[j][0] - 35
#    rightEnd = timesA2[j][0] + 35
#    print(np.shape(neurons[0, leftEnd:rightEnd]))# : timesA2[j][0]+35)
#    windowsA[j, :] = neurons[0, leftEnd:rightEnd]

#windowsB = np.empty([len(timesB2), 70])
#for j in range(0, len(timesB2)):
#    print(timesA2[j][0])
#    leftEnd =  timesB2[j][0] - 35
#    rightEnd = timesB2[j][0] + 35
#    print(np.shape(neurons[0, leftEnd:rightEnd]))# : timesA2[j][0]+35)
#    windowsB[j, :] = neurons[1, leftEnd:rightEnd]
#print(np.shape(windowsA))
#print(len(timesA2))
#print(np.shape(windowsB))
#print(len(timesB2))

peaksA, _ = find_peaks(neurons[0, :], prominence=40)
#print("peaksA")
#print(len(peaksA))
#print(peaksA)
peaksB, _ = find_peaks(neurons[1, :], prominence=40)
#print("peaksB")
#print(len(peaksB))
#print(peaksB)

windowsA = np.empty([len(peaksA), 300])
for j in range(0, len(peaksA)):
    leftEnd =  peaksA[j] - 20
    rightEnd = peaksA[j] + 280
    windowsA[j, :] = neurons[0, leftEnd:rightEnd]

windowsB = np.empty([len(peaksB), 300])
for j in range(0, len(peaksB)):
    leftEnd =  peaksB[j] - 20
    rightEnd = peaksB[j] + 280
    windowsB[j, :] = neurons[1, leftEnd:rightEnd]
    
print(np.shape(windowsA))
print(len(peaksA))
print(np.shape(windowsB))
print(len(peaksB))






tW = np.linspace(0, 10, 300)
plt.figure(figsize=(9, 3))
#plt.plot(tW, windowsA[0,:], 'r-',label='voltage')
plt.plot(tW[::10], windowsA[0,::10], 'r-',label='voltage')

#figN,figM = np.shape(windowsA)
#for k in range(0,figN*figM):
#    plt.subplot(figN*figM, 1, k + 1)
#    plt.plot(tW[::10], windowsA[0,::10], 'r-',label='voltage')



plt.figure(figsize=(9, 3))
plt.subplot(4, 1, 1)
plt.plot(t, timesA, 'g-',label='voltage')
plt.subplot(4, 1, 2)
plt.plot(t, neurons[0, :], 'b-',label='voltage')
plt.subplot(4, 1, 3)
plt.plot(t, timesB, 'g-',label='voltage')
plt.subplot(4, 1, 4)
plt.plot(t, neurons[1, :], 'b-',label='voltage')

electrode = np.empty([4, len(t)])
electrode[0, :] = -1.0 * neurons[0, :] + (-1.0) * neurons[1, :] / 2.0
electrode[1, :] = (-1.0/np.sqrt(2)) * neurons[0, :] + (-1.0/np.sqrt(2)) * neurons[1, :] / 2.0
electrode[2, :] = -0.5 * neurons[0, :] + -0.5 * neurons[1, :] / 2.0
electrode[3, :] = (-1.0/np.sqrt(5)) * neurons[0, :] + (-1.0/np.sqrt(5)) * neurons[1, :] / 2.0

numwinA, _ = np.shape(windowsA)
numwinB, _ = np.shape(windowsB)

print(numwinA)
fieldA = np.empty([27000, numwinA])
#for j in range(0, numwinA):
#    print(j)
for i in range(0, numwinA):
#    print (i)
#    print(windowsA[i,::10])
    V_x = -1.0 * windowsA[i,::10]
    V_y = (-1.0 / np.sqrt(2)) * windowsA[i, ::10]
    V_z = -0.5 * windowsA[i, ::10]
    V_w = (-1.0 / np.sqrt(5)) * windowsA[i, ::10]
    fieldA[:, i] = 0.5*(-1.0 * np.kron(np.kron(V_x, V_y), V_z) - 1.0 * np.kron(np.kron(V_y, V_z), V_w))
print(np.shape(fieldA))

print(numwinB)
fieldB = np.empty([27000, numwinB])
#for j in range(0, numwinA):
#    print(j)
for i in range(0, numwinB):
#    print (i)
#    print(windowsA[i,::10])
    V_x = -1.0 * windowsB[i,::10] / 2
    V_y = (-1.0 / np.sqrt(2)) * windowsB[i, ::10] / 2
    V_z = -0.5 * windowsB[i, ::10] / 2
    V_w = (-1.0 / np.sqrt(5)) * windowsB[i, ::10] / 2
    fieldB[:, i] = 0.5*(-1.0 * np.kron(np.kron(V_x, V_y), V_z) - 1.0 * np.kron(np.kron(V_y, V_z), V_w))
print(np.shape(fieldB))

L_a = np.empty([numwinA, 5000])
for i in range(0, numwinA):
    cubical_complex = gd.CubicalComplex(
            dimensions=[30, 30, 30],
            top_dimensional_cells=fieldA[:, i],
        )
    diag = cubical_complex.persistence(homology_coeff_field=2, min_persistence=10)
    #gd.plot_persistence_diagram(diag)
    #print(cubical_complex.betti_numbers())

    LS = gd.representations.Landscape(resolution=1000)
    L_a[i, :] = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(2)])
    print(np.shape(L_a))
L_hatA = np.mean(L_a, axis=0)

L_b = np.empty([numwinB, 5000])
for i in range(0, numwinB):
    cubical_complex = gd.CubicalComplex(
            dimensions=[30, 30, 30],
            top_dimensional_cells=fieldB[:, i],
        )
    diag = cubical_complex.persistence(homology_coeff_field=2, min_persistence=10)
    #gd.plot_persistence_diagram(diag)
    #print(cubical_complex.betti_numbers())

    LS = gd.representations.Landscape(resolution=1000)
    L_b[i, :] = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(2)])
    print(np.shape(L_b))
L_hatB = np.mean(L_b, axis=0)




plt.figure(figsize=(9, 3))
plt.subplot(2, 1, 1)
plt.plot(L_hatA[:1000])
plt.plot(L_hatA[1000:2000])
plt.plot(L_hatA[2000:3000])
plt.plot(L_hatA[3000:4000])
plt.plot(L_hatA[4000:5000])
plt.title("Landscape")
plt.subplot(2, 1, 2)
plt.plot(L_hatB[:1000])
plt.plot(L_hatB[1000:2000])
plt.plot(L_hatB[2000:3000])
plt.plot(L_hatB[3000:4000])
plt.plot(L_hatB[4000:5000])
plt.title("Landscape")


plt.figure(figsize=(9, 3))
plt.subplot(4, 1, 1)
plt.plot(t, electrode[0, :], 'g-',label='voltage')
plt.ylim(ymax = 110, ymin = -45)
plt.subplot(4, 1, 2)
plt.plot(t, electrode[1, :], 'g-',label='voltage')
plt.ylim(ymax = 110, ymin = -45)
plt.subplot(4, 1, 3)
plt.plot(t, electrode[2, :], 'g-',label='voltage')
plt.ylim(ymax = 110, ymin = -45)
plt.subplot(4, 1, 4)
plt.plot(t, electrode[3, :], 'g-',label='voltage')
plt.ylim(ymax = 110, ymin = -45)
#plt.plot(t, 0.0 - electrode[0, :] - 0.5*electrode[1, :]+ np.random.normal(-0.1, 0.1, 9).dot(electrode[2:11, :]), 'b-',label='voltage')
#plt.plot(t, 0.0 - electrode[0, :]  + 0.5*abs(np.random.normal(0, 0.1, 100)).dot(electrode[1:101, :]), 'b-',label='voltage')

plt.show()