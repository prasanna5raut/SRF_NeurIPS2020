import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_style("white", {'axes.grid': False})
numPoints = 3

plt.figure(dpi=1200)
# L 1
plt.plot(1/2*np.ones(numPoints), np.linspace(1/2, 1.01, numPoints), color='k')
plt.plot(np.linspace(0.5, 1.01, numPoints), 1/2*np.ones(numPoints), color='k')
# L 2
plt.plot(2/3*np.ones(numPoints), np.linspace(2/3, 1.01, numPoints), color='k')
plt.plot(np.linspace(2/3, 1.01, numPoints), 2/3*np.ones(numPoints), color='k')
# L 3
plt.plot(3/4*np.ones(numPoints), np.linspace(3/4, 1.01, numPoints), color='k')
plt.plot(np.linspace(3/4, 1.01, numPoints), 3/4*np.ones(numPoints), color='k')
# L 4
plt.scatter(1, 1, color='k', marker='o')

plt.plot(1/2*np.ones(numPoints), np.linspace(0, 1/2, numPoints), color='red')
plt.plot(np.linspace(0, 1/2, numPoints), 1/2*np.ones(numPoints), color='red')

plt.plot(1/2*np.ones(numPoints), np.linspace(0, 1/2, numPoints), color='#c5c5c5', linestyle='--')
plt.plot(np.linspace(0, 1/2, numPoints), 1/2*np.ones(numPoints), color='#c5c5c5', linestyle='--')

plt.plot(2/3*np.ones(numPoints), np.linspace(0, 2/3, numPoints), color='#c5c5c5', linestyle='--')
plt.plot(np.linspace(0, 2/3, numPoints), 2/3*np.ones(numPoints), color='#c5c5c5', linestyle='--')

plt.plot(3/4*np.ones(numPoints), np.linspace(0, 3/4, numPoints), color='#c5c5c5', linestyle='--')
plt.plot(np.linspace(0, 3/4, numPoints), 3/4*np.ones(numPoints), color='#c5c5c5', linestyle='--')

plt.fill_between(np.linspace(1/2, 1.01, numPoints), 0, 0.5*np.ones(numPoints), facecolor='red', alpha=0.75)
plt.fill_between(np.linspace(0, 0.5, numPoints), 0, 0.5*np.ones(numPoints), facecolor='red', alpha=0.75)
plt.fill_between(np.linspace(0, 0.5, numPoints), 0.5, 0.01+np.ones(numPoints), facecolor='red', alpha=0.75)
plt.xlabel('$C_T$')
plt.ylabel('$R_T$')
plt.xlim([0, 1.01])
plt.ylim([0, 1.01])
ax = plt.axes()
ax.set_aspect('equal')
ax.annotate('$W=1$', xy=(1, 0.5), xytext=(0.8, 0.51))
ax.annotate('$W=T^{1/3}$', xy=(1, 0.5), xytext=(0.8, 2/3+0.01))
ax.annotate('$W=T^{1/2}$', xy=(1, 0.5), xytext=(0.8, 3/4+0.01))
ax.annotate('$W=T$', xy=(1, 0.5), xytext=(0.85, 0.95))
plt.xticks([0, 0.5, 2/3, 3/4, 1], ['0', '$T^{1/2}$', '$T^{2/3}$', '$T^{3/4}$', '$T$'])
plt.yticks([0, 0.5, 2/3, 3/4, 1], ['0', '$T^{1/2}$', '$T^{2/3}$', '$T^{3/4}$', '$T$'])