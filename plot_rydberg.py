import numpy as np
import matplotlib.pyplot as plt

N = 9

x = np.fromfile(f"data/ryd_x_meas_L_{N}.dat")
z = np.fromfile(f"data/ryd_z_meas_L_{N}.dat")
xzp = np.fromfile(f"data/ryd_xzp_meas_L_{N}.dat")
xzm = np.fromfile(f"data/ryd_xzm_meas_L_{N}.dat")

data = np.zeros(N+1)

data[0] = 2*xzp[0] - x[0] - z[0]
data[N-1] = 2*xzp[N-1] - x[N-1] - z[N-1]
data[N] = x[N]

data[1:N-1] = np.sqrt(2)*(xzp[1:N-1]-xzm[1:N-1]) - x[1:N-1]

plt.figure(figsize=(8,4))
plt.bar(range(1,N+2),data,width=0.5)
plt.xticks(range(1,N+2),[f"$M_{{ {i} }}$" for i in range(1,N+2)],fontsize=25)
plt.yticks([0,0.5,1],fontsize=25)
plt.tight_layout()
plt.savefig("plot/rydberg.pdf",format='pdf')
plt.show()
