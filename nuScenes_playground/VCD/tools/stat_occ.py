import numpy as np 
import matplotlib.pyplot as plt

stat_occ = np.loadtxt('data/test_data/stat_occ.txt')
stat_occ_total = stat_occ.mean(0)
stat_occ_total = np.log(stat_occ_total)/np.log(10) #1 / (1 + np.exp(-stat_occ_total))
plt.figure()
plt.bar(np.arange(101), stat_occ_total)
plt.xlabel('Depth distribution cumulative value')
plt.ylabel('Count/log_10')
plt.savefig('figs/stat_occ.jpg')
