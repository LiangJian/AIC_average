from read_gnuplot import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import os

f = open("AIC.txt")

n = []
s = []
s2 = []
v = []
e1 = []
e2 = []
for line in f:
    words = line.split()
    if len(words) != 0:
        if words[0] == 'nboot':
            continue
        n.append(int(words[2]))
        s.append(int(words[3]))
        s2.append(int(words[4]))
        v.append(float(words[6]))
        e1.append(float(words[7]))
        e2.append(float(words[8]))

n = np.array(n)
s = np.array(s)
s2 = np.array(s2)
v = np.array(v)
e1 = np.array(e1)
e2 = np.array(e2)

pdf = PdfPages('AIC.pdf')


plt.ylim(.1, .2)
plt.xlabel("# windows")
a = 0
b = 3
plt.errorbar(n[a:b],     v[a:b], yerr=e1[a:b], color='r', fmt='.', label='0-5-2, sys')
plt.errorbar(n[a:b]+0.5, v[a:b], yerr=e2[a:b], color='r', fmt='_', label='0-5-2, sta')
a = 9
b = 12
plt.errorbar(n[a:b]+1.0, v[a:b], yerr=e1[a:b], color='g', fmt='.', label='1-5-0, sys')
plt.errorbar(n[a:b]+1.5, v[a:b], yerr=e2[a:b], color='g', fmt='_', label='1-5-0, sta')
a = 15
b = 19
plt.errorbar(n[a:b]+2.0, v[a:b], yerr=e1[a:b], color='b', fmt='.', label='2-5-2, sys')
plt.errorbar(n[a:b]+2.5, v[a:b], yerr=e2[a:b], color='b', fmt='_', label='2-5-2, sta')
a = 25
b = 30
plt.errorbar(n[a:b]+3.0, v[a:b], yerr=e1[a:b], color='m', fmt='.', label='3-5-0, sys')
plt.errorbar(n[a:b]+3.5, v[a:b], yerr=e2[a:b], color='m', fmt='_', label='3-5-0, sta')
# a = 30
# b = 36
# plt.errorbar(n[a:b]+3.0, v[a:b], yerr=e1[a:b], color='m', fmt='.', label='3-6-0, sys')
# plt.errorbar(n[a:b]+3.5, v[a:b], yerr=e2[a:b], color='m', fmt='_', label='3-6-0, sta')
plt.legend()
pdf.savefig()
plt.close()

# plt.ylim(.1, .2)
# plt.xlabel("# windows")
# a = 9
# b = 12
# plt.errorbar(n[a:b],    v[a:b], yerr=e1[a:b], color='r', fmt='.', label='1-5-0, sys')
# plt.errorbar(n[a:b]+.1, v[a:b], yerr=e2[a:b], color='r', fmt='_', label='1-5-0, sta')
# a = 12                                                            
# b = 15                                                            
# plt.errorbar(n[a:b]+.2, v[a:b], yerr=e1[a:b], color='g', fmt='.', label='1-6-0, sys')
# plt.errorbar(n[a:b]+.3, v[a:b], yerr=e2[a:b], color='g', fmt='_', label='1-6-0, sta')
# plt.legend()
# pdf.savefig()
# plt.close()

plt.ylim(.1, .2)
plt.xlabel("window size")
a = 3
b = 6
plt.errorbar(s[a:b]+.0, v[a:b], yerr=e1[a:b], color='r', fmt='.', label='0-40-2, sys')
plt.errorbar(s[a:b]+.1, v[a:b], yerr=e2[a:b], color='r', fmt='_', label='0-40-2, sta')
a = 50
b = 53
plt.errorbar(s[a:b]+.2, v[a:b], yerr=e1[a:b], color='g', fmt='.', label='1-10-0, sys')
plt.errorbar(s[a:b]+.3, v[a:b], yerr=e2[a:b], color='g', fmt='_', label='1-10-0, sta')
a = 19                                                            
b = 22                                                            
plt.errorbar(s[a:b]+.4, v[a:b], yerr=e1[a:b], color='b', fmt='.', label='2-40-2, sys')
plt.errorbar(s[a:b]+.5, v[a:b], yerr=e2[a:b], color='b', fmt='_', label='2-40-2, sta')
a = 53
b = 56
plt.errorbar(s[a:b]+.6, v[a:b], yerr=e1[a:b], color='m', fmt='.', label='3-40-0, sys')
plt.errorbar(s[a:b]+.7, v[a:b], yerr=e2[a:b], color='m', fmt='_', label='3-40-0, sta')
plt.legend()
pdf.savefig()
plt.close()

plt.ylim(.1, .2)
plt.xlabel("variable range")
a = 6
b = 9
plt.errorbar(s2[a:b],    v[a:b], yerr=e1[a:b], color='r', fmt='.', label='0-40-8, sys')
plt.errorbar(s2[a:b]+.1, v[a:b], yerr=e2[a:b], color='r', fmt='_', label='0-40-8, sta')
a = 22                                                             
b = 25                                                             
plt.errorbar(s2[a:b]+.2, v[a:b], yerr=e1[a:b], color='g', fmt='.', label='2-40-8, sys')
plt.errorbar(s2[a:b]+.3, v[a:b], yerr=e2[a:b], color='g', fmt='_', label='2-40-8, sta')
plt.legend()
pdf.savefig()
plt.close()

plt.ylim(.1, .2)
plt.xlabel("formula combination")
a = 36
b = 40
plt.errorbar(np.array([0.,1.,2.,3.]),     v[a:b], yerr=e1[a:b], color='r', fmt='.', label='2-8-4-40, sys')
plt.errorbar(np.array([0.,1.,2.,3.])+0.1, v[a:b], yerr=e2[a:b], color='r', fmt='_', label='2-8-4-40, sta')
plt.legend()
pdf.savefig()
plt.close()

pdf.close()
