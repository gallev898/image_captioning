import re
import matplotlib.pyplot as plt
import numpy as np

x1, y1 = [], []
test_likelihood = open('likelihood_5_beam_size/test_seq_sum.txt', 'r')
for i, line in enumerate(test_likelihood.readlines()):
    likelihood = re.findall(r'[+-]?\d+\.\d+', line)
    ic = float(likelihood[0])
    lm = float(likelihood[1])
    x1.append(ic)
    y1.append(lm)

print('avg of x test: {}  avg of y test: {}'.format(np.average(x1), np.average(y1)))
plt.plot(x1, y1, 'o', color='black')
plt.show()




plt.clf()
x, y = [], []
custom_likelihood = open('likelihood_5_beam_size/custom_seq_sum.txt', 'r')
for i, line in enumerate(custom_likelihood.readlines()):
    likelihood = re.findall(r'[+-]?\d+\.\d+', line)
    ic = float(likelihood[0])
    lm = float(likelihood[1])
    x.append(ic)
    y.append(lm)

print('avg of x custom: {}  avg of y custom: {}'.format(np.average(x), np.average(y)))
plt.plot(x, y, 'o', color='red')
plt.show()