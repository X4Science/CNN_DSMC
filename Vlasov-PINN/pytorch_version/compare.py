import matplotlib.pyplot as plt
import numpy as np

train_loss = np.genfromtxt('./result/train_loss.txt')
lbfgs_loss = np.genfromtxt('./result/lbfgs_loss.txt')
loss = np.concatenate((train_loss, lbfgs_loss))
plt.plot(np.log10(loss))
plt.savefig('loss.png')
plt.show()