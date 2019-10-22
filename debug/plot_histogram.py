import matplotlib.pyplot as plt
import torch
import os

myDictionary = torch.load('pos_dic_random')['pos']

lll = [sum(e)/ len(e) for e in myDictionary.values()]
plt.bar(myDictionary.keys(), lll, 1, color='g')
plt.show()