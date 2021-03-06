import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
matplotlib.rc('font', size=22)
POS = ['VERB', 'NOUN', 'ADJECTIVE']
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

index = np.arange(3)
bar_width = 0.08
opacity = 0.8
plt.clf()

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.bar(index,[0.34, 0.52, 0.32], width=bar_width, color='pink', label='COCO')
plt.bar(index+bar_width, [0.22, 0.19, 0.11], width=bar_width, color='green', label='Random')
plt.bar(index+bar_width*2, [0.27, 0.34, 0.2], width=bar_width, color='blue', label='Unknown objects')
plt.bar(index+bar_width*3, [0.2, 0.18, 0.12], width=bar_width, color='orange', label='Google Cartoon')
plt.bar(index+bar_width*4, [0.26, 0.29, 0.25], width=bar_width, color='red', label='Cropped unknown objects')
plt.bar(index+bar_width*5, [0.25, 0.19, 0.1], width=bar_width, color='yellow', label='JPEG compression')
plt.bar(index+bar_width*6, [0.26, 0.2, 0.11], width=bar_width, color='purple', label='Salt-and-pepper')
plt.bar(index+bar_width*7, [0.27, 0.21, 0.17], width=bar_width, color='brown', label='Snow')
plt.bar(index+bar_width*8, [0.26, 0.27, 0.15], width=bar_width, color='black', label='Cartoon corruption')
#plt.legend()
plt.xticks(index + bar_width+bar_width+bar_width*2, ('VERB', 'NOUN', 'ADJ'))
plt.xlabel("part of speech", fontsize=25)
plt.ylabel("avg prob", fontsize=25)
plt.savefig('{}.png'.format('topp.0.8'))
plt.clf()


index = np.arange(3)
bar_width = 0.08
opacity = 0.8

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.bar(index,[0.0, 0.0, 0.0], width=bar_width, color='pink', label='COCO')
plt.bar(index+bar_width, [0.0, 0.0, 0.0], width=bar_width, color='green', label='Random')
plt.bar(index+bar_width*2, [0.0, 0.0, 0.0], width=bar_width, color='blue', label='Unknown objects')
plt.bar(index+bar_width*3, [0.0, 0.0, 0.0], width=bar_width, color='orange', label='Google cartoon')
plt.bar(index+bar_width*4, [0.0, 0.0, 0.0], width=bar_width, color='red', label='Cropped unknown objects')
plt.bar(index+bar_width*5, [0.0, 0.0, 0.0], width=bar_width, color='yellow', label='JPEG compression')
plt.bar(index+bar_width*6, [0.0, 0.0, 0.0], width=bar_width, color='purple', label='Salt-and-pepper')
plt.bar(index+bar_width*7, [0.0, 0.0, 0.0], width=bar_width, color='brown', label='Snow flakes')
plt.bar(index+bar_width*8, [0.0, 0.0, 0.0], width=bar_width, color='black', label='Cartoon corruption')
plt.legend(fontsize=25)
plt.xticks(index + bar_width+bar_width, ('VERB', 'NOUN', 'ADJ'))
plt.xlabel("part of speech", fontsize=25)
plt.ylabel("avg prob", fontsize=25)
plt.savefig('{}.png'.format('topk10'))
plt.clf()


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.bar(index,[0.52, 0.61, 0.37], width=bar_width, color='pink', label='COCO')
plt.bar(index+bar_width, [0.42, 0.4, 0.06], width=bar_width, color='green', label='Random')
plt.bar(index+bar_width*2, [0.44, 0.49, 0.3], width=bar_width, color='blue', label='Unknown objects')
plt.bar(index+bar_width*3, [0.09, 0.19, 0.31], width=bar_width, color='orange', label='Google cartoon')
plt.bar(index+bar_width*4, [0.45, 0.41, 0.31], width=bar_width, color='red', label='Cropped unknown objects')
plt.bar(index+bar_width*5, [0.32, 0.32, 0.32], width=bar_width, color='yellow', label='JPEG compression')
plt.bar(index+bar_width*6, [0.39, 0.31, 0.3], width=bar_width, color='purple', label='Salt-and-pepper')
plt.bar(index+bar_width*7, [0.39, 0.38, 0.27], width=bar_width, color='brown', label='Snow')
plt.bar(index+bar_width*8, [0.41, 0.34, 0.29], width=bar_width, color='black', label='Cartoon corruption')
#plt.legend()
plt.xticks(index + bar_width+bar_width+bar_width*2, ('VERB', 'NOUN', 'ADJ'))
plt.xlabel("part of speech", fontsize=25)
plt.ylabel("avg prob", fontsize=25)
plt.savefig('{}.png'.format('beam10'))
plt.clf()

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.bar(index,[0.32, 0.51, 0.31], width=bar_width, color='pink', label='COCO')
plt.bar(index+bar_width, [0.4, 0.41, 0.09], width=bar_width, color='green', label='random')
plt.bar(index+bar_width*2, [0.27, 0.39, 0.21], width=bar_width, color='blue', label='unknown obj')
plt.bar(index+bar_width*3, [0.19, 0.14, 0.23], width=bar_width, color='orange', label='cartoon')
plt.bar(index+bar_width*4, [0.26, 0.36, 0.25], width=bar_width, color='red', label='cropped')
plt.bar(index+bar_width*5, [0.22, 0.22, 0.15], width=bar_width, color='yellow', label='JPEG comp')
plt.bar(index+bar_width*6, [0.26, 0.29, 0.25], width=bar_width, color='purple', label='salt-and-pepper')
#plt.legend()
plt.xticks(index + bar_width+bar_width, ('VERB', 'NOUN', 'ADJ'))
plt.xlabel("part of speech", fontsize=25)
plt.ylabel("avg prob", fontsize=25)
plt.savefig('{}.png'.format('greedy'))
plt.clf()