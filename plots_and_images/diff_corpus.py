import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

sns.set(style="dark")

font_size=14
font = {'size' : font_size}
matplotlib.rc('font', **font)

labels = ['XNLI', 'NER', 'POS', 'XQuAD']

# Data
# For XQuAD, the \Delta SUP is positive for parallel.
# Hence, I've made the 0 and adjusted the other XQuAD scores accordingly.
# parallel = [1.0, 1.9, 0.4, -1.3 + 1.3]
# non_same = [1.8, 4.9, 0.7, 0.6 + 1.3]
# non_diff = [5.9, 22.0, 4.2, 2.9 + 1.3]

parallel = np.array([1, 2, 0, 0])
non_same = np.array([2, 5, 1, 2])
non_diff = np.array([6, 22, 4, 4])

# Label location
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# The numbers that need to be plotted
fig, ax = plt.subplots()
offset = -2
rects1 = ax.bar(x - 1.5 * width, parallel - offset, width, label='Parallel', bottom=offset)
rects2 = ax.bar(x - 0.5 * width, non_same - offset, width, label='Non-parallel (same)', bottom=offset)
rects3 = ax.bar(x + 0.5 * width, non_diff - offset, width, label='Non-parallel (diff)', bottom=offset)
# rects4 = ax.bar(x + 1.5 * width, xquad, width, label='XQuAD')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r'$\mathbf{|\Delta_{(BZ-BS)}|}$', fontsize=font_size+2, labelpad=10)
# ax.set_xlabel('BZ for different tasks')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=font_size)
ax.legend(loc='upper left', fontsize=font_size)

ax.set_ylim(top=39, bottom=-4)
plt.yticks(fontsize=font_size)

# Add numbers on top of the bars
ax.bar_label(rects1)
ax.bar_label(rects2)
ax.bar_label(rects3)
# ax.bar_label(rects4)

fig.tight_layout()

# Default DPI is 100
# plt.savefig('images/diff_corpus.png', dpi=100)
plt.savefig('images/diff_corpus.pdf', dpi=100)