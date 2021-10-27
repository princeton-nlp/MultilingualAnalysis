import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

sns.set(style="dark")

font_size=22
font = {'size' : font_size}
matplotlib.rc('font', **font)
offset = -2

labels = ['XNLI', 'NER', 'POS']

# Data
trans = np.rint(-np.array([-1.7, -2.1, -0.5]))
inv = np.rint(-np.array([-3.4, -44.5, -0.7]))
trans_inv = np.rint(-np.array([-19.2, -51.5, -42.7]))

# Label location
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# The numbers that need to be plotted
fig, ax = plt.subplots(figsize=(7, 8))
# rects1 = ax.bar(x - 1.5 * width, trans - offset, width, label='Transliteration', bottom=offset)
# rects2 = ax.bar(x - 0.5 * width, inv - offset, width, label='Inversion', bottom=offset)
# rects3 = ax.bar(x + 0.5 * width, trans_inv - offset, width, label='Trans' + r'$\circ$' + 'Inv', bottom=offset)
rects1 = ax.bar(x - 1.5 * width, trans - offset, width, label=r'$\mathbf{\mathcal{T}}_{trans}}$', bottom=offset)
rects2 = ax.bar(x - 0.5 * width, inv - offset, width, label=r'$\mathbf{\mathcal{T}}_{inv}}$', bottom=offset)
rects3 = ax.bar(x + 0.5 * width, trans_inv - offset, width, label=r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{inv}}$', bottom=offset)
# rects4 = ax.bar(x + 1.5 * width, xquad, width, label='XQuAD')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r'$\mathbf{|\Delta_{(BZ-BS)}|}$', fontsize=font_size+2, labelpad=14)
# ax.set_xlabel('BZ for different tasks')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=font_size)
ax.legend(fontsize=font_size-2, loc='upper left')

ax.set_ylim(top=70, bottom=-5)
plt.yticks(fontsize=font_size)

# Add numbers on top of the bars
ax.bar_label(rects1)
ax.bar_label(rects2)
ax.bar_label(rects3)
# ax.bar_label(rects4)

fig.tight_layout()

# Default DPI is 100
# plt.savefig('../images/composed_inv.png', dpi=100)
plt.savefig('../images/composed_inv.pdf', dpi=100)