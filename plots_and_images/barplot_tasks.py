import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="dark")

# labels = ['Inversion', 'Permutation', 'Syntax', 'Transliteration']
labels = ['Inversion', 'Permutation', 'Syntax']

# Data
xnli = [10.2, 3.6, 0.9]
ner = [49.1, 26.3, 14.6]
pos = [30.2, 11.2, 4.4]
xquad = [32.8, 0, 0]

# Label location
x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

# The numbers that need to be plotted
fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, xnli, width, label='XNLI')
rects2 = ax.bar(x - 0.5 * width, ner, width, label='NER')
rects3 = ax.bar(x + 0.5 * width, pos, width, label='POS')
rects4 = ax.bar(x + 1.5 * width, xquad, width, label='XQuAD')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
ax.set_xlabel(r'$\mathbf{-\Delta_{\,SUP}}$'+' for different tasks')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.set_ylim(top=110, bottom=-5)

# Add numbers on top of the bars
ax.bar_label(rects1)
ax.bar_label(rects2)
ax.bar_label(rects3)
ax.bar_label(rects4)

fig.tight_layout()

# Default DPI is 100
plt.savefig('images/barplot_tasks.png', dpi=100)