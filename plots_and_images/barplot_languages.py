import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="dark")

labels = [r'$\mathbf{BZ}$', r'$\mathbf{-\Delta_{\,SUP}}$']

# Data
inversion = [64.2, 30.2]
permutation = [73.6, 11.2]
syntax = [89.4, 4.4]
transliteration = [95.4, 0.4]

# Label location
x = np.arange(len(labels))  # the label locations
width = 0.17  # the width of the bars

# The numbers that need to be plotted
fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, inversion, width, label='Inversion')
rects2 = ax.bar(x - 0.5 * width, permutation, width, label='Permutation')
rects3 = ax.bar(x + 0.5 * width, syntax, width, label='Syntax')
rects4 = ax.bar(x + 1.5 * width, transliteration, width, label='Transliteration')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
ax.set_xlabel('Different metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.set_ylim(top=110, bottom=-5)

# Add numbers on top of the bars
ax.bar_label(rects1)
ax.bar_label(rects2)
ax.bar_label(rects3)
ax.bar_label(rects4)

fig.tight_layout()

# Default DPI is 100
plt.savefig('images/barplot_transformations.pdf', dpi=100)