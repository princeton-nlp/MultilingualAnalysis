import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
sns.set()

font_size=18
font = {'size' : font_size}
matplotlib.rc('font', **font)
matplotlib.rcParams['lines.markersize'] = matplotlib.rcParams['lines.markersize'] + 8

# Define the means and variance of the athletes plot
alignment = np.array([90, 43, 11.8, 0.16, 0.01, 57.3, 4.9])
# alignment = np.array([7, 5, 4, 2, 1, 6, 3])

alpha = 0.0

xnli = np.array([-2.1, -3.8, -5.7, -19.2, -27.7, -5.7, -9.2])
# xnli = np.array([7, 6, 4, 2, 1, 5, 3])

# ner = np.array([-2.3, -4.1, -14.3, -51.5, -46.3, -14.2, -19.1])
# ner = np.array([7, 6, 4, 2, 1, 5, 3])

# pos = np.array([-0.5, -0.7, -1.5, -42.7, -59, -2, -2.5])
# pos = np.array([7, 6, 5, 2, 1, 4, 3])


# Set plot limits
# xmin = 5.5
# xmax=7.5
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(y-0.1,y+1)

# Plot the data
plt.scatter(xnli - alpha, alignment - alpha, color='r', marker='*', label='XNLI')
# plt.scatter(ner, alignment, color='g', marker='*', label='NER')
# plt.scatter(pos + alpha, alignment + alpha, color='b', marker='*', label='POS')


# Show the plot
plt.xlabel(r'$\mathbf{\Delta_{(BZ-BS)}}$', fontsize=font_size)
plt.ylabel("Alignment", fontsize=font_size, labelpad=8)
plt.tight_layout()
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)

plt.text(-27, 65, 'Spearman\'s\n' + r'$\rho = 0.94$, $p < .005$', fontsize = font_size-3, color='darkblue')
# plt.text(-50, 65, 'Spearman\'s correlation\n' + r'$\rho = 0.93$, $p < .005$', fontsize = font_size-2)
# plt.text(-60, 65, 'Spearman\'s correlation\n' + r'$\rho = 0.89$, $p < .01$', fontsize = font_size-2)

plt.legend(prop={'size': font_size-2})

# Add annotations for different points
delta=-0.5
alpha=0.7
dict_design = dict(facecolor='black', shrink=0.05, width=2, headwidth=8, alpha=0.15)
ax.annotate('Parallel', xy=(-2.1 + delta, 90), xytext=(-9, 88), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
ax.annotate('Non-parallel (Same)', xy=(-3.8 + delta, 43), xytext=(-15, 42), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
ax.annotate('Non-parallel (Diff)', xy=(-5.7 + delta, 11.8), xytext=(-15, 25), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)
ax.annotate(r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{inv}}$', xy=(-19.2 + delta, 0.16), xytext=(-25, 10), xycoords='data', textcoords='data', fontsize=font_size-4, arrowprops=dict_design, alpha=alpha)
ax.annotate(r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{perm}}$', xy=(-27.7 + delta, 0.01), xytext=(-28, 25), xycoords='data', textcoords='data', fontsize=font_size-4, arrowprops=dict_design, alpha=alpha)
ax.annotate(r'$\mathbf{\mathcal{T}}_{trans}}$' + r'$\circ$' + r'$\mathbf{\mathcal{T}}_{syn}}$', xy=(-5.7 + delta, 57.3), xytext=(-15, 55), xycoords='data', textcoords='data', fontsize=font_size-4, arrowprops=dict_design, alpha=alpha)
ax.annotate('Non-parallel (50%)', xy=(-9.2 + delta, 4.9), xytext=(-19, 13), xycoords='data', textcoords='data', fontsize=font_size-6, arrowprops=dict_design, alpha=alpha)

# plt.savefig('../images/correlation.png')
plt.savefig('../images/XNLI_correlation_annotated.pdf')