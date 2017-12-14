import matplotlib.pyplot as plt
fig = plt.figure()
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'


ax1 = plt.subplot()

x_cora = [1000,3000,5000]
y_cora = [78.4, 81.2, 82.8]

ax1.plot(x_cora,y_cora, linewidth=2, linestyle=':', marker='o', label='TF-IDF with global features into an MLP accuracies')
plt.xlabel('Vocabulary size')
plt.ylabel('Testing Accuracy')

leg=plt.legend(loc='best', numpoints=1, fancybox=True)

# Axes alteration to put zero values inside the figure Axes
# Avoids axis white lines cutting through zero values - fivethirtyeight style

ax1.set_title('Training on different values of Vocabulary size', fontstyle='italic')

#fig.suptitle("Training on different size of Graphs")
#plt.savefig('seaborn-style.svg', bbox_inches='tight')
plt.show()
