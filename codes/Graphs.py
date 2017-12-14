import numpy as np
acc_1 = [0.80000067, 0.81600076, 0.77100062, 0.80500072, 0.75000054, 0.76400059, 0.65000027, 0.77100062, 0.70100039, 0.72600049, 0.65900028, 0.138]
acc_1 = acc_1[::-1]
acc_2 = [0.76700062, 0.64700025, 0.64000022, 0.54899997, 0.22199991, 0.24199986, 0.17200001, 0.154, 0.35199955]
hyperPara = [10,20,30,40,50,60,70,80,90,100,120,130]
hyperPara = hyperPara[::-1]
x = list((-1)*np.asarray(hyperPara)+ 140)
x_1 = [190,210,220,230,250,280,290,330,320]
x_cora = x+ x_1
print(x)
x_citeseer = [20,30,40,50,60,70,80,90,100,110,120, 150, 160 ,170 ,180,190 ,200,210,230 ,260,270 ,300,310 ]
y_citeseer = [0.076999992, 0.079000004, 0.090000018, 0.16800004, 0.18400005, 0.34499988, 0.65199983, 0.69600016, 0.70000017, 0.71300024, 0.721, 0.61999971, 0.7,0.67499995,0.63500, 0.59799957, 0.37999982, 0.48499969, 0.24999999, 0.19100004, 0.19800003, 0.25899997, 0.31499994]
y_cora = acc_1 + acc_2
x_pubmed = [10,20,30,40,50,60,80,100, 120,140,160,180,200,220,240,260,280,300,320]
y_pubmed = [0.17999995, 0.17999995, 0.19899994, 0.74800104, 0.76800108, 0.75000,0.49700,0.57000,0.67500, 0.48100,0.40600,0.18, 0.42100,0.40900,0.41100,0.40900,0.51400,0.41300, 0.40300]
print(x_cora)
print(y_cora)
import matplotlib.pyplot as plt
fig = plt.figure()
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# Marketing channels line plot
"""ax1 = plt.subplot()

ax1.plot( x_cora,y_cora, linewidth=2, linestyle=':', marker='o', label='Cora')
ax1.plot(x_citeseer, y_citeseer,linewidth=2, linestyle='--', marker='v', label='Citeseer')
ax1.plot(x_pubmed, y_pubmed,linewidth=2, linestyle='--', marker='v', label='PubMed')
#.plot(y_series, x_2, linewidth=2, linestyle='--', marker='v', label='telemarketing')
#ax3.plot(y_series, x_3, linewidth=2, linestyle='-.', marker='s', label='sales team')
plt.xlabel('No.of Training Samples')
plt.ylabel('Testing Accuracy')

leg=plt.legend(loc='best', numpoints=1, fancybox=True)

# Axes alteration to put zero values inside the figure Axes
# Avoids axis white lines cutting through zero values - fivethirtyeight style

ax1.set_title('Training on different size of Graphs', fontstyle='italic')

#fig.suptitle("Training on different size of Graphs")
#plt.savefig('seaborn-style.svg', bbox_inches='tight')
plt.show()


ax2 = plt.subplot()
x_nodes = [16,32,64,128,256,512]
y_2 = [0.81000072, 0.82300073, 0.81700075, 0.81500071, 0.81300074, 0.81400073]
y_3 = [0.79800069, 0.79700071, 0.80300069, 0.79500067, 0.78100061, 0.76500058]
y_4 = [0.74600053, 0.78000063, 0.77400059, 0.77000058, 0.74100053, 0.74600053]
y_5 = [0.73700052, 0.77100062, 0.73800051, 0.78500062, 0.75100052, 0.70800042]
y_6 = [0.6960004, 0.77500063, 0.76400059, 0.72400045, 0.72200048, 0.32299963]
y_7 = [0.63900024, 0.73200047, 0.72000045, 0.68800038, 0.18499997, 0.28399974]
ax2.plot( x_nodes,y_2, linewidth=2, linestyle=':', marker='o', label='2 Hidden Layers')
ax2.plot(x_nodes, y_3,linewidth=2, linestyle='--', marker='v', label='3 Hidden layers')
ax2.plot(x_nodes, y_4,linewidth=2, linestyle='--', marker='v', label='4 Hidden layers')
ax2.plot(x_nodes, y_5,linewidth=2, linestyle='--', marker='v', label='6 Hidden layers')
ax2.plot(x_nodes, y_6,linewidth=2, linestyle='--', marker='v', label='8 Hidden layers')
ax2.plot(x_nodes, y_7,linewidth=2, linestyle='--', marker='v', label='10 Hidden layers')

plt.xlabel('No.of Hidden Nodes')
plt.ylabel('Testing Accuracy')

leg=plt.legend(loc='best', numpoints=1, fancybox=True)

ax2.set_title('Accuracy VS No of Hidden Nodes', fontstyle='italic')

plt.show()

############## Loss vs hidden nodes #######################
ax3 = plt.subplot()
loss_2 = [1.2582082, 1.1073905, 1.0014384, 0.92608148, 0.87701112, 0.83357441]
loss_3 = [0.84347689, 0.78620434, 0.76190704, 0.85326332, 1.1471072, 1.2733232]
loss_4 = [1.1009367, 1.1009841, 1.3712401, 1.5357549, 2.4593627, 2.3715489]
loss_5 = [1.665536, 1.8168284, 2.1743279, 2.0025313, 9.0177107, 180.97577]
loss_6 = [2.2568319, 1.7870873, 2.6102355, 2.8760269, 10.358013, 28.059679]
loss_7 = [1.7012841, 2.0650334, 3.9397058, 1.7307527, 5.6378884, 3014.3125]
ax3.plot(x_nodes, loss_2,linewidth=2, linestyle='--', marker='v', label='2 Hidden Layers')
ax3.plot(x_nodes, loss_3,linewidth=2, linestyle='--', marker='o', label='3 Hidden Layers')
ax3.plot(x_nodes, loss_4,linewidth=2, linestyle='--', marker='o', label='4 Hidden Layers')
ax3.plot(x_nodes, loss_5,linewidth=2, linestyle='--', marker='v', label='6 Hidden layers')
ax3.plot(x_nodes, loss_6,linewidth=2, linestyle='--', marker='v', label='8 Hidden layers')
ax3.plot(x_nodes, loss_7,linewidth=2, linestyle='--', marker='v', label='10 Hidden layers')

plt.xlabel('No.of Hidden Nodes')
plt.ylabel('Testing Loss')
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

ax3.set_title('Accuracy VS No of Hidden Nodes', fontstyle='italic')

plt.show()

###########################################################################################
##########################    Cora Datset      ############################################

x_dropout = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
y_dropout_acc =[0.81000072, 0.82600075, 0.82100075, 0.82400078, 0.81900072, 0.82000077, 0.81700075, 0.81500071, 0.80100071]
y_dropout_loss =  [1.0798782, 1.0681982, 1.0834199, 1.075927, 1.1057713, 1.1190468, 1.164982, 1.1990341, 1.4369359]
ax5 = plt.subplot()
ax5.plot(x_dropout, y_dropout_acc,linewidth=2, linestyle='--', marker='v', label='Test Accuracy')
ax5.plot(x_dropout, y_dropout_loss, linewidth=2, linestyle=':', marker='o', label='Test Loss')
plt.xlabel('Dropout Rates')
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

ax5.set_title('Accuracy VS Dropout Rate', fontstyle='italic')
plt.show()

############################################################################################
########################### WeBKb Data######################################################
############################################################################################
"""
ax4 = plt.subplot()
x_label = [2.8,5,10,12.5,15,17.5,20]
acc_per = [0.56143, 0.61200, 0.60400, 0.61800, 0.60400, 0.63000, 0.62800]
ax4.plot(x_label, acc_per,linewidth=2, linestyle='--', marker='v', label='Test Accuracy')
"""loss_per = [1.38653, 1.1073905, 1.36218,1.36850,  1.36218, 1.35799, 1.34936]
ax4.plot(x_label, loss_per,linewidth=2, linestyle=':', marker='o', label='Test Loss')"""
plt.xlabel('Percent of data for Training')
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

ax4.set_title('WebKB Data', fontstyle='italic')
plt.show()

# Hidden Nodes vs acc and loss   
# 2 layers

x_web = [16,32,64,128,256,512]  
y_web_acc = [0.61399972, 0.60399973, 0.58999979, 0.5619998, 0.36600003, 0.58599973]
y_web_loss = [1.4681157, 1.5384302, 1.4725921, 1.5185881, 2.2414355, 1.8739171]
# Dropout rate vs accuracy

x_web_drop = [0.1,0.2,0.3,0.4,0.5,0.6]
y_web_drop_acc =[0.61600,0.57600,0.59800,0.552000.60000,0.55400]
y_web_drop_loss = [1.53477 ,1.43179,1.45134,1.42299,1.36398,1.46245]