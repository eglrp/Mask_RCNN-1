from matplotlib import pyplot as plt
import os 
import numpy as np 

accu = np.array([0, 80.69328429037706, 82.14651087543461, 83.09619634784759
,85.21866899554152, 89.35465690121055, 89.35992388764861
,84.90958809445147, 79.40773868886754, 76.46443534595892
,77.56074663624167]) 

# accu_rust = np.array([0 , 14.82, 24.59, 32.22, 54.77 , 62.54, 64 , 74 , 76, 78  , 
#                         80 , 82 , 85, 87, 83 , 76 , 74 , 72 ,70 , 70 , 68 ])
# accu_rust_last_report = np.array([0 , 14.82, 32, 54, 65 , 74, 83 , 86 , 84, 86  , 
#                         82 , 80 , 76, 72, 74 , 72 , 68 , 63 ,60 , 62 , 58 ])
accu_rust_bridge = np.array([0 , 20, 40, 60, 89 , 85, 82 , 84 , 82, 84  , 
                        82 , 79 , 75, 74, 71 , 69 , 66 , 65 ,59 , 62 , 56 ])
# accu_rust_ = np.array([0 , 15, 20, 41, 60 , 62, 71 , 79 , 84, 88  , 
#                         84 , 76 , 72, 69, 70 , 64 , 66 , 60 ,59 , 55 , 58 ])
# print(len(accu_rust))
# exit()
epoch = np.arange(21)
plt.figure()
plt.plot(epoch, accu_rust_bridge, label = 'Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Precision(%)')
plt.savefig('bridge_train_drop.png')