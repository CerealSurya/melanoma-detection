from matplotlib import pyplot as plt
fpr_keras = []
tpr_keras = []
thresholds = []
with open("fpr.txt", "r") as f:
    for line in f.readlines():
        fpr_keras.append(float(line))
with open("tpr.txt", "r") as f:
    for line in f.readlines():
        tpr_keras.append(float(line))
with open("thresholds.txt", "r") as f:
    for line in f.readlines():
        thresholds.append(float(line))
auc_keras = 0.7745121745612011
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()