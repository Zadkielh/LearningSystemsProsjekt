import numpy as np
from keras.datasets import cifar10
import os
print("Current working directory:", os.getcwd())


(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

Y_test_scores_threshold = np.loadtxt("class_sums/CIFAR10AdaptiveThresholding_99_2000_500_10.0_10_32_1.txt", delimiter=',')
Y_test_scores_thermometer_3 = np.loadtxt("class_sums/CIFAR10ColorThermometers_99_2000_1500_2.5_3_8_32_1.txt", delimiter=',')
Y_test_scores_thermometer_4 = np.loadtxt("class_sums/CIFAR10ColorThermometers_99_2000_1500_2.5_4_8_32_1.txt", delimiter=',')
Y_test_scores_hog = np.loadtxt("class_sums/CIFAR10HistogramOfGradients_99_2000_50_10.0_0_32_0.txt", delimiter=',')
Y_test_scores_canny = np.loadtxt("class_sums/CIFAR10Canny_44_2000_500_5.0_10_32_1.txt", delimiter=',')
Y_test_scores_lbp = np.loadtxt("class_sums/CIFAR10LBP_5_2000_700_4.0_7_32_1.txt", delimiter=',')

votes = np.zeros(Y_test_scores_threshold.shape, dtype=np.float32)
for i in range(Y_test.shape[0]):
    votes[i] += 1.0*Y_test_scores_threshold[i]/(np.max(Y_test_scores_threshold) - np.min(Y_test_scores_threshold))
    votes[i] += 1.2*Y_test_scores_thermometer_3[i]/(np.max(Y_test_scores_thermometer_3) - np.min(Y_test_scores_thermometer_3))
    votes[i] += 1.2*Y_test_scores_thermometer_4[i]/(np.max(Y_test_scores_thermometer_4) - np.min(Y_test_scores_thermometer_4))
    votes[i] += 1.3*Y_test_scores_hog[i]/(np.max(Y_test_scores_hog) - np.min(Y_test_scores_hog))
    votes[i] += 0.4*Y_test_scores_canny[i]/(np.max(Y_test_scores_canny) - np.min(Y_test_scores_canny))
    votes[i] += 0.3*Y_test_scores_lbp[i]/(np.max(Y_test_scores_lbp) - np.min(Y_test_scores_lbp))

Y_test_predicted = votes.argmax(axis=1)

print("Team Accuracy: %.1f" % (100*(Y_test_predicted == Y_test).mean()))