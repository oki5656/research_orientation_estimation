# This program is used for analysis last epoch result (LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss) of 25 trial.
# For example calculate each top 5 data average, plot result ascending order.

from statistics import mean
import matplotlib.pyplot as plt

#################################################################################################
Last_epoch_Result = {
"LastEpochTestAngleErr" :  [18.79257, 23.34381, 23.47875, 14.47183, 18.93212, 24.36137, 19.78794, 19.10458, 15.73708, 19.55578],
"LastEpochTestDistanceErr" :  [910.46345, 1018.29508, 1009.2899, 702.7363, 800.39807, 1103.41734, 893.18019, 824.98506, 773.52323, 959.50234],
"LastEpochTestLoss" :  [407.32977, 485.46469, 500.65546, 299.95074, 407.51227, 530.83185, 422.9697, 400.02686, 359.03461, 440.14972]
}
#################################################################################################

RankingLastEpochTestAngleError = sorted(Last_epoch_Result["LastEpochTestAngleErr"])
RankingLastEpochTestDistanceErr = sorted(Last_epoch_Result["LastEpochTestDistanceErr"])
RankingLastEpochTestLoss = sorted(Last_epoch_Result["LastEpochTestLoss"])
print("Average last epoch test angle error", round(mean(RankingLastEpochTestAngleError[:5]), 5))
print("Average last epoch test distance error", round(mean(RankingLastEpochTestDistanceErr[:5]), 5))
print("Average last epoch test loss", round(mean(RankingLastEpochTestLoss[:5]), 5))

# graph plotting
number_of_last_epoch_result = len(Last_epoch_Result["LastEpochTestAngleErr"])
Xaxis = [i for i in range(number_of_last_epoch_result)]
Yaxis1 = RankingLastEpochTestAngleError
Yaxis2 = RankingLastEpochTestDistanceErr
Yaxis3 = RankingLastEpochTestLoss

fig = plt.figure()

# 1行2列に分割した中の1(左側)
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(Xaxis, Yaxis1, marker="o", color = "red", linestyle = "--")

# 1行2列に分割した中の2(右側)
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(Xaxis, Yaxis2, marker="v", color = "blue", linestyle = ":")

# 1行2列に分割した中の2(右側)
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(Xaxis, Yaxis3, marker="v", color = "green", linestyle = ":")

ax1.set_title("test angle error")
ax2.set_title("test distance error")
ax3.set_title("epoch test loss")

plt.show()
