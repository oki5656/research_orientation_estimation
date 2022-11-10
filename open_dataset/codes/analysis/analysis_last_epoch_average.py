# This program is used for analysis last epoch result (LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss) of 25 trial.
# For example calculate each top 5 data average, plot result ascending order.

from statistics import mean
import matplotlib.pyplot as plt

#################################################################################################
Last_epoch_Result = {
"LastEpochTestAngleErr":[6.81858,8.98204,13.71329,14.21483,7.98704,8.92155,13.59949,9.02676,6.83336,5.72655,15.3052,8.6753,8.26029,4.69854,13.36101,9.91575],
"LastEpochTestDistanceErr":[199.67396,350.65436,342.60827,366.32961,322.16422,284.07606,390.69003,237.21095,186.21616,166.15754,384.58439,264.50111,214.12926,215.58754,391.92223,297.03712],
"LastEpochTestLoss":[97.19674,157.91808,183.07376,170.31676,164.31998,134.7551,197.28801,119.69387,93.08525,85.02995,187.70425,137.7478,107.76198,110.54442,200.50214,142.43503]

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
