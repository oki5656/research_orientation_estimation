# This program is used for analysis last epoch result (LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss) of 25 trial.
# For example calculate each top 5 data average, plot result ascending order.

from statistics import mean
import matplotlib.pyplot as plt

#################################################################################################
Last_epoch_Result = {
"LastEpochTestAngleErr":[11.16501,14.31914,8.16444,11.4913,14.01482,6.9775,10.9188,17.89321,7.36926,16.88079,11.32528,8.57346,11.70079,12.54799,9.57659,10.33737,8.00607,6.44142,6.87469],
"LastEpochTestDistanceErr":[383.55754,362.83457,191.72593,267.80065,351.74858,254.52345,288.14241,436.73523,235.66862,442.32507,294.51729,251.69755,284.86358,311.31938,230.13869,290.95573,224.8735,202.69628,186.66995],
"LastEpochTestLoss":[182.84636,161.87907,90.76284,128.90289,180.64966,122.65343,130.32202,203.38156,112.08075,228.80316,139.78867,127.71734,138.36514,164.54047,113.69171,147.49689,103.04497,95.46465,90.79352]

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
