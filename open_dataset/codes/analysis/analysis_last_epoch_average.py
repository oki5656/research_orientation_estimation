# This program is used for analysis last epoch result (LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss) of 25 trial.
# For example calculate each top 5 data average, plot result ascending order.

from statistics import mean
import matplotlib.pyplot as plt

#################################################################################################
Last_epoch_Result = {
"LastEpochTestAngleErr" :  [12.53458, 17.77147, 10.22558, 15.47443, 12.89866, 11.69971, 71.24879, 21.7614, 9.22577, 19.81381, 9.52992, 10.12857, 9.60968, 8.52421, 25.7438, 6.48391, 15.08651, 27.17507, 17.46106, 14.92762, 27.75736, 10.43945, 6.11968, 8.15799, 7.43864],
"LastEpochTestDistanceErr" :  [0.08361, 0.10832, 0.08476, 0.09717, 0.09442, 0.08933, 0.25624, 0.12137, 0.0596, 0.10974, 0.05946, 0.07218, 0.05863, 0.05703, 0.09839, 0.07687, 0.09285, 0.10452, 0.11679, 0.10952, 0.13633, 0.06298, 0.04573, 0.05602, 0.05229],
"LastEpochTestLoss" :  [0.04139, 0.04849, 0.04361, 0.0494, 0.04541, 0.04613, 0.12658, 0.06109, 0.02917, 0.05273, 0.02897, 0.03516, 0.0284, 0.02742, 0.05036, 0.03953, 0.04457, 0.05339, 0.05764, 0.05502, 0.0648, 0.03168, 0.02432, 0.02955, 0.02439]
}
#################################################################################################

RankingLastEpochTestAngleError = sorted(Last_epoch_Result["LastEpochTestAngleErr"])
RankingLastEpochTestDistanceErr = sorted(Last_epoch_Result["LastEpochTestDistanceErr"])
RankingLastEpochTestLoss = sorted(Last_epoch_Result["LastEpochTestLoss"])
print("Average last epoch test angle error", round(mean(RankingLastEpochTestAngleError[:5]), 5))
print("Average last epoch test distance error", round(mean(RankingLastEpochTestDistanceErr[:5]), 5))
print("Average last epoch test loss", round(mean(RankingLastEpochTestLoss[:5]), 5))

# graph plotting
Xaxis = [i for i in range(25)]
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
