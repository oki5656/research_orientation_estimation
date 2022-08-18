# This program is used for analysis last epoch result (LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss) of 25 trial.
# For example calculate each top 5 data average, plot result ascending order.

from statistics import mean
import matplotlib.pyplot as plt

#################################################################################################
Last_epoch_Result = {
"LastEpochTestAngleErr" : [24.19568, 30.97466, 32.15537, 26.53652, 33.0313, 25.01934, 16.58921, 26.16969, 42.50447, 32.47314, 32.13533, 21.8863, 19.23929, 40.39082, 21.22007, 26.24514, 40.23221, 30.1697, 30.951, 27.3692, 18.38939, 44.23119, 33.16408, 23.86596],
"LastEpochTestDistanceErr" : [259.19865, 488.0713, 482.48719, 418.10523, 160.91136, 322.01665, 511.43671, 324.03202, 375.27921, 454.45514, 314.20656, 467.97388, 359.34619, 501.8596, 505.95842, 475.23616, 338.48444, 543.63595, 345.64401, 402.1165, 437.69907, 329.12474, 461.30425, 519.95707],
"LastEpochTestLoss" : [126.67255, 238.6187, 229.11223, 210.62671, 77.47876, 156.48795, 234.82036, 153.67026, 178.45261, 223.28589, 152.50394, 223.70471, 175.34897, 236.23737, 242.05901, 228.66837, 158.23094, 271.07489, 167.7106, 194.55176, 222.72038, 158.68845, 224.36679, 257.29926]
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
