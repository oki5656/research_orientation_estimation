# This program is used for analysis last epoch result (LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss) of 25 trial.
# For example calculate each top 5 data average, plot result ascending order.

from statistics import mean
import matplotlib.pyplot as plt

#################################################################################################
Last_epoch_Result = {
    "LastEpochTestAngleErr": [
        12.21778,
        11.82203,
        10.05142
    ],
    "LastEpochTestDistanceErr": [
        369.55231,
        353.72016,
        337.76517
    ],
    "LastEpochTestLoss": [
        175.11496,
        171.65381,
        176.36253
    ]
}
#################################################################################################

class AnalysisLastEpochAverage():
    def _init_():
        Last_epoch_Result = {
            "LastEpochTestAngleErr":[],
            "LastEpochTestDistanceErr":[],
            "LastEpochTestLoss":[]
        }

    def add_result(self, test_angle_err, test_distance_err, test_loss):
        self.Last_epoch_Result["LastEpochTestAngleErr"].append( test_angle_err)
        self.Last_epoch_Result["LastEpochTestDistanceErr"].append(test_distance_err)
        self.Last_epoch_Result["LastEpochTestLoss"].append(test_loss)
        pass

    def calc_average_last_epoch_result(self):
        RankingLastEpochTestAngleError = sorted(self.Last_epoch_Result["LastEpochTestAngleErr"])
        RankingLastEpochTestDistanceErr = sorted(self.Last_epoch_Result["LastEpochTestDistanceErr"])
        RankingLastEpochTestLoss = sorted(self.Last_epoch_Result["LastEpochTestLoss"])
        # print("Average last epoch test angle error", round(mean(RankingLastEpochTestAngleError[:5]), 5))
        # print("Average last epoch test distance error", round(mean(RankingLastEpochTestDistanceErr[:5]), 5))
        # print("Average last epoch test loss", round(mean(RankingLastEpochTestLoss[:5]), 5))
        average_last_epoch_test_angle_error = round(mean(RankingLastEpochTestAngleError[:5]), 5)
        average_last_epoch_test_distance_error = round(mean(RankingLastEpochTestDistanceErr[:5]), 5)
        average_last_epoch_test_loss = round(mean(RankingLastEpochTestLoss[:5]), 5)
        
        return average_last_epoch_test_angle_error, average_last_epoch_test_distance_error, average_last_epoch_test_loss


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
