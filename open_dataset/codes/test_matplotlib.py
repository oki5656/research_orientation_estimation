from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
 
# # ランダムな点を生成する(x, y, z座標)
# x = np.random.rand(50)
# y = np.random.rand(50)
# z = np.random.rand(50)
 
# # 点(x, y, z)がもつ量
# value = np.random.rand(50)
 
# # figureを生成する
# fig = plt.figure()
 
# # axをfigureに設定する
# ax = Axes3D(fig)
 
# # カラーマップを生成
# cm = plt.cm.get_cmap('RdYlBu')
 
# # axに散布図を描画、戻り値にPathCollectionを得る
# mappable = ax.scatter(x, y, c=value)#, cmap=cm)
# fig.colorbar(mappable, ax=ax)
 
# # 中身確認
# print("value")
# print(value)

# # 表示する
# plt.show()


x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

colors = np.random.random_sample((100, 3))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, color=colors)
plt.show()