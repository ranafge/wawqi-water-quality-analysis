# pandas ekta tool jeta excel er moto data jemon table akare karte pare
# row and colum
# data analysis karte help kare

import pandas as pd

import pandas as pd

data = {
    'নাম': ['রানা', 'তুহিন', 'নুসরাত'],
    'রেজাল্ট': [90, 85, 95]
}

df = pd.DataFrame(data)
print(df)

#      নাম  রেজাল্ট
# 0   রানা       90
# 1  তুহিন       85
# 2  নুসরাত       95

import numpy as np

# ganitik calculation a user hoy +-*/ matrix , gor etc

arr = np.array([1,2,3,4])

print("SUM ", np.sum(arr)) 
# 10

print("Gor", np.mean(arr))
# 2.5

import matplotlib.pyplot as plt
# bivinto dharoner chart graph er kaje use hoy

x = [1, 2, 3]
y = [5, 7, 4]

plt.plot(x, y)
plt.title("my title")
plt.xlabel("thisis x label")
plt.ylabel("thisis y label")
plt.show()


# seaborn (matplotlib er sundor and sohoj varsion)

import seaborn as sns

sns.barplot(x='নাম', y='রেজাল্ট', data=df)
plt.title("রেজাল্ট bar chart")
plt.show()