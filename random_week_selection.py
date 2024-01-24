# In our experiments we evaluate our schedulers week by week.
# Our workload is constituted of 51 weeks, ranging from January the 3rd 2022 to December the 25th 2022.
# This code randomly generate 12 numbers between 1 and 51 using the seed "0".
# We used these 12 randomly selected weeks to conduct our experiments.

import random

random.seed(0)

nums = list(range(1, 52))

random.shuffle(nums)

print(nums[0:12]) # List of 12 unique random numbers

# The produce numbers are [28, 13, 43, 41, 42, 8, 6, 36, 2, 50, 40, 1]
# Which corresponds to the weeks (format is month-day):

# 28:	07-11 07-17
# 13:	03-28 04-03
# 43:	10-24 10-30
# 41:	10-10 10-16
# 42:	10-17 10-23
# 8:	02-21 02-27
# 6:	02-07 02-13
# 36:	09-05 09-11
# 2:	01-10 01-16
# 50:	12-12 12-18
# 40:	10-03 10-09
# 1: 	01-03 01-09

# The full list would be [28, 13, 43, 41, 42, 8, 6, 36, 2, 50, 40, 1, 5, 35, 21, 15, 51, 29, 34, 44, 39, 12, 30, 18, 16, 11, 22, 48, 24, 4, 45, 10, 47, 7, 37, 19, 9, 46, 14, 38, 23, 31, 20, 26, 32, 33, 17, 3, 27, 49, 25]
