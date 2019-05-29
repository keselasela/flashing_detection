import numpy as np
c = np.arange(27).reshape(3,3,3)
print(c)

#conditions = (dst[:,:,0]<150)*1 * ((dst[:,:,0]>90)*1) * (dst[:,:,1]>100)
a = (c[:,:,0]>3)*1
b = (c[:,:,0]<21)*1
d = (c[:,:,1]>12)*1
print(a)
print(a+1)
print(sum(a))
t_a = np.transpose(a)

sum_item = np.sum(a)
temp = range(len(a))


