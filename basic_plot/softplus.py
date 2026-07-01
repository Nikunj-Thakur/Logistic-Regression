import numpy as np

a=1000  # when a is very large
print(np.exp(a)) # gives infinity inf
print(np.log(1+np.exp(a))) # gives infinity inf

'''
But we know that when a is very large (larger than threshold 20 in this case), log(1+np.exp(a)) ~ a, so we can simply return a
'''
if(a>=20): # 
    print(a)

'''
NumPy also provides a built-in implementation of Softplus function which computes
log(e^0+e^x)=log(1+e^x) in a numerically stable way without needing to choose a threshold yourself.
'''
print(np.logaddexp(0, a)) 