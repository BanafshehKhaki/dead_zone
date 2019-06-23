import h5py
import numpy as np

##############
# Printing Weights for checking 
##############

filename = 'Results/test/ph/weights_checkpoints/sonde1_2018-06-01-2018-07-01_TD3_PrH288_weights_checkpoint_epoch30_june19.hdf5'
f2 = h5py.File(filename, 'r')
g2 = f2.get('gru_1/gru_1')
l2 = list(g2.items())
d2 =np.array(g2.get('recurrent_kernel:0'))
print(d2)
f2.close()
# l:
# bias:0 , kernel:0 , recurrent_kernel:0

f2 = h5py.File(filename, 'r')
g2 = f2.get('gru_2/gru_2')
l2 = list(g2.items())
# bias:0 , kernel:0 , recurrent_kernel:0
d2 =np.array(g2.get('recurrent_kernel:0'))
print(d2)
f2.close()

g2 = f2.get('dense_1/dense_1')
#'bias:0' , 'kernel:0'
d2 =np.array(g2.get('kernel:0')
print(d2)
f2.close()

# list(f.keys())
# Out[204]: ['dense_1', 'gru_1', 'gru_2']
filename = 'Results/test/pHcategory/weights_checkpoints/sonde1_2018-06-01-2018-07-01_TD3_PrH288_weights_checkpoint_epoch30_june19.hdf5'
f2 = h5py.File(filename, 'r')
g2 = f2.get('gru_1/gru_1')
l2 = list(g2.items())
d2 =np.array(g2.get('recurrent_kernel:0'))
print(d2)
f2.close()











