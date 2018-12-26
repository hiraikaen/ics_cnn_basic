import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import signal
import numpy as np

def bit_signal(t):#, f):
        #return signal.square(2*np.pi*f*t, duty=0.5).astype('float64')
        return signal.square(t).astype('float64')
    
np.random.seed(2)

T = 2
L = 500
N = 10

#x = np.empty((N, L), 'int64')
#x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
#x[:] = np.array(range(L))
#x = np.linspace(0, 1, 500, endpoint=False)
#print(x)
#data = np.sin(x / 1.0 / T).astype('float64')
# data = bit_signal(x)
# print(data[0])
# plt.plot(x[0], data[0])
# plt.ylim(-2, 2)
#plt.savefig('tmp.pdf')
#plt.show()
#torch.save(data, open('traindata.pt', 'wb'))

#label,1,2,3,4,5,6,7,8,9,10
#0,154,232,....

plt.figure()

def bit_signal_noise(coef_f, cycle=5, rate=100, level_noise=1.5):
    t = np.linspace(0, 2*np.pi*cycle, 2*np.pi*cycle*rate, endpoint=False)
    noise = np.random.uniform(-pow(0.1,level_noise), pow(0.1, level_noise), [1,int(2*np.pi*cycle*rate)])[0]
    data = signal.square(t/pow(2,coef_f)) + noise
    return t, data

level_noise = 1.5
rate = 50 #resolution per cycle
max_coef_f = 5
cycle = 100

data_box = 'label'
for i in range(max_coef_f):
    for j in range(int(rate)):
        buf_str = ','+str(i)+'x'+str(j)
        data_box += buf_str
    buf_str = '\n'
    data_box += buf_str

#slice and insert data
buf_data = list()
for i in range(max_coef_f):
    coef_f = i
    t, data = bit_signal_noise(coef_f=coef_f, cycle=cycle, rate=rate)
    plt.subplot(max_coef_f,1,coef_f+1)
    plt.plot(t, data)
    buf_data.append(data)
   
print(buf_data)
input('ok?')

width = rate
# print('dbg:', buf_data[0][0].itemsize-width)
#print(buf_data[0].size)

#
#loop have to be rebuilt
#
for i in range(buf_data[0].size-width):
    #data_box = '0,'+','.join(str(buf_data[i]))+'\n'
    data_box += '0,'
    for j in range(buf_data[i] - width):
        idx = 0
        for number in buf_data[i][idx:width]:
            buf_str = ','+str(round(number,4))
            data_box += buf_str
    buf_str += '\n'
    print(data_box)
    print('size:', len(data_box), len(data_box[0]))
    input('ok?')
torch.save(data_box, open('traindata.pt', 'wb'))
    

plt.savefig('tmp4.pdf')
