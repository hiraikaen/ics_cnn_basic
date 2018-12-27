import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import time
import csv

coef = 6
width = 200*coef

def bit_signal(t):
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

#plt.figure()

# def bit_signal_noise(coef_f, cycle=5, rate=100, level_noise=1.5):
#     t = np.linspace(0, 2*np.pi*cycle, 2*np.pi*cycle*rate, endpoint=False)
#     noise = np.random.uniform(-pow(0.1,level_noise), pow(0.1, level_noise), [1,int(2*np.pi*cycle*rate)])[0]
#     data = signal.square(t/pow(2,coef_f)) + noise
#     data = data + abs(min(data))
#     data = data/max(data)*255.0
#     data = data.astype(int)
    
#     return t, data

def bit_signal_noise(coef_f, cycle=5, rate=100, level_noise=1.5, offset=0):
    t = np.linspace(0, 2*np.pi*cycle, 2*np.pi*cycle*rate, endpoint=False)
    t = t+offset
    noise = np.random.uniform(-pow(0.1,level_noise), pow(0.1, level_noise), [1,int(2*np.pi*cycle*rate)])[0]
    data = signal.square(t/pow(2,coef_f)) + noise
    data = data + abs(min(data))
    data = data/max(data)*255.0
    data = data.astype(int)
    
    return t, data

level_noise = 1.5
rate = 20 #resolution per cycle
max_coef_f = 4
cycle = 50

data_box = ''
for i in range(max_coef_f):
    for j in range(int(rate)):
        buf_str = ','+str(i)+'x'+str(j)
        data_box += buf_str
    buf_str = '\n'
    data_box += buf_str


def compose_signals(max_coef_f, attack=False, attack_rate=0.2, visualize=False, offset_idx=None, offset_vol=None):
    plt.figure()
    composed_data = list()
    for i in range(max_coef_f):
        coef_f = i

        # if i == offset_idx:
        #     offset_vol = offset_vol
        #     print('   >>> adding offset:', offset_vol, 'to_idx:', offset_idx)
        # else:
        #     offset_vol = 0

        if i == offset_idx:
            t, data = bit_signal_noise(coef_f=coef_f, cycle=cycle, rate=rate, offset=offset_vol)
            print('   >>> adding offset:', offset_vol, 'to_idx:', offset_idx)
        else:
            t, data = bit_signal_noise(coef_f=coef_f, cycle=cycle, rate=rate)
            
        if attack:
            #add max value in signal in random spots
            rand_spots = np.random.randint(0, len(data), size=int(len(data)*attack_rate))
            bar = 10
            for idx in rand_spots:
                if idx+bar < len(data):
                    data[idx:idx+bar] = 0
            
            print('attack', data[:10], type(data), max(data), min(data))

        elif offset_idx != None:
            print('attack offset', data[:10], type(data), max(data), min(data))

        else:
            print('normal', data[:10], type(data), max(data), min(data))
            
            
        if visualize:
            plt.subplot(max_coef_f,1,coef_f+1)
            plt.plot(t, data)
        print('length of each slot:', len(data))
        composed_data.append(data)
        
    #print('total length t:', len(composed_data[0]))
    #input('ok?')
    if visualize:
        if attack:
            print('save attack...')
            plt.savefig('abnormal_composed.pdf')
        elif offset_idx != None:
            print('save attack_offset...')
            plt.savefig('abnormal_offset_composed.pdf')            
        else:
            print('save normal...')
            plt.savefig('normal_composed.pdf')
    print('len of composed_data:', len(composed_data)*len(composed_data[0]))
    return composed_data
    
#slice and insert data
# normal_data = list()
# for i in range(max_coef_f):
#     coef_f = i
#     t, data = bit_signal_noise(coef_f=coef_f, cycle=cycle, rate=rate)
#     plt.subplot(max_coef_f,1,coef_f+1)
#     plt.plot(t, data)
#     normal_data.append(data)

#compose normal signal
normal_data = compose_signals(max_coef_f, attack=False, visualize=True)
#compose abnormal signal (by zero bar noise)
abnormal_data = compose_signals(max_coef_f, attack=True, attack_rate=0.005, visualize=True)

np.random.seed(3)
normal_test_data = compose_signals(max_coef_f, attack=False, visualize=True)
#compose abnormal signal (by zero bar noise)
abnormal_test_data = compose_signals(max_coef_f, attack=True, attack_rate=0.005, visualize=True)

np.random.seed(4)
abnormal_offset_train_data = compose_signals(max_coef_f, attack=False, attack_rate=0.0, visualize=True, offset_idx=1, offset_vol=20)

np.random.seed(5)
abnormal_offset_test_data = compose_signals(max_coef_f, attack=False, attack_rate=0.0, visualize=True, offset_idx=1, offset_vol=20)


#todo
#compose abnormal signal (by latency offset)
#abnormal_data_latency = ...

#print('total length t:', len(normal_data[0]))
#input('ok?')
    
plt.savefig('tmp.pdf')
    
#print(normal_data)
#input('ok?')

#loop have to be rebuilt
#
#for at t=0,1,2,... size-width
#   add label to buf
#   for at s=0,1,2,...n(sensors)
#      for at t_cursor=0,1,...,width
#         add data[s, t_cursor] to buf
#   add n to data
#   add buf to data

def make_images(ts_data, width, label):
    time_start = time.time()
    images = list()
    for t in range(len(ts_data[0])-width):
        buf = [label]
        #for s in range(len(ts_data)):
        #print('len(ts_data) == 4?', len(ts_data), 't:', t, t+width)
        #input('ok?')
        for s in range(len(ts_data)):
            #buf += list(ts_data[s][t:t+width])
            buf += ts_data[s][t:t+width].tolist() #faster
        # print('a image:', buf)
        # input('ok?')
        images.append(buf)
        #print('t:', t, 'n of images:', len(images), 'n of each image:', len(buf))
        if t % 3000 == 0:
            print('progress:', t, '/', len(ts_data[0])-width)
            
    time_elapsed = time.time() - time_start
    print('total num of image:', len(images))
    print('time elapsed:', time_elapsed)
    
    return images

#make normal_data to images
normal_images = make_images(normal_data, width, 0)
#make abnormal_data to images
abnormal_images = make_images(abnormal_data, width, 1)
abnormal_offset_train_images =  make_images(abnormal_offset_train_data, width, 2)

#make normal_data to images test
normal_test_images = make_images(normal_test_data, width, 0)
#make abnormal_data to images test
abnormal_test_images = make_images(abnormal_test_data, width, 1)
abnormal_offset_test_images =  make_images(abnormal_offset_test_data, width, 2)

print('writing csv files...')
with open("ics_train.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(normal_images)
    writer.writerows(abnormal_images)
    writer.writerows(abnormal_offset_train_images)

#test data == train data same temporarily
with open("ics_test.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(normal_test_images)
    writer.writerows(abnormal_test_images)
    writer.writerows(abnormal_offset_test_images)
