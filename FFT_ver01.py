import os
import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

Defect_Amplitude=[]
Defect_Time=[]
i=1

for d in [1]:
    for w in [1, 2, 3]:
        plt.subplot(3, 1, i)
        ROOT = 'C:/Projects/LSAW_Defect__/Data/Width%d/Depth%d' % (w, d)
        data_file_list = os.listdir(ROOT)

        data_file_list=data_file_list[0]

        data_path = os.path.join(ROOT, data_file_list)
        print(data_file_list)
        Data_temp = np.genfromtxt(data_path, dtype=None, delimiter=',')

        Time_temp = Data_temp[::, 0]
        Amplitude_temp = Data_temp[::, 1]

        Defect_Amplitude.append(Amplitude_temp)
        Defect_Time.append(Time_temp)

        plt.plot(Time_temp, Amplitude_temp, label='W%d_D%d' % (w, d))
        plt.ylim((-0.003, 0.003))
        plt.grid()
        plt.legend()
        i += 1
        
        
plt.show()

Defect_Amplitude=np.array(Defect_Amplitude)
Defect_Time=np.array(Defect_Time)

Amplitude = Defect_Amplitude[2,:]
Time = Defect_Time[2,:]

# Cut Time
Cut_Start_percent=0.02
Cut_End_percent=0.7

Cut_Start_Idx=round(Cut_Start_percent*len(Amplitude))
Cut_End_Idx=round((1-Cut_End_percent)*len(Amplitude))

Cut_Amplitude = Amplitude[Cut_Start_Idx:Cut_End_Idx]
Cut_Time= Time[Cut_Start_Idx:Cut_End_Idx]

plt.plot(Cut_Time,Cut_Amplitude, label='W3_D1')
plt.ylim((-0.002, 0.002))
plt.grid()
plt.legend()
plt.show()

# FFT
FFT_Time=Cut_Time
FFT_Amplitude=Cut_Amplitude
Sampling_time=FFT_Time[2]-FFT_Time[1]

Sample_Frequency = 100
Frequency_Range = 5e06 # Hertz
Sampling_Frequency=1/Sampling_time

FFT_Magnitude=fft(FFT_Amplitude)
FFT_Frequency=fftfreq(len(FFT_Amplitude)*Sample_Rate, 1/Sample_Rate)
FFT_Frequency=FFT_Frequency[:Sample_Rate]

plt.plot(FFT_Frequency, FFT_Magnitude)
plt.show()

plt.plot(FFT_Magnitude)
plt.show()

# https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html

from scipy.fft import fft, fftfreq
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

plt.plot(x,y)
plt.show()
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

from scipy.fft import fft, fftfreq
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
from scipy.signal import blackman
w = blackman(N)
ywf = fft(y*w)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
plt.legend(['FFT', 'FFT w. window'])
plt.grid()
plt.show()