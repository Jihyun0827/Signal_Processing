import os
import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

Defect_Amplitude=[]
Defect_Time=[]

for d in [1]:
    for w in [1, 2, 3]:
        # plt.subplot(6, 3, i)
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

Defect_Amplitude=np.array(Defect_Amplitude)
Defect_Time=np.array(Defect_Time)

FFT_Amplitude = Defect_Amplitude[1,:]
FFT_Time = Defect_Time[1,:]

Sampling_time=FFT_Time[2]-FFT_Time[1]

Sample_Rate = 100  # Hertz
Sampling_Frequency=1/Sampling_time

FFT_Magnitude=fft(FFT_Amplitude)
FFT_Frequency=fftfreq(len(FFT_Amplitude)*Sample_Rate, 1/Sample_Rate)

plt.plot(FFT_Frequency, FFT_Magnitude)
plt.show()

2.0 / N * np.abs(yf[0:N // 2])

yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
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