import os
import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from scipy.signal import blackman

Defect_Amplitude=[]
Defect_Time=[]
i=1

for d in [1]:
    for w in [1, 2, 3]:

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

Amplitude = Defect_Amplitude[2,:]
Amplitude -= np.mean(Amplitude)
Time = Defect_Time[2,:]

plt.clf()
plt.subplot(211)
plt.cla()
plt.plot(Time*1e06, Amplitude*1e03, label='W%d_D%d' % (w, d))

# Cut Time
Cut_Start_percent=0.02
Cut_End_percent=0.7

Cut_Start_Idx=round(Cut_Start_percent*len(Amplitude))
Cut_End_Idx=round((1-Cut_End_percent)*len(Amplitude))

Cut_Amplitude = Amplitude[Cut_Start_Idx:Cut_End_Idx]
Cut_Time= Time[Cut_Start_Idx:Cut_End_Idx]

plt.plot(Cut_Time*1e06,Cut_Amplitude*1e03, label='Cut for FFT')
plt.xlim((min(Time)*1e06, max(Time)*1e06))
plt.ylim((min(Cut_Amplitude)*1.2e03, max(Cut_Amplitude)*1.2e03))

plt.xlabel('Time [us]')
plt.ylabel('Amplitude [mV]')
plt.grid()
plt.legend(loc='lower right')
plt.tight_layout()

# FFT
FFT_Time=Cut_Time
FFT_Amplitude=Cut_Amplitude
Sampling_time=FFT_Time[2]-FFT_Time[1]

Sampling_rate = 100
Frequency_Range = 3e06 # Hertz
Sampling_Frequency=1/Sampling_time
Sampling_Number = len(FFT_Amplitude) * Sampling_rate

Frequency_Range_parameter=round(Frequency_Range / Sampling_Frequency * Sampling_Number)
FFT_Frequency=np.linspace(0.0, Frequency_Range_parameter, Frequency_Range_parameter+1)
FFT_Frequency*=Sampling_Frequency/Sampling_Number
FFT_Magnitude=fft(FFT_Amplitude,Sampling_Number)
FFT_Magnitude=np.abs(FFT_Magnitude)/len(FFT_Amplitude)*2
FFT_Magnitude = FFT_Magnitude[0:len(FFT_Frequency)]


plt.subplot(212)
plt.cla()
plt.plot(FFT_Frequency/1e06, FFT_Magnitude*1e03)
plt.xlabel('Frequency [MHz]')
plt.ylabel('Magnitude [mV]')
plt.xlim((min(FFT_Frequency)/1e06, max(FFT_Frequency)/1e06))
plt.ylim((0, max(FFT_Magnitude)*1.2e03))
plt.grid()
plt.tight_layout()
plt.show()
