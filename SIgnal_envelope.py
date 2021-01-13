import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy import interpolate

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

def get_envelope_v1(x, y, interval):

    assert len(x) == len(y)

    x=x[::interval]
    y=y[::interval]

    # First data
    ui, ux, uy = 0, x[0], y[0]
    li, lx, ly = 0, x[0], y[0]

    # Find upper peaks and lower peaks
    for i in range(1, len(x) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            ui = np.r_[ui, i]
            ux = np.r_[ux, x[i]]
            uy = np.r_[uy, y[i]]
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            li = np.r_[li, i]
            lx = np.r_[lx, x[i]]
            ly = np.r_[ly, y[i]]

    # Last data
    ui = np.r_[ui, len(x) - 1]
    ux = np.r_[ux, x[len(x) - 1]]
    uy = np.r_[uy, y[len(y) - 1]]

    li = np.r_[li, len(x) - 1]
    lx = np.r_[lx, x[len(x) - 1]]
    ly = np.r_[ly, y[len(y) - 1]]

    func_ub = interpolate.interp1d(ux, uy, kind='cubic', bounds_error=False)
    func_lb = interpolate.interp1d(lx, ly, kind='cubic', bounds_error=False)

    ub = func_ub(x)
    lb = func_lb(x)

    return ub, lb

Interval=200
uy, ly = get_envelope_v1(Time, Amplitude,Interval)
Envelope_Time=Time[::Interval]

fig, ax = plt.subplots()
ax.plot(Time, Amplitude, c="b")
ax.plot(Envelope_Time, uy, c="m")
ax.plot(Envelope_Time, ly, c="c")
plt.show()
#
#
# analytic_signal=hilbert(Amplitude, N=round(len(Amplitude)*0.1))
# Envelope_Amplitude = np.abs(analytic_signal)
# Envelope_Time=np.linspace(min(Time), max(Time), len(Envelope_Amplitude))
#
# plt.clf()
# plt.subplot(211)
# plt.cla()
# plt.plot(Time*1e06, Amplitude*1e03, label='W%d_D%d' % (w, d))
#
# plt.plot(Envelope_Time, Envelope_Amplitude*1e03, label='W%d_D%d' % (w, d))
# plt.xlim((min(Time)*1e06, max(Time)*1e06))
# plt.ylim((min(Amplitude)*1.2e03, max(Amplitude)*1.2e03))
# plt.xlabel('Time [us]')
# plt.ylabel('Amplitude [mV]')
# plt.grid()
# plt.legend(loc='lower right')
# plt.tight_layout()
