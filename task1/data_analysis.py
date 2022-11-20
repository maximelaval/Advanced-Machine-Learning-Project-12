import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy as bp
from scipy.stats import entropy
import pywt
import biosppy.signals.ecg as ecg

print(pywt.Modes.modes)
def compute_entropy(signal):
    value, counts = np.unique(signal, return_counts=True)
    return entropy(counts)

def compute_wavelet_energy(signal): # look at wavelets to look if there are better alternatives
    dwt_mode = getattr(pywt.Modes, "periodic")
    # print(pywt.wavelist(kind="discrete"))
    # print(pywt.Wavelet('db1'))
    # print(pywt.Wavelet('db2'))
    # print(pywt.Wavelet('db3'))
    # print(pywt.Wavelet('db10'))
    # print(pywt.Wavelet('db25'))
    # print(pywt.Wavelet('db38'))
    cA, cD = pywt.dwt(data=signal, wavelet='db38', mode=dwt_mode)
    cD = np.power(cD, 2)
    return np.mean(cD)


def compute_fft(signal): # not included at the moment
    fft = np.fft.fft(signal)
    return fft


def extract_template_features(templates):
    template_features   = []
    template_headers    = []

    r_peak_amplitudes   = []
    q_index             = []
    q_amplitude         = []
    p_index             = []
    p_amplitude         = []
    s_index             = []
    s_amplitude         = []
    t_index             = []
    t_amplitude         = []
    p_start             = []
    p_end               = []
    q_start             = []
    s_end               = []
    t_start             = []
    t_end               = []
    

    # template_headers.extend

    for template in templates:

        # get r_peak amplitude
        r_peak_amplitudes.append(template[60])

        # get q_index and amplitude
        q_index_value = np.argmin(template[30:60]) + 30
        q_index.append(q_index_value)
        q_amplitude.append(template[q_index_value])

        # get p_index and amplitude
        p_index_value = np.argmax(template[:q_index_value])
        p_index.append(p_index_value)
        p_amplitude.append(template[p_index_value])

        # get s_index and amplitude
        s_index_value = np.argmin(template[61:90]) + 61
        s_index.append(s_index_value)
        s_amplitude.append(template[s_index_value])

        # get t_index and amplitude
        t_index_value = np.argmax(template[s_index_value:120]) + s_index_value
        t_index.append(t_index_value)
        t_amplitude.append(template[t_index_value])
        
        # get p_start
        if p_amplitude[-1] > 0:
            zero_crossings = np.where(np.diff(np.sign(template[:p_index_value + 1])))[0]
            if len(zero_crossings)>0:
                p_start.append(zero_crossings[-1])
            else:
                p_start.append(np.argmin(template[:p_index_value + 1]))
        else:
            p_start.append(np.argmin(template[:p_index_value + 1]))

        # get p_end and q_start
        if ((p_amplitude[-1] > 0) and (q_amplitude[-1] <= 0)) or ((p_amplitude[-1] >= 0) and (q_amplitude[-1] < 0)):
            zero_crossings = np.where(np.diff(np.sign(template[p_index_value:q_index_value + 1])))[0] + p_index_value
            p_end.append(zero_crossings[0])
            if len(zero_crossings)>1:
                q_start.append(zero_crossings[-1])
            else:
                q_start.append(zero_crossings[0])
        else:
            middle_crossing = p_index_value + round((q_index_value - p_index_value)/2)
            p_end.append(middle_crossing)
            q_start.append(middle_crossing)
        
        # get s_end and t_start
        if ((t_amplitude[-1] > 0) and (s_amplitude[-1] <= 0)) or ((t_amplitude[-1] >= 0) and (s_amplitude[-1] < 0)):
            zero_crossings = np.where(np.diff(np.sign(template[s_index_value:t_index_value + 1])))[0] + s_index_value
            s_end.append(zero_crossings[0])
            if len(zero_crossings)>1:
                t_start.append(zero_crossings[-1])
            else:
                t_start.append(zero_crossings[0])
        else:
            middle_crossing = s_index_value + round((s_index_value - t_index_value)/2)
            s_end.append(middle_crossing)
            t_start.append(middle_crossing)
        # get t_end
        if t_amplitude[-1] > 0:
            zero_crossings = np.where(np.diff(np.sign(template[t_index_value:])))[0] + t_index_value
            if len(zero_crossings)>0:
                t_end.append(zero_crossings[0])
            else:
                t_end.append(np.argmin(template[t_index_value:]) + t_index_value)
        else:
            t_end.append(np.argmin(template[t_index_value:]) + t_index_value)

    # convert list to array
    p_start             = np.array(p_start )
    p_end               = np.array(p_end   )
    q_start             = np.array(q_start )
    s_end               = np.array(s_end   )
    t_start             = np.array(t_start )
    t_end               = np.array(t_end   )

    # compute segments and intervals
    pr_interval         = q_start   - p_start
    pr_segment          = q_start   - p_end
    qrs_complex         = s_end     - q_start
    st_segment          = t_start   - s_end
    qt_interval         = t_end     - q_start
    
    # get r-peak_features
    r_peak_min = min(r_peak_amplitudes)
    r_peak_max = max(r_peak_amplitudes)
    r_peak_mean = np.mean(r_peak_amplitudes)
    r_peak_median = np.median(r_peak_amplitudes)
    r_peak_stdev = np.std(r_peak_amplitudes)
    template_features.extend([r_peak_min, r_peak_max, r_peak_mean, r_peak_median, r_peak_stdev])
    template_headers.extend(['r_peak_min', 'r_peak_max', 'r_peak_mean', 'r_peak_median', 'r_peak_stdev'])

    # get q-ampl_features
    q_ampl_min = min(q_amplitude)
    q_ampl_max = max(q_amplitude)
    q_ampl_mean = np.mean(q_amplitude)
    q_ampl_median = np.median(q_amplitude)
    q_ampl_stdev = np.std(q_amplitude)
    template_features.extend([q_ampl_min, q_ampl_max, q_ampl_mean, q_ampl_median, q_ampl_stdev])
    template_headers.extend(['q_ampl_min', 'q_ampl_max', 'q_ampl_mean', 'q_ampl_median', 'q_ampl_stdev'])

    # get q-ind_features
    q_ind_min = min(q_index)
    q_ind_max = max(q_index)
    q_ind_mean = np.mean(q_index)
    q_ind_median = np.median(q_index)
    q_ind_stdev = np.std(q_index)
    template_features.extend([q_ind_min, q_ind_max, q_ind_mean, q_ind_median, q_ind_stdev])
    template_headers.extend(['q_ind_min', 'q_ind_max', 'q_ind_mean', 'q_ind_median', 'q_ind_stdev'])

    # get p-ampl_features
    p_ampl_min = min(p_amplitude)
    p_ampl_max = max(p_amplitude)
    p_ampl_mean = np.mean(p_amplitude)
    p_ampl_median = np.median(p_amplitude)
    p_ampl_stdev = np.std(p_amplitude)
    template_features.extend([p_ampl_min, p_ampl_max, p_ampl_mean, p_ampl_median, p_ampl_stdev])
    template_headers.extend(['p_ampl_min', 'p_ampl_max', 'p_ampl_mean', 'p_ampl_median', 'p_ampl_stdev'])

    # get p-ind_features
    p_ind_min = min(p_index)
    p_ind_max = max(p_index)
    p_ind_mean = np.mean(p_index)
    p_ind_median = np.median(p_index)
    p_ind_stdev = np.std(p_index)
    template_features.extend([p_ind_min, p_ind_max, p_ind_mean, p_ind_median, p_ind_stdev])
    template_headers.extend(['p_ind_min', 'p_ind_max', 'p_ind_mean', 'p_ind_median', 'p_ind_stdev'])

    # get s-ampl_features
    s_ampl_min = min(s_amplitude)
    s_ampl_max = max(s_amplitude)
    s_ampl_mean = np.mean(s_amplitude)
    s_ampl_median = np.median(s_amplitude)
    s_ampl_stdev = np.std(s_amplitude)
    template_features.extend([s_ampl_min, s_ampl_max, s_ampl_mean, s_ampl_median, s_ampl_stdev])
    template_headers.extend(['s_ampl_min', 's_ampl_max', 's_ampl_mean', 's_ampl_median', 's_ampl_stdev'])

    # get s-ind_features
    s_ind_min = min(s_index)
    s_ind_max = max(s_index)
    s_ind_mean = np.mean(s_index)
    s_ind_median = np.median(s_index)
    s_ind_stdev = np.std(s_index)
    template_features.extend([s_ind_min, s_ind_max, s_ind_mean, s_ind_median, s_ind_stdev])
    template_headers.extend(['s_ind_min', 's_ind_max', 's_ind_mean', 's_ind_median', 's_ind_stdev'])

    # get t-ampl_features
    t_ampl_min = min(t_amplitude)
    t_ampl_max = max(t_amplitude)
    t_ampl_mean = np.mean(t_amplitude)
    t_ampl_median = np.median(t_amplitude)
    t_ampl_stdev = np.std(t_amplitude)
    template_features.extend([t_ampl_min, t_ampl_max, t_ampl_mean, t_ampl_median, t_ampl_stdev])
    template_headers.extend(['t_ampl_min', 't_ampl_max', 't_ampl_mean', 't_ampl_median', 't_ampl_stdev'])

    # get t-ind_features
    t_ind_min = min(t_index)
    t_ind_max = max(t_index)
    t_ind_mean = np.mean(t_index)
    t_ind_median = np.median(t_index)
    t_ind_stdev = np.std(t_index)
    template_features.extend([t_ind_min, t_ind_max, t_ind_mean, t_ind_median, t_ind_stdev])
    template_headers.extend(['t_ind_min', 't_ind_max', 't_ind_mean', 't_ind_median', 't_ind_stdev'])

    # get p_start features
    p_start_min = min(p_start)
    p_start_max = max(p_start)
    p_start_mean = np.mean(p_start)
    p_start_median = np.median(p_start)
    p_start_stdev = np.std(p_start)
    template_features.extend([p_start_min, p_start_max, p_start_mean, p_start_median, p_start_stdev])
    template_headers.extend(['p_start_min', 'p_start_max', 'p_start_mean', 'p_start_median', 'p_start_stdev'])

    # get p_end features
    p_end_min = min(p_end)
    p_end_max = max(p_end)
    p_end_mean = np.mean(p_end)
    p_end_median = np.median(p_end)
    p_end_stdev = np.std(p_end)
    template_features.extend([p_end_min, p_end_max, p_end_mean, p_end_median, p_end_stdev])
    template_headers.extend(['p_end_min', 'p_end_max', 'p_end_mean', 'p_end_median', 'p_end_stdev'])

    # get q_start features
    q_start_min = min(q_start)
    q_start_max = max(q_start)
    q_start_mean = np.mean(q_start)
    q_start_median = np.median(q_start)
    q_start_stdev = np.std(q_start)
    template_features.extend([q_start_min, q_start_max, q_start_mean, q_start_median, q_start_stdev])
    template_headers.extend(['q_start_min', 'q_start_max', 'q_start_mean', 'q_start_median', 'q_start_stdev'])

    # get s_end features
    s_end_min = min(s_end)
    s_end_max = max(s_end)
    s_end_mean = np.mean(s_end)
    s_end_median = np.median(s_end)
    s_end_stdev = np.std(s_end)
    template_features.extend([s_end_min, s_end_max, s_end_mean, s_end_median, s_end_stdev])
    template_headers.extend(['s_end_min', 's_end_max', 's_end_mean', 's_end_median', 's_end_stdev'])

    # get t_start features
    t_start_min = min(t_start)
    t_start_max = max(t_start)
    t_start_mean = np.mean(t_start)
    t_start_median = np.median(t_start)
    t_start_stdev = np.std(t_start)
    template_features.extend([t_start_min, t_start_max, t_start_mean, t_start_median, t_start_stdev])
    template_headers.extend(['t_start_min', 't_start_max', 't_start_mean', 't_start_median', 't_start_stdev'])

    # get t_end features
    t_end_min = min(t_end)
    t_end_max = max(t_end)
    t_end_mean = np.mean(t_end)
    t_end_median = np.median(t_end)
    t_end_stdev = np.std(t_end)
    template_features.extend([t_end_min, t_end_max, t_end_mean, t_end_median, t_end_stdev])
    template_headers.extend(['t_end_min', 't_end_max', 't_end_mean', 't_end_median', 't_end_stdev'])

    # get pr_interval features
    pr_interval_min = min(pr_interval)
    pr_interval_max = max(pr_interval)
    pr_interval_mean = np.mean(pr_interval)
    pr_interval_median = np.median(pr_interval)
    pr_interval_stdev = np.std(pr_interval)
    template_features.extend([pr_interval_min, pr_interval_max, pr_interval_mean, pr_interval_median, pr_interval_stdev])
    template_headers.extend(['pr_interval_min', 'pr_interval_max', 'pr_interval_mean', 'pr_interval_median', 'pr_interval_stdev'])

    # get pr_segment features
    pr_segment_min = min(pr_segment)
    pr_segment_max = max(pr_segment)
    pr_segment_mean = np.mean(pr_segment)
    pr_segment_median = np.median(pr_segment)
    pr_segment_stdev = np.std(pr_segment)
    template_features.extend([pr_segment_min, pr_segment_max, pr_segment_mean, pr_segment_median, pr_segment_stdev])
    template_headers.extend(['pr_segment_min', 'pr_segment_max', 'pr_segment_mean', 'pr_segment_median', 'pr_segment_stdev'])

    # get qrs_complex features
    qrs_complex_min = min(qrs_complex)
    qrs_complex_max = max(qrs_complex)
    qrs_complex_mean = np.mean(qrs_complex)
    qrs_complex_median = np.median(qrs_complex)
    qrs_complex_stdev = np.std(qrs_complex)
    template_features.extend([qrs_complex_min, qrs_complex_max, qrs_complex_mean, qrs_complex_median, qrs_complex_stdev])
    template_headers.extend(['qrs_complex_min', 'qrs_complex_max', 'qrs_complex_mean', 'qrs_complex_median', 'qrs_complex_stdev'])

    # get st_segment features
    st_segment_min = min(st_segment)
    st_segment_max = max(st_segment)
    st_segment_mean = np.mean(st_segment)
    st_segment_median = np.median(st_segment)
    st_segment_stdev = np.std(st_segment)
    template_features.extend([st_segment_min, st_segment_max, st_segment_mean, st_segment_median, st_segment_stdev])
    template_headers.extend(['st_segment_min', 'st_segment_max', 'st_segment_mean', 'st_segment_median', 'st_segment_stdev'])

    # get qt_interval features
    qt_interval_min = min(qt_interval)
    qt_interval_max = max(qt_interval)
    qt_interval_mean = np.mean(qt_interval)
    qt_interval_median = np.median(qt_interval)
    qt_interval_stdev = np.std(qt_interval)
    template_features.extend([qt_interval_min, qt_interval_max, qt_interval_mean, qt_interval_median, qt_interval_stdev])
    template_headers.extend(['qt_interval_min', 'qt_interval_max', 'qt_interval_mean', 'qt_interval_median', 'qt_interval_stdev'])
    
     

    return template_features, template_headers

def extract_mean_template_features(template):
    template_features   = []
    template_headers    = []

    # get r_peak amplitude
    r_peak_amplitude = template[60]

    # get q_index and amplitude
    q_index = np.argmin(template[30:60]) + 30
    q_amplitude = template[q_index]

    # get p_index and amplitude
    p_index = np.argmax(template[:q_index])
    p_amplitude = template[p_index]

    # get s_index and amplitude
    s_index = np.argmin(template[61:90]) + 61
    s_amplitude = template[s_index]

    # get t_index and amplitude
    t_index = np.argmax(template[s_index:120]) + s_index
    t_amplitude = template[t_index]
    
    # get p_start
    if p_amplitude > 0:
        zero_crossings = np.where(np.diff(np.sign(template[:p_index + 1])))[0]
        if len(zero_crossings)>0:
            p_start = zero_crossings[-1]
        else:
            p_start = np.argmin(template[:p_index + 1])
    else:
        p_start = np.argmin(template[:p_index + 1])

    # get p_end and q_start
    if ((p_amplitude > 0) and (q_amplitude <= 0)) or ((p_amplitude >= 0) and (q_amplitude < 0)):
        zero_crossings = np.where(np.diff(np.sign(template[p_index:q_index + 1])))[0] + p_index
        p_end = zero_crossings[0]
        if len(zero_crossings)>1:
            q_start = zero_crossings[-1]
        else:
            q_start = zero_crossings[0]
    else:
        middle_crossing = p_index + round((q_index - p_index)/2)
        p_end = middle_crossing
        q_start = middle_crossing
    
    # get s_end and t_start
    if ((t_amplitude > 0) and (s_amplitude <= 0)) or ((t_amplitude >= 0) and (s_amplitude < 0)):
        zero_crossings = np.where(np.diff(np.sign(template[s_index:t_index + 1])))[0] + s_index
        s_end = zero_crossings[0]
        if len(zero_crossings)>1:
            t_start = zero_crossings[-1]
        else:
            t_start = zero_crossings[0]
    else:
        middle_crossing = s_index + round((s_index - t_index)/2)
        s_end = middle_crossing
        t_start = middle_crossing
    # get t_end
    if t_amplitude > 0:
        zero_crossings = np.where(np.diff(np.sign(template[t_index:])))[0] + t_index
        if len(zero_crossings)>0:
            t_end = zero_crossings[0]
        else:
            t_end = np.argmin(template[t_index:]) + t_index
    else:
        t_end = np.argmin(template[t_index:]) + t_index
    

    # compute segments and intervals
    pr_interval         = q_start   - p_start
    pr_segment          = q_start   - p_end
    qrs_complex         = s_end     - q_start
    st_segment          = t_start   - s_end
    qt_interval         = t_end     - q_start

    template_features.extend([r_peak_amplitude, q_index, q_amplitude, p_index, p_amplitude, s_index, s_amplitude, t_index, t_amplitude, p_start, p_end, q_start, s_end, t_start, t_end, pr_interval, pr_segment, qrs_complex, st_segment, qt_interval])
    template_headers.extend(["r_peak_amplitude", "q_index", "q_amplitude", "p_index", "p_amplitude", "s_index", "s_amplitude", "t_index", "t_amplitude", "p_start", "p_end", "q_start", "s_end", "t_start", "t_end", "pr_interval", "pr_segment", "qrs_complex", "st_segment", "qt_interval"])
    
    # plt.plot(template)
    # plt.scatter([p_index, q_index, 60, s_index, t_index], [p_amplitude, q_amplitude, r_peak_amplitude, s_amplitude, t_amplitude])
    # plt.show()

    return template_features, template_headers

def compute_features(signal):
    raw = signal.dropna()
    # raw = signal[~np.isnan(signal)]

    # mean, std-dev, varianz, min, max
    # i.a. for wavelets
    # time differences between heartbeats (rpeaks) -> statistics

    out = ecg.ecg(signal=raw, sampling_rate=Fs, show=False)
    filtered = out['filtered']
    templates = out['templates']
    r_peak_idices = out['rpeaks']
    heart_rate = out['heart_rate']

    features = []
    headers = []

    # entropy
    entropy_raw = compute_entropy(raw)
    entropy_filtered = compute_entropy(filtered)
    features.append(entropy_raw)
    features.append(entropy_filtered)
    headers.append('entropy_raw')
    headers.append('entropy_filtered')

    # wavelet_energy
    # energy_raw = compute_wavelet_energy(raw)
    # energy_filtered = compute_wavelet_energy(filtered)
    # features.append(energy_raw)
    # features.append(energy_filtered)
    # headers.append('energy_raw')
    # headers.append('energy_filtered')

    # fft
    # print("FFT RAW TYPE: ", type(raw))
    # fft_raw = compute_fft(raw)
    # fft_filtered = compute_fft(filtered)
    # print(len(fft_raw))
    # print(len(fft_filtered))
    # plt.subplot(2, 1, 1)
    # plt.plot(fft_raw)
    # plt.subplot(2, 1, 2)
    # plt.plot(fft_filtered)
    # plt.show()

    # r peaks ratio
    r_peak_ratio = len(r_peak_idices)/len(filtered)
    features.append(r_peak_ratio)
    headers.append('r_peak_ratio')

    # r peak difference
    if len(r_peak_idices)>1:
        difference = r_peak_idices[1:] - r_peak_idices[:-1]
        r_peak_ind_diff_min = min(difference)
        r_peak_ind_diff_max = max(difference)
        r_peak_ind_diff_mean = np.mean(difference)
        r_peak_ind_diff_median = np.median(difference)
        r_peak_ind_diff_stdev = np.std(difference)
    else:
        r_peak_ind_diff_min = 0
        r_peak_ind_diff_max = 0
        r_peak_ind_diff_mean = 0
        r_peak_ind_diff_median = 0
        r_peak_ind_diff_stdev = 0
    
    features.extend([r_peak_ind_diff_min, r_peak_ind_diff_max, r_peak_ind_diff_mean, r_peak_ind_diff_median, r_peak_ind_diff_stdev])
    headers.extend(['r_peak_ind_diff_min', 'r_peak_ind_diff_max', 'r_peak_ind_diff_mean', 'r_peak_ind_diff_median', 'r_peak_ind_diff_stdev'])

    #find meanheartbeat and stdev to mean hearbeat
    templates_mean  = np.mean(templates, axis=0)
    templates_stdev = np.std(templates.tolist(), axis=0)

    suffix_mean     = "_mean"
    suffix_stdev    = "_stdev"
    headers_mean    = [str(i) + suffix_mean for i in range(180)]
    headers_stdev   = [str(i) + suffix_stdev for i in range(180)]
    
    features.extend(templates_mean)
    features.extend(templates_stdev)
    headers.extend(headers_mean)
    headers.extend(headers_stdev)

    # find features from mean hearbeat
    mean_template_features, mean_template_headers = extract_mean_template_features(templates_mean)
    features.extend(mean_template_features)
    headers.extend(mean_template_headers)
    
    # template features
    template_features, template_headers = extract_template_features(templates)
    features.extend(template_features)
    headers.extend(template_headers)

    # hearbeat features
    # get heart_rate_features
    if(len(heart_rate)>0):
        heart_rate_min = min(heart_rate)
        heart_rate_max = max(heart_rate)
        heart_rate_mean = np.mean(heart_rate)
        heart_rate_median = np.median(heart_rate)
        heart_rate_stdev = np.std(heart_rate)
    else:
        heart_rate_min = 0
        heart_rate_max = 0
        heart_rate_mean = 0
        heart_rate_median = 0
        heart_rate_stdev = 0
    features.extend([heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_median, heart_rate_stdev])
    headers.extend(['heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'heart_rate_median', 'heart_rate_stdev'])
    # print(list(zip(headers, features)))
    return features, headers



# read data
df_test     = pd.read_csv('../../data/X_test.csv', sep=',')
df_X_train  = pd.read_csv('../../data/X_train.csv', sep=',')
df_y_train  = pd.read_csv('../../data/y_train.csv', sep=',')

# sample_ids  = df_sample["id"]
train_set_x = df_X_train.drop(columns = ['id'])
train_set_y = df_y_train.drop(columns = ['id'])
test_set_x  = df_test.drop(columns = ['id'])

choice = "test"
if choice == "train":
    data_x = train_set_x
    path_suffix = "X_train_new_2.csv"
else:
    data_x = test_set_x
    path_suffix = "X_test_new_2.csv"

Fs = 300
N = data_x.shape[1]
T = (N -1) / Fs
ts = np.linspace(0, T, N, endpoint=False)


signal = data_x.iloc[0]
features, headers = compute_features(signal)
header = ['id']
header.extend(headers)
X_train_new = pd.DataFrame( columns=header)


df_len = len(data_x)
length = 0
print("indexes that didnt work")
for i in range(df_len):
    try:
        signal = data_x.iloc[i]
        feature = [i]
        features, _ = compute_features(signal)
        feature.extend(features)
        # print(X_train_new.shape)
        # print(len(feature))
        X_train_new.loc[length]=feature
        length += 1
    except:
        print(i)
        print("try again with last value removed")
        try:
            signal = data_x.iloc[i][:-1]
            feature = [i]
            features, _ = compute_features(signal)
            feature.extend(features)
            # print(X_train_new.shape)
            # print(len(feature))
            X_train_new.loc[length]=feature
            length += 1
            print("worked")
        except:
            print("still doesnt work, skipped this index")

# print(X_train_new.shape)
# print(X_train_new.columns.values)

X_train_new.to_csv('../../data/' + path_suffix, index=False) # X_train_new.csv'