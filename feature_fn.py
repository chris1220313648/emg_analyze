import numpy as np
import math
def mf(records):
    # 计算FFT和幅度谱
    fft = np.fft.fft(records)
    magnitude = np.abs(fft)

    # 对幅度谱进行排序，并记录每个幅度值对应的频率
    sort_idx = np.argsort(magnitude)
    sorted_mag = magnitude[sort_idx]
    freqs = np.fft.fftfreq(len(records), d=1/2148)[sort_idx]

    # 计算幅度谱的累积和，归一化到总和的一半
    cumsum_mag = np.cumsum(sorted_mag)
    normalized_cumsum_mag = cumsum_mag / np.sum(sorted_mag)
    median_mag = np.interp(0.5, normalized_cumsum_mag, sorted_mag)

    # 找到幅度等于中位数的那个频率
    median_freq = np.interp(median_mag, sorted_mag, freqs)
    return median_freq


def mpf(records):
    # 计算FFT
    fft = np.fft.fft(records)

    # 计算幅度谱和相位谱
    magnitude = np.abs(fft)

    # 计算信号的总功率
    total_power = np.sum(magnitude**2)

    # 计算每个频率的功率贡献
    power_contribution = magnitude**2 / total_power

    # 计算平均功率频率
    mean_power_freq = np.sum(power_contribution * np.fft.fftfreq(len(records))) * np.pi
    return mean_power_freq


# def ssc(records):
#     length = len(records)
#     sum = 0
#
#     for i in range(1, length-1):
#         sum = 0
#         for i in range(1, len(records)-1):
#             temp = (records[i] - records[i-1]) * (records[i] - records[i-1])
#             if temp > 50:
#                 sum = sum + 1
#
#     data = sum
#     return data

def ssc(records):
    length = len(records)
    count = 0

    for i in range(1, length-1):
        temp = (records[i] - records[i-1]) * (records[i] - records[i+1])
        if temp > 2.5e-12:
            count += 1

    return count

def waveLength(records):
    length = len(records)
    sum = 0
    for i in range(length - 1):
        sum = sum + abs(records[i + 1] - records[i])
    data = sum
    return data



def Mav(records):
    length = len(records)
    sum = 0
    count = 0
    for i in range(length):
        count = count + 1
        sum = sum + abs(records[i])

    value = sum / count
    data = value

    return data


def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def sampEn(L: np.array, std: float, m: int = 2, r: float = 0.15):
    """
    计算时间序列的样本熵

    Input:
        L: 时间序列
        std: 原始序列的标准差
        m: 1或2
        r: 阈值

    Output:
        SampEn
    """
    N = len(L)
    B = 0.0
    A = 0.0

    # Split time series and save all templates of length m
    xmi = np.array([L[i:i + m] for i in range(N - m)])
    xmj = np.array([L[i:i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r * std) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([L[i:i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r * std) - 1 for xmi in xm])
    # Return SampEn

    if A > 0 and B > 0:
        return -np.log(A / B)
    else:
        # 当没有匹配时，返回一个高值而不是无穷大，反映高度不可预测性
        return np.max([N, 10])  # 可以根据具体情况选择一个适当的高值