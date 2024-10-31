#
# credit: https://github.com/allisonnicoledeal/VideoSync/blob/master/alignment_by_row_channels.py
#

from sys import int_info
from numpy.core.numeric import Infinity
import scipy.io.wavfile
import numpy as np
import math
import os
import matplotlib.pyplot as plt


# Extract audio from video file, save as wav auido file
# INPUT: Video file
# OUTPUT: Does not return any values, but saves audio as wav file
def extract_audio(dir, video_name):
    track_name = video_name.split(".")
    output_name = track_name[0] + "WAV.wav"  # !! CHECK TO SEE IF FILE IS IN UPLOADS DIRECTORY
    output_path = dir + output_name
    
    command = f"ffmpeg -i {video_name} -vn -acodec copy {output_path}"
    command = f"ffmpeg -i {video_name} -ar 44100 {output_path}"
    os.system(command)
    return output_path


# Read file
# INPUT: Audio file
# OUTPUT: Sets sample rate of wav file, Returns data read from wav file (numpy array of integers)
def read_audio(audio_file, duration=60):
    rate, data = scipy.io.wavfile.read(audio_file)  # Return the sample rate (in samples/sec) and data from a WAV file
    # Only take the left channel for convenience
    if type(data[0]) == type(np.array([])):
        data = data[:, 1]
    if len(data) > duration * rate:
        print("length:", len(data)/rate)
        data = data[:duration * rate]
        print("rate:", rate)
    return data, rate


def find_peaks(raw_audio, time_bin_size=int(44100/5), threshold=0):
    peaks = []
    highest_peak = 0
    for i in range(0, len(raw_audio), time_bin_size):
        peak = 0
        peak_i = None
        sample_i = i
        for sample in raw_audio[i:i+time_bin_size]:
            if sample > peak:
                peak = sample
                peak_i = sample_i
            sample_i += 1
        if peak > threshold:
            highest_peak = max(highest_peak, peak)
            peaks.append({
                "sample_i": peak_i,
                "peak": peak,
            })
    
    highest_peaks = []
    for dic in peaks:
        if dic["peak"] > highest_peak / 3:
            highest_peaks.append(dic)

    peaks = highest_peaks
    plt.plot(range(len(raw_audio)), raw_audio)
    plt.plot([peak["sample_i"] for peak in peaks], [peak["peak"] for peak in peaks], "o")
    # plt.vlines(range(0, len(raw_audio), time_bin_size), -50000, 50000)

    return peaks


def find_peak_pairs(peaks1, peaks2):
    pairs = []
    for peak1 in peaks1:
        time1 = peak1["sample_i"]
        for peak2 in peaks2:
            time2 = peak2["sample_i"]
            pairs.append((time1, time2))
    return pairs

def find_delay_by_peaks(pairs, leeway=1/100, rate=44100):
    t_diffs = {}
    for t_diff in [time[1] - time[0] for time in pairs]:
        t_diff = round(t_diff / rate, 3)

        if t_diff in t_diffs:
            t_diffs[t_diff] += 1
        else:
            t_diffs[t_diff] = 1

    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1], reverse=True)

    print (t_diffs_sorted)

    t_diffs = {}
    for t_diff, score in t_diffs_sorted:
        added = False
        for higher_score_t_diff in t_diffs:
            if abs(higher_score_t_diff - t_diff) < leeway/2:
                t_diffs[higher_score_t_diff] += score
                added = True
        if not added:
            t_diffs[t_diff] = score

    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1], reverse=True)
    t_diffs_sorted = list(filter(lambda i: i[1]>1, t_diffs_sorted))
    print (t_diffs_sorted)
    time_delay = t_diffs_sorted[0][0]

    plt.plot([i[0] for i in t_diffs_sorted], [i[1] for i in t_diffs_sorted], "o")
    plt.show()

    return time_delay


def make_time_bins(raw_audio, time_bin_size=1024, overlap=0, box_height=512):
    time_bins = {}
    # process bins
    time_bin = 0  # starting at first bin, with x index 0
    for i in range(0, len(raw_audio), time_bin_size-overlap):
        bin_data = raw_audio[i:i + time_bin_size] # get data for time bin
        if (len(bin_data) == time_bin_size): # if there are enough audio samples left to create a full time bin
            intensities = fourier(bin_data) # intensities is list of fft results (array of intensities by frequency)
            freq_bin_count = math.floor(len(intensities)/box_height)+1
            # print(intensities)
            # print(len(intensities))
            # plt.plot(range(len(intensities)), intensities)  # Plot some data on the axes.
            # plt.show()
            for freq, intensity in enumerate(intensities):
                freq_bin = freq/box_height

                if freq_bin > freq_bin_count: # prevent incomplete bins
                    break

                freq_bin = math.floor(freq_bin)
                if freq_bin in time_bins:
                    time_bins[freq_bin].append((intensity, time_bin, freq))
                else:
                    time_bins[freq_bin] = [(intensity, time_bin, freq)]
        time_bin += 1

    return time_bins


# Compute the one-dimensional discrete Fourier Transform
# INPUT: list with length of number of samples per second
# OUTPUT: list of real values len of num samples per second
def fourier(raw_audio):  #, overlap):
    mag = []
    fft_data = np.fft.rfft(raw_audio, 5000)  # Returns real values
    # for i in range(math.floor(len(fft_data))):
    #     r = fft_data[i].real**2
    #     j = fft_data[i].imag**2
    #     mag.append(round(math.sqrt(r+j),2))
    # return mag

    fft_data = fft_data[100:]

    # print(len(fft_data)/512)

    return fft_data


def make_vert_bins(time_bins, box_width=43):
    boxes = {}
    for freq_bin in time_bins:
        for tup in time_bins[freq_bin]:
            _, time_bin, _ = tup
            box_x = time_bin / box_width
            if (box_x,freq_bin) in boxes:
                boxes[(box_x,freq_bin)].append(tup)
            else:
                boxes[(box_x,freq_bin)] = [tup]

    return boxes


def find_bin_max(boxes, maxes_per_box):
    freqs_dict = {}
    for box in boxes.values():
        max_intensities = [(0,0,0)]
        for tup in box:
            intensity, time_bin, freq = tup
            if intensity > min(max_intensities)[0]:
                if len(max_intensities) < maxes_per_box:  # add if < number of points per box
                    max_intensities.append(tup)
                else:  # else add new number and remove min
                    max_intensities.append(tup)
                    max_intensities.remove(min(max_intensities))
        for tup in max_intensities:
            intensity, time_bin, freq = tup
            if freq in freqs_dict:
                freqs_dict[freq].append(time_bin)
            else:
                freqs_dict[freq] = [time_bin]

    return freqs_dict


def find_freq_pairs(freqs_dict1, freqs_dict2):
    time_pairs = []
    for freq in freqs_dict1:  # iterate through freqs in one
        if freq in freqs_dict2:  # if same freq has been a max in the other
            for time_bin1 in freqs_dict1[freq]:  # determine time offset
                for time_bin2 in freqs_dict2[freq]:
                    time_pairs.append((time_bin1, time_bin2))

    return time_pairs


def find_delay(time_pairs, time_bin_size, rate):
    t_diffs = {}
    for pair in time_pairs:
        delta_bin = pair[0] - pair[1]

        if delta_bin == 0:
            continue

        delta_t = delta_bin * time_bin_size / rate
        if delta_t in t_diffs:
            t_diffs[delta_t] += 1
        else:
            t_diffs[delta_t] = 1
    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
    print (t_diffs_sorted)
    time_delay = t_diffs_sorted[-1][0]

    t_diffs_sorted2 = sorted(t_diffs.items(), key=lambda x: x[0])
    plt.plot([i[0] for i in t_diffs_sorted2], [i[1] for i in t_diffs_sorted2])
    plt.show()

    return time_delay


def process_file(file_path, time_bin_size, overlap=0, box_height=512, box_width=43, maxes_per_box=7):
    # file_path = extract_audio(file_path)
    raw_audio, rate = read_audio(file_path)
    bins_dict = make_time_bins(raw_audio, time_bin_size, overlap, box_height) #bins, overlap, box height
    boxes = make_vert_bins(bins_dict, box_width)  # box width
    return find_bin_max(boxes, maxes_per_box), rate  # samples per box

# Find time delay between two video files
def align(video1, video2, time_bin_size=int(44100/10)):
    raw_audio1, rate = read_audio(video1, Infinity)
    raw_audio2, _ = read_audio(video2, Infinity)
    # plt.plot([i/rate for i in range(len(raw_audio1))], raw_audio1)
    # plt.plot([i/rate for i in range(len(raw_audio1))], [i*10 for i in raw_audio2])
    # plt.show()

    peaks1 = find_peaks(raw_audio1)
    peaks2 = find_peaks(raw_audio2)
    plt.show()

    pairs = find_peak_pairs(peaks1, peaks2)
    time_delay = find_delay_by_peaks(pairs, 1/100, rate)

    # print("Starting...")
    # ft_dict1, rate = process_file(video1, time_bin_size)

    # print("Second file...")
    # ft_dict2, _ = process_file(video2, time_bin_size)

    # print("Time delay...")

    # # Determine time delay
    # pairs = find_freq_pairs(ft_dict1, ft_dict2)
    # time_delay = find_delay(pairs, time_bin_size, rate)

    seconds = round(time_delay, 4)

    if seconds > 0:
        return (seconds, 0)
    else:
        return (0, abs(seconds))



# ======= TEST FILES ==============
# rate = 44100 Hz

dir = "../../../stuff/ai_videoedit/"
audio1 = dir+"WAV1.wav"
audio2 = dir+"WAV0.wav"

t = align(audio1, audio2)
print(t)