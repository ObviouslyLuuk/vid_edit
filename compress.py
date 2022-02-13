import scipy.io.wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt


# Extract audio from video file, save as wav audio file
# INPUT: Video file
# OUTPUT: Does not return any values, but saves audio as wav file
def extract_audio(dir, video_name):
    track_name = video_name.split(".")[0]
    output_path = dir + track_name + "_WAV.wav"
    
    # https://stackoverflow.com/questions/9913032/how-can-i-extract-audio-from-video-with-ffmpeg
    # command = f"ffmpeg -i {video_name} -vn -acodec copy {output_path}"
    # -y    : auto overwrite
    # -ar   : output sample rate
    command = f"ffmpeg -y -i {dir+video_name} -ar 44100 {output_path}"
    os.system(command)
    return output_path


# Read file
# INPUT: Audio file
# OUTPUT: Sets sample rate of wav file, Returns data read from wav file (numpy array of integers)
def read_audio(audio_file, duration=60):
    rate, data = scipy.io.wavfile.read(audio_file)  # Return the sample rate (in samples/sec) and data from a WAV file
    # Only take the left channel for convenience
    if isinstance(data[0], np.ndarray):
        data = data[:, 1]
    if len(data) > duration * rate:
        print("length:", len(data)/rate)
        data = data[:duration * rate]
        print("rate:", rate)
    return data, rate


def compress(raw_channel, threshold, ratio, attack, release, rate):
    samples_p_millisecond = rate * 1000
    attack_samples = attack*samples_p_millisecond
    release_samples = release*samples_p_millisecond
    attack_counter = 0
    release_counter = 0
    active_ratio = attack_counter / attack_samples

    compressed_channel = []
    for sample in raw_channel:
        if sample > threshold:
            sample = threshold + (sample - threshold) / ratio
        else:
            compressed_channel.append(sample)

    return compressed_channel



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

    return time_delay


# Find time delay between two video files
def align(video1, video2, dir, time_bin_size=int(44100/5), max_duration=60, plotting=False):
    audio1 = extract_audio(dir, video1)
    audio2 = extract_audio(dir, video2)

    raw_audio1, rate = read_audio(audio1, max_duration)
    raw_audio2, _ = read_audio(audio2, max_duration)

    os.remove(audio1)
    os.remove(audio2)

    peaks1 = find_peaks(raw_audio1, time_bin_size)
    peaks2 = find_peaks(raw_audio2, time_bin_size)
    if plotting:
        plt.show()

    pairs = find_peak_pairs(peaks1, peaks2)
    time_delay = find_delay_by_peaks(pairs, 1/100, rate)
    if plotting:
        plt.show()

    seconds = round(time_delay, 4)

    if seconds > 0:
        return (seconds, 0)
    else:
        return (0, abs(seconds))


def offset_audio(audio, offset):
    seconds = math.floor(offset % 60)
    minutes = math.floor(offset / 60 % 60)
    hours   = math.floor(offset / 60 / 60)
    decimals = str((round(offset,3) % 1)).split('.')[-1]
    track_name = audio.split(".")[0]
    offset_str = f"{hours:02}:{minutes:02}:{seconds:02}.{decimals}"
    offset_audio = track_name + "_TEMP.wav"

    command = f"ffmpeg -y -i {dir+audio} -ss {offset_str} {dir+offset_audio}"
    os.system(command)

    return offset_audio


def add_a_to_v(video, audio, dir, offset=0, swap_channels=True):
    if offset != 0:
        audio = offset_audio(audio, offset)

    track_name = video.split(".")[0]
    output_path = dir + track_name + "_COMB.mp4"
    swap = ''
    if swap_channels:
        # https://trac.ffmpeg.org/wiki/AudioChannelManipulation#Switchstereochannels
        swap = '-af "pan=stereo|c0=c1|c1=c0"'

    # https://superuser.com/questions/1137612/ffmpeg-replace-audio-in-video
    # -y        : auto overwrite
    # -c:v copy : don't reencode video
    # -map      : video and audio stream mapping to output
    # -shortest : end output when the first input file ends
    command = f"ffmpeg -y -i {dir+video} -i {dir+audio} -c:v copy -map 0:v:0 -map 1:a:0 {swap} -shortest {output_path}"
    os.system(command)

    if offset != 0:
        os.remove(dir+audio)


# ======= TEST FILES ==============
dir = "../../../stuff/ai_videoedit/"
video1 = "VID_20210607_121613.mp4"
audio2 = "ASMR_Masters_1.wav"

t = align(video1, audio2, dir)
print(t)

add_a_to_v(video1, audio2, dir, t[0])