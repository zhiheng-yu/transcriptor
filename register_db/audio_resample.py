import os
from pydub import AudioSegment


# 读取音频文件
file_path = input("选择需要resample的音频: ")
file_name, file_format = os.path.splitext(file_path)
audio = AudioSegment.from_file(file_path)

# 设置采样率为 16kHz
audio = audio.set_frame_rate(16000)

# 设置音频数据为 int16 格式
audio = audio.set_sample_width(2)  # 2 字节 = 16 位
# 将双声道转换为单声道
if audio.channels == 2:  # 如果是双声道
    audio = audio.set_channels(1)  # 设置为单声道

audio = audio + 10

audio.export(file_name+'_16k.wav', format='wav')
