import os
import soundfile as sf
import numpy as np
import librosa

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from speech_enhance import SpeechEnhance
except ImportError:
    from ..speech_enhance import SpeechEnhance

RATE_16K = 16000


if __name__ == "__main__":
    speech_enhance = SpeechEnhance()

    # 读取音频文件
    file_path = input("选择需要resample的音频: ")
    file_name, file_format = os.path.splitext(file_path)
    audio_np, samplerate = sf.read(file_path)

    # 获取声道数
    if len(audio_np.shape) == 1:
        n_channels = 1
    else:
        n_channels = audio_np.shape[1]
        # 如果需要转换为单声道
        if n_channels > 1:
            audio_np = np.mean(audio_np, axis=1)  # 多声道转单声道

    # 进行语音增强
    audio_enhanced = speech_enhance.enhance(audio_np, samplerate)

    # 转换为 16kHz
    if samplerate != RATE_16K:
        audio_enhanced_16k = librosa.resample(audio_enhanced, orig_sr=samplerate, target_sr=RATE_16K)
    else:
        audio_enhanced_16k = audio_enhanced

    # 导出增强后的音频
    sf.write(file_name+'_16k.wav', audio_enhanced_16k, RATE_16K)
