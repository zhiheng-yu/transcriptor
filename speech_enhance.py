import warnings
import numpy as np
import pyloudnorm as pyln
import librosa
from clearvoice import ClearVoice

RATE_48K = 48000


class SpeechEnhance:
    def __init__(self, model_name="MossFormer2_SE_48K", target_lufs=-16.0, true_peak_limit=-1.0):
        self.myClearVoice = ClearVoice(task='speech_enhancement', model_names=[model_name])
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit

    def normalize_loudness_advanced(self, audio_np, samplerate):
        """
        高级响度归一化函数，带真峰值限制

        Args:
            audio_segment (AudioSegment): pydub 的 AudioSegment 对象
            target_lufs (float): 目标响度值 (单位：LUFS)
            true_peak_limit (float): 真峰值限制 (单位：dBTP)

        Returns:
            AudioSegment: 归一化后的音频数据 (pydub 的 AudioSegment 对象)
        """
        # pyloudnorm 需要至少 0.4 秒的音频（默认 block_size）
        min_length = int(samplerate * 0.4)

        # 如果音频太短，只进行简单的峰值归一化
        if len(audio_np) < min_length:
            peak_normalized = pyln.normalize.peak(audio_np, self.true_peak_limit)
            return np.nan_to_num(peak_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # 创建响度表并测量
        meter = pyln.Meter(samplerate)
        original_loudness = meter.integrated_loudness(audio_np)
        # print(f"原始响度: {original_loudness:.2f} LUFS")

        # 1. 进行响度归一化
        # 忽略 pyloudnorm 的削波警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Possible clipped samples in output.")
            normalized_audio = pyln.normalize.loudness(audio_np, original_loudness, self.target_lufs)

        # 2. 进行真峰值限制
        # 将真峰值限制从 dB 转换为线性值
        peak_normalized = pyln.normalize.peak(normalized_audio, self.true_peak_limit)

        # 再次测量最终响度
        # final_loudness = meter.integrated_loudness(peak_normalized)
        # print(f"目标响度: {self.target_lufs:.1f} LUFS")
        # print(f"最终响度: {final_loudness:.2f} LUFS")
        # print(f"真峰值限制: {self.true_peak_limit:.1f} dBTP")

        return np.nan_to_num(peak_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def clearvoice_enhance(self, audio_np):
        if len(audio_np.shape) < 2:
            audio_np = np.reshape(audio_np, [1, audio_np.shape[0]])
        audio_enhanced = self.myClearVoice(audio_np)[0,:]
        return np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

    def enhance(self, audio_np, samplerate):
        if samplerate != RATE_48K:
            audio_48k = librosa.resample(audio_np, orig_sr=samplerate, target_sr=RATE_48K)
        else:
            audio_48k = audio_np

        audio_48k = self.clearvoice_enhance(audio_48k)
        audio_48k = self.normalize_loudness_advanced(audio_48k, RATE_48K)

        if samplerate != RATE_48K:
            audio_np = librosa.resample(audio_48k, orig_sr=RATE_48K, target_sr=samplerate)
        else:
            audio_np = audio_48k

        return audio_np
