import os
import scipy
import numpy as np
from pydub import AudioSegment
from funasr import AutoModel

from config import Config
from speaker_recognize import SpeakerVerifier
from speech_enhance import SpeechEnhance


class Segment:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end


class Transcriptor:
    def __init__(self):
        self.samplerate = Config.samplerate
        self.epoch = 0
        self.load_models(Config.models)
        self.preheat(Config.preheat_audio)

    def load_models(self, models):
        asr_config = models["asr"]
        self.asr_model = AutoModel(
            model=asr_config["name"],
            vad_model=models["vad"]["name"],
            punc_model=models["punc"]["name"],
            vad_kwargs={"max_single_segment_time": Config.max_speech_duration * 1000},
            device=asr_config["device"],
            disable_update=True
        )

        self.speaker_verifier = SpeakerVerifier()

        se_config = Config.speech_enhance
        if se_config.get("enable"):
            self.speech_enhance = SpeechEnhance(
                model_name=se_config.get("model_name"),
                target_lufs=se_config.get("target_lufs"),
                true_peak_limit=se_config.get("true_peak_limit"),
                mute_if_too_quiet=se_config.get("mute_if_too_quiet"),
                threshold_dbfs=se_config.get("threshold_dbfs"),
            )
        else:
            self.speech_enhance = None

    def preheat(self, preheat_audio):
        self.asr_model.generate(
            input=preheat_audio,
            cache={},
            language="zh",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
            sentence_timestamp=True,
            disable_pbar=True
        )

    def dump(self, final, audio_buffer):
        dump_config = Config.dump
        save_mode = dump_config.get("audio_save")

        if save_mode not in ["all", "final"]:
            return

        if save_mode == "final" and not final:
            return

        audio_dir = dump_config.get("audio_dir")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        self.epoch += 1
        audio_path = os.path.join(audio_dir, f"{self.epoch:06d}.wav")
        scipy.io.wavfile.write(audio_path, rate=self.samplerate, data=audio_buffer)

    def parse_to_segments(self, funasr_res):
        segments = []
        if "sentence_info" in funasr_res:
            new_sentence = True
            sentence = ""
            for res in funasr_res["sentence_info"]:
                if new_sentence:
                    start = res["start"] / 1000
                    new_sentence = False
                sentence += res["text"]
                end = res["end"] / 1000
                if any(punct in sentence for punct in ["。", "？", "！", ".", "!", "?"]):
                    segments.append(Segment(sentence, start, end))
                    new_sentence = True
                    sentence = ""
            if sentence != "":
                segments.append(Segment(sentence, start, end))
        return segments

    def transcript(self, audio_buffer, last_speaker, last_sentence, last_transcript):
        res = self.asr_model.generate(
            input=audio_buffer,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        final = False
        speaker = last_speaker
        sentence = last_sentence
        transcript = last_transcript
        new_buffer = audio_buffer

        # 获取转录结果
        if not res or len(res) == 0:
            print("No result from asr model")
            return final, speaker, sentence, transcript, new_buffer

        segments = self.parse_to_segments(res[0])
        num_segments = len(segments)

        audio_duration = len(audio_buffer) / self.samplerate

        if num_segments == 0:
            if audio_duration > Config.max_silence_interval:
                new_buffer = np.array([],dtype=np.float32)
        elif num_segments == 1:
            # 只有一段
            if audio_duration - segments[0].end > Config.max_silence_interval:
                # 音频尾段过长，则认为结束
                final = True
                sentence = segments[0].text
                transcript = ""
                new_buffer = np.array([],dtype=np.float32)
            else:
                # 音频尾段不长，则认为继续
                transcript = segments[0].text

            self.dump(final, audio_buffer)
        elif num_segments >= 2:
            # 如果有多段，则截取最后一段
            sentence = ""
            for i in range(num_segments - 1):
                sentence += segments[i].text
            transcript = segments[num_segments - 1].text

            # 截取最后一段音频作为新的音频缓冲区
            cut_point = int(segments[num_segments - 2].end * self.samplerate)
            last_buffer = audio_buffer[:cut_point]
            speaker = self.speaker_verifier.match_speaker(last_buffer)
            final = True
            new_buffer = audio_buffer[cut_point:]

            self.dump(final, last_buffer)

        return final, speaker, sentence, transcript, new_buffer

    def inference(self, audio_data, last_speaker, last_sentence, last_transcript, last_buffer):
        if Config.speech_enhance.get("enable"):
            # 语音增强
            audio_data = self.speech_enhance.enhance(audio_data, self.samplerate)

        # 合并 last_buffer 和 chunk_audio
        audio_buffer = np.concatenate([last_buffer, audio_data])

        # 转录，last_sentence 为上一段转录的完整句子，可作为 prompt 或 hotwords
        final, speaker, sentence, transcript, new_buffer = self.transcript(audio_buffer, last_speaker, last_sentence, last_transcript)

        return final, speaker, sentence, transcript, new_buffer


if __name__ == "__main__":
    transcriptor = Transcriptor()

    # 读取音频文件
    audio = AudioSegment.from_file("./examples/asr_example.wav")
    audio = audio.set_frame_rate(transcriptor.samplerate)

    # 设置音频数据为 int16 格式
    audio = audio.set_sample_width(2)
    # 将双声道转换为单声道
    if audio.channels == 2:
        audio = audio.set_channels(1)

    # 打印信息
    print(f"采样率: {audio.frame_rate} Hz")
    print(f"样本宽度: {audio.sample_width} 字节")
    print(f"音频时长: {len(audio) / 1000} 秒")

    samples = np.array(audio.get_array_of_samples())
    # print(samples.shape)

    last_speaker = "guest"
    last_sentence = ""
    last_transcript = ""
    last_buffer = np.array([],dtype=np.float32)

    chunk_size = int(Config.samplerate * 0.5)
    for i in range(0, len(samples), chunk_size):
        audio_data = samples[i:i + chunk_size]

        audio_f32 = audio_data.astype(np.float32) / 32768.0
        final, speaker, sentence, transcript, new_buffer = transcriptor.inference(
            audio_f32, last_speaker, last_sentence, last_transcript, last_buffer)
        if final:
            print("\r\033[K", end="", flush=True)
            print(f"{speaker}: {sentence}")
            print(transcript, end="", flush=True)
        else:
            print("\r\033[K", end="", flush=True)
            print(transcript, end="", flush=True)

        last_sentence = sentence
        last_transcript = transcript
        last_buffer = new_buffer

    print("")
