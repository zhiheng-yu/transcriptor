import os
import torch
import scipy
from itertools import groupby
import numpy as np
from pydub import AudioSegment
import librosa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faster_whisper import WhisperModel

from config import Config


class Transcriptor:
    def __init__(self):
        self.load_models(Config.models)
        self.preheat(Config.preheat_audio)

        self.vectorizer = TfidfVectorizer()

        whisper_config = Config.whisper_config
        if whisper_config.get("tradition_to_simple"):
            import opencc
            self.cc = opencc.OpenCC('t2s.json')

        self.epoch = 0

    def load_models(self, models):
        asr_config = models.get("asr")
        vad_config = models.get("vad")

        self.asr_model = WhisperModel(
            model_size_or_path = asr_config["path"],
            device = asr_config["device"],
            local_files_only = False,
            compute_type = asr_config["compute_type"]
        )

        self.vad_model, _ = torch.hub.load(
            repo_or_dir = vad_config["path"],
            model = 'silero_vad',
            trust_repo = None,
            source = 'local',
        )

    def preheat(self, preheat_audio):
        whisper_config = Config.whisper_config

        preheat_audio_, _ = librosa.load(preheat_audio, sr=16000, dtype=np.float32)
        self.asr_model.transcribe(
            preheat_audio_,
            beam_size = whisper_config.get("beam_size"),
            best_of = whisper_config.get("best_of"),
            patience = whisper_config.get("patience"),
            suppress_blank = whisper_config.get("suppress_blank"),
            repetition_penalty = whisper_config.get("repetition_penalty"),
            log_prob_threshold = whisper_config.get("log_prob_threshold"),
            no_speech_threshold = whisper_config.get("no_speech_threshold"),
            condition_on_previous_text = whisper_config.get("condition_on_previous_text"),
            initial_prompt = whisper_config.get("initial_prompt"),
            hotwords = whisper_config.get("hotwords_text"),
            prefix = whisper_config.get("previous_text_prefix"),
            temperature = whisper_config.get("temperature"),
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
        scipy.io.wavfile.write(audio_path, rate=16000, data=audio_buffer)

    def vad_rm_silence(self, audio_chunk):
        vad_config = Config.vad
        if vad_config.get("skip"):
            return audio_chunk

        vad_flags = []
        chunk_num = len(audio_chunk) // 512
        sampling_rate = vad_config.get("sampling_rate")
        sampling_per_chunk = vad_config.get("sampling_per_chunk")

        for i in range(chunk_num):
            chunk = audio_chunk[i*sampling_per_chunk:(i+1)*sampling_per_chunk]

            chunk_torch = torch.tensor(chunk).unsqueeze(0)
            silero_score = self.vad_model(chunk_torch, sampling_rate).item()

            # 如果人生检测概率大于阈值，则认为有语音
            if silero_score > vad_config.get("vad_threshold"):
                vad_flags.append(1)
            else:
                vad_flags.append(0)

        # print("vad_flags: ", vad_flags)

        # 如果语音时间小于最小语音时间，则认为没有语音，直接返回空
        voice_duration = vad_flags.count(1)
        if voice_duration < vad_config.get("min_voice_duration"):
            return None

        # 如果静音时间小于最小静音时间，则认为没有静音，直接返回原始音频
        silence_duration = vad_flags.count(0)
        if silence_duration < vad_config.get("min_silence_duration"):
            return audio_chunk

        # 删除静音部分，但是语音前后均保留 silence_reserve 个采样点
        silence_reserve = vad_config.get("silence_reserve")
        # 找到所有语音段的起始和结束位置
        indices = []
        for flag, group in groupby(enumerate(vad_flags), lambda x: x[1]):
            if flag == 1:  # 语音段
                group = list(group)
                start = group[0][0]
                end = group[-1][0]
                indices.append((start, end))

        # print("indices: ", indices)

        split_chunk = []
        for start, end in indices:
            # 计算保留的前后静音区间
            # print("start: ", (start - silence_reserve), "end: ", (end + 1 + silence_reserve))
            start_sample = max(0, (start - silence_reserve) * sampling_per_chunk)
            end_sample = min(len(audio_chunk), (end + 1 + silence_reserve) * sampling_per_chunk)
            split_chunk.extend(audio_chunk[start_sample:end_sample])

        if len(split_chunk) > 0:
            return np.array(split_chunk, dtype=np.float32)
        else:
            return None

    def filter(self, text):
        filter_match = Config.filter_match
        if filter_match.get("skip"):
            return text

        for match_text in filter_match.get("find_match"):
            if text.find(match_text) != -1:
                return ""

        for match_text in filter_match.get("cos_match"):
            tfidf_matrix = self.vectorizer.fit_transform([match_text, text])

            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            if cos_sim > filter_match.get("cos_sim"):
                return ""

        return text

    def transcript(self, audio_buffer, last_sentence):
        whisper_config = Config.whisper_config

        initial_prompt = whisper_config.get("initial_prompt")
        if whisper_config.get("previous_text_prompt"):
            initial_prompt += last_sentence

        hotwords = whisper_config.get("hotwords_text")
        if whisper_config.get("previous_text_hotwords"):
            hotwords += last_sentence

        prefix_text = None
        if whisper_config.get("previous_text_prefix"):
            prefix_text = last_sentence

        interruption_duration = whisper_config.get("interruption_duration")

        segments, info = self.asr_model.transcribe(
            audio_buffer,
            beam_size = whisper_config.get("beam_size"),
            best_of = whisper_config.get("best_of"),
            patience = whisper_config.get("patience"),
            suppress_blank = whisper_config.get("suppress_blank"),
            repetition_penalty = whisper_config.get("repetition_penalty"),
            log_prob_threshold = whisper_config.get("log_prob_threshold"),
            no_speech_threshold = whisper_config.get("no_speech_threshold"),
            condition_on_previous_text = whisper_config.get("condition_on_previous_text"),
            initial_prompt = initial_prompt,
            hotwords = hotwords,
            prefix = prefix_text,
            temperature = whisper_config.get("temperature"),
        )
        # print("transcript info: ", info)

        final = False
        sentence = last_sentence
        transcript = ""
        new_buffer = audio_buffer

        # 计算音频时长
        audio_duration = len(audio_buffer) / 16000

        # 获取转录结果
        generated_segments = []
        for segment in segments:
            generated_segments.append(segment)
        num_segments = len(generated_segments)

        if num_segments == 0:
            # 如果转录结果为空，则直接返回
            return False, sentence, transcript, new_buffer
        elif num_segments == 1:
            # 如果只有一段，则记录转录信息
            # print("log: ", generated_segments[0].avg_logprob)
            if generated_segments[0].avg_logprob > whisper_config.get("log_prob_threshold"):
                transcript = generated_segments[0].text
            else:
                transcript = ""

            # 如果音频时长超过最大中断时长，则认为中断结束
            if audio_duration > interruption_duration:
                print(f"Warning: audio buffer over {interruption_duration} seconds, interrupt")
                sentence = transcript
                transcript = ""
                new_buffer = np.array([],dtype=np.float32)
                final = True
            else:
                final = False

            self.dump(final, audio_buffer)
        elif num_segments >= 2:
            # 如果有多段，则截取最后一段
            sentence = ""
            for i in range(num_segments - 1):
                sentence += generated_segments[i].text
            # print("log: ", generated_segments[num_segments - 1].avg_logprob)
            if generated_segments[num_segments - 1].avg_logprob > whisper_config.get("log_prob_threshold"):
                transcript = generated_segments[num_segments - 1].text
            else:
                transcript = ""
            cut_point = int(generated_segments[num_segments - 2].end * 16000)
            new_buffer = audio_buffer[cut_point:]

            final = True
            self.dump(final, audio_buffer[:cut_point])

        if whisper_config.get("tradition_to_simple"):
            # 繁体到简体
            transcript = self.cc.convert(transcript)

        return final, sentence, transcript, new_buffer

    def inference(self, audio_data, last_sentence, last_transcript, last_buffer):
        # vad 过滤静音
        audio_data = self.vad_rm_silence(audio_data)

        # 如果 audio_f32 为空，不做转录
        if audio_data is None:
            if len(last_buffer) > 0 and len(last_transcript) > 0:
                # 如果 last_buffer 不为空，则视为结束，完整句子为 last_transcript ，新的转录结果为空，新的音频缓冲区为空
                self.dump(True, last_buffer)
                new_buffer = np.array([],dtype=np.float32)
                return True, last_transcript, "", new_buffer
            else:
                # 如果 last_buffer 为空，则视为未结束
                return False, last_sentence, last_transcript, last_buffer

        # 合并 last_buffer 和 chunk_audio
        audio_buffer = np.concatenate([last_buffer, audio_data])

        # 转录，last_sentence 为上一段转录的完整句子，可作为 prompt 或 hotwords
        final, sentence, transcript, new_buffer = self.transcript(audio_buffer, last_sentence)

        # 过滤幻觉词
        transcript = self.filter(transcript)

        return final, sentence, transcript, new_buffer


if __name__ == "__main__":
    transcriptor = Transcriptor()

    # 读取音频文件
    audio = AudioSegment.from_file("./asr_example.wav")
    audio = audio.set_frame_rate(16000)

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

    last_sentence = ""
    last_transcript = ""
    last_buffer = np.array([],dtype=np.float32)

    # 按 1 秒的频率读取数据
    audio_size = 16384  # 每秒的样本数
    for i in range(0, len(samples), audio_size):
        audio_data = samples[i:i + audio_size]

        audio_f32 = audio_data.astype(np.float32) / 32768.0
        final, sentence, transcript, new_buffer = transcriptor.inference(
            audio_f32, last_sentence, last_transcript, last_buffer)
        if final:
            print("\r\033[K", end="", flush=True)
            print(sentence)
            print(transcript, end="", flush=True)
        else:
            print("\r\033[K", end="", flush=True)
            print(transcript, end="", flush=True)

        last_sentence = sentence
        last_transcript = transcript
        last_buffer = new_buffer

    print("")
