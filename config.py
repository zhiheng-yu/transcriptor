import os

class Config:
    model_path = "./checkpoints"
    registers_path = "./register_db"

    models = {
        "asr": {
            "name": "paraformer-zh",
            "device": "cuda"
        },
        "vad": {
            "name": "fsmn-vad",
            "device": "cuda"
        },
        "punc": {
            "name": "ct-punc-c",
            "device": "cuda"
        },
        "speaker_verifier": {
            "name": "ERes2NetV2",
            "path": os.path.join(model_path, "ERes2NetV2_w24s4ep4"),
            "speakers": [
                # 注册说话人，格式：
                # { "id": "speaker1", "path": os.path.join(registers_path, "speaker1_a_cn_16k.wav") },
                # { "id": "speaker2", "path": os.path.join(registers_path, "speaker2_a_cn_16k.wav") },
            ]
        }
    }

    samplerate = 16000
    preheat_audio = "./preheat_audio.wav"
    max_silence_interval = 2    # 最大间隔时长，单位：秒，超过该时长则认为中断
    max_speech_duration = 20    # 最大音频时长，单位：秒，超过该时长则强制结束说话人验证

    dump = {
        "audio_save": "none",   # all: 保存所有音频，final: 只保存最终音频, none: 不保存
        "audio_dir": "./dumps"
    }

    speech_enhance = {
        "enable": False,
        "model_name": "MossFormer2_SE_48K",
        "target_lufs": -16.0,
        "true_peak_limit": -1.0,
        "mute_if_too_quiet": True,
        "threshold_dbfs": -50,
    }
