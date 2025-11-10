import os

class Config:
    model_path = "./models"

    models = {
        "asr" : {
            "name" : "faster-whisper",
            "path" : os.path.join(model_path, "faster-whisper-large-v3-turbo"),
            "compute_type" : "float16",
            "device" : "cuda"
        },
        "vad" : {
            "name" : "silero",
            "path" : os.path.join(model_path, "silero-vad"),
            "compute_type" : "float16",
            "device" : "cuda"
        },
        "speaker_verifier" : {
            "name" : "ERes2NetV2",
            "path" : os.path.join(model_path, "ERes2NetV2_w24s4ep4")
        }
    }

    preheat_audio = "./preheat_audio.wav"

    dump = {
        "audio_save": "final",  # all: 保存所有音频，final: 只保存最终音频, none: 不保存
        "audio_dir": "./cache"
    }

    vad = {
        "skip" : False,
        "vad_threshold" : 0.1,
        "sampling_rate" : 16000,
        "sampling_per_chunk" : 512,
        "min_silence_duration" : 12,        # 12 * 31.25ms = 375ms
        "min_voice_duration" : 8,           # 8 * 31.25ms = 250ms
        "silence_reserve" : 6,              # 6 * 31.25ms = 187.5ms
    }

    filter_match = {
        "skip" : False,
        "find_match" : ["谢谢大家", "简体中文", "优独播剧场", "大家好，这是一段会议录音。"],
        "cos_match" : [
            "请不吝点赞 订阅 转发 打赏支持明镜与点栏目",
            "志愿者 李宗盛",
            "大家好，这是一段会议录音。",
            "字幕志愿者 杨栋梁",
            "明镜需要您的支持 欢迎订阅明镜",
            "优优独播剧场——YoYo Television Series Exclusive",
            "中文字幕——Yo Television Series Exclusive"
        ],
        "cos_sim" : 0.02
    }

    whisper_config = {
        "tradition_to_simple" : False,
        "interruption_duration": 20,    # 最大中断时长，单位：秒
        "beam_size" : 2,  # 1、beam_size调整为8 best_of调整为4 提高模型效果
        "best_of" : 1,    # 2、beam_size调整为4 best_of调整为1 速度更快
        "patience" : 1.0,
        "suppress_blank" : True,     # 幻觉抑制
        "repetition_penalty" : 1.2,  # 重复惩罚 但降低效果
        "log_prob_threshold" : -1.0,
        "no_speech_threshold" : 0.8,
        "condition_on_previous_text" : True,
        "previous_text_prompt" : False,
        "previous_text_hotwords" : True, # 把上段语句做为提示 断句相对更保守 以提升效果
        "previous_text_prefix" : False,
        "initial_prompt" : "大家好，这是一段会议录音。",
        "hotwords_text" : "",
        "temperature" : [0.0, 0.2, 0.6, 1.0],
        "avg_logprob_score" : -1.0  # 设置过滤阈值 低于阈值则不输出
    }
