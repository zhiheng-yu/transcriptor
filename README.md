# faster-whisper 实时语音转录系统

本项目是一个基于 `faster-whisper` 和 `Silero VAD` 的实时语音转录系统，支持流式音频输入和低延迟转录。系统由后端服务和前端客户端组成，可应用于会议记录、实时字幕等场景。

## 项目结构

- `transcriptor.py`: 核心转录类，集成语音活动检测（VAD）、ASR转录和文本过滤功能
- `config.py`: 配置文件，包含模型路径、VAD参数、过滤规则等
- `web_server.py`: WebSocket 服务端，处理客户端连接和转录请求
- `web_client.py`: WebSocket 客户端，采集麦克风音频并发送到服务器
- `models/`: 模型存储目录
- `cache/`: 转录音频缓存目录
- `preheat_audio.wav`: 模型预热音频文件

## 核心功能

### 1. 语音活动检测 (VAD)

使用 Silero VAD 模型检测语音活动，有效过滤静音段，提升转录效率和准确性。

**配置参数** (`config.py`):
- `vad_threshold`: VAD 检测阈值 (默认 0.1)
- `min_silence_duration`: 最小静音时长 (默认 12 帧 ≈ 375ms)
- `min_voice_duration`: 最小语音时长 (默认 8 帧 ≈ 250ms)
- `silence_reserve`: 语音段前后保留的静音采样点 (默认 6 帧 ≈ 187.5ms)

### 2. 实时转录

基于 faster-whisper 模型实现流式转录，支持以下特性：

- **上下文感知**: 使用上一段落文本作为 prompt 或 hotwords，提升转录连贯性
- **幻觉抑制**: 通过 `suppress_blank` 和 `repetition_penalty` 参数减少模型幻觉
- **多温度采样**: 支持 `[0.0, 0.2, 0.6, 1.0]` 温度序列，平衡生成质量和多样性
- **繁体转简体**: 可选开启繁体中文到简体中文的转换

### 3. 文本过滤

集成基于 TF-IDF 和余弦相似度的文本过滤机制，自动过滤预设的干扰文本。

**过滤规则**:
- **精确匹配**: 过滤包含特定关键词的文本（如 "谢谢大家"、"这是一段会议录音。"）
- **语义相似度**: 使用余弦相似度过滤与预设文本相似度超过阈值（默认 0.02）的文本

## 配置说明

详细配置项请参考 `config.py` 文件，主要配置包括：

- **模型路径**: `models.asr.path` 和 `models.vad.path`
- **设备设置**: 支持 CUDA 加速 (`device: "cuda"`)
- **计算类型**: `compute_type: "float16"` 以提升性能
- **预热音频**: `preheat_audio` 用于模型预热，减少首次推理延迟
- **音频保存**: `dump.audio_save` 可设置为 `all`、`final` 或 `none`

## 依赖安装

```bash
pip install -r requirements.txt
```

## 模型准备

1. 从 huggingface 下载 `faster-whisper-large-v3-turbo` 模型到 `models/faster-whisper-large-v3-turbo` 目录
2. 从 github 克隆 snakers4/silero-vad 到 `models/silero-vad` 目录

## 运行方式

### 1. 服务端启动

```bash
python web_server.py
```

服务将监听 `0.0.0.0:6002`，等待客户端连接。

**服务端返回消息格式**:

服务端通过 WebSocket 返回 JSON 格式的转录结果，包含以下字段：

- `final`: 布尔值，代表是否包含完整句子
- `speaker`: 字符串，上一个完整句子的发言人
- `sentence`: 字符串，包含上一个完整句子（当 `final` 为 `true` 时有效）
- `transcript`: 字符串，当前句子的实时转录结果
- `buffer_base64`: Base64 编码的字符串，为当前句子的音频缓存（Opus 编码），需要在下次推理时传入以保持上下文连续性

### 2. 客户端（Demo）启动

```bash
python web_client.py
```

客户端将连接到默认的 `ws://localhost:6002`，采集麦克风音频并实时显示转录结果。

> **注意**: 可通过修改 `web_client.py` 中的 URL 参数连接到远程服务器，例如：
> ```python
> client = WebClient("wss://your_server")
> ```
