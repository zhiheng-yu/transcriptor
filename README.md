# faster-whisper 实时语音转录系统

本项目是一个基于 `faster-whisper` 和 `Silero VAD` 的实时语音转录系统，支持流式音频输入和低延迟转录。系统由后端服务和前端客户端组成，可应用于会议记录、实时字幕等场景。

## 项目结构

- `transcriptor.py`: 核心转录类，集成语音活动检测（VAD）、ASR转录和文本过滤功能
- `config.py`: 配置文件，包含模型路径、VAD参数、过滤规则等
- `web_server.py`: WebSocket 服务端，处理客户端连接和转录请求
- `web_client.py`: WebSocket 客户端，采集麦克风音频并发送到服务器
- `cache/`: 转录音频缓存目录
- `models/`: 模型存储目录
- `examples/`: 示例音频文件目录，用于测试和演示系统功能
- `register_db/`: 发言人注册音频库，用于存放注册用户的说话人样本
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

### 3. 发言人识别

基于 ModelScope 的 ERes2NetV2 模型实现发言人验证，支持多发言人场景的自动识别。

**工作原理**:
- 预先注册发言人音频样本
- 当检测到完整句子时，自动匹配最相似的发言人
- 通过余弦相似度计算匹配度，低于阈值则标记为 `guest`

**配置参数** (`config.py`):
- `models.speaker_verifier.path`: ERes2NetV2 模型路径
- `models.speaker_verifier.speakers`: 注册发言人列表，包含 `id` 和 `path` 字段
- 相似度阈值默认为 0.3

**注册发言人示例**:

```python
"speakers": [
    { "id": "speaker1", "path": "./register_db/speaker1_sample.wav" },
    { "id": "speaker2", "path": "./register_db/speaker2_sample.wav" },
]
```

> **注意**: 音频样本建议使用 16kHz 采样率的单声道 WAV 格式，时长建议 5~10 秒。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 模型准备

下载 `faster-whisper` 、 `ERes2NetV2` 和 `silero-vad` 模型到 `models/` 目录

```bash
cd models

modelscope download --model mobiuslabsgmbh/faster-whisper-large-v3-turbo --local_dir ./faster-whisper-large-v3-turbo
modelscope download --model iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common --local_dir ./ERes2NetV2_w24s4ep4

git clone https://github.com/snakers4/silero-vad.git
```

## 运行方式

### 1. 服务端启动（python）

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

### 2. 服务端启动（docker）

```bash
# 1. 构建 Docker 镜像
docker build -t transcriptor:latest .

# 2. 运行容器
docker compose up -d
```

### 3. 客户端（Demo）启动

```bash
python web_client.py
```

客户端将连接到默认的 `ws://localhost:6002`，采集麦克风音频并实时显示转录结果。

> **注意**: 可通过修改 `web_client.py` 中的 URL 参数连接到远程服务器，例如：
> ```python
> client = WebClient("wss://your_server")
> ```
