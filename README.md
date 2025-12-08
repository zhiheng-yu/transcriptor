# funasr 实时语音转录系统

本项目是一个基于 `funasr` 的实时语音转录系统，支持流式音频输入和低延迟转录。系统由后端服务和前端客户端组成，可应用于会议记录、实时字幕等场景。

## 项目结构

- `transcriptor.py`: 核心转录类，集成语音活动检测（VAD）、ASR转录和文本过滤功能
- `config.py`: 配置文件，包含模型路径、VAD参数、过滤规则等
- `web_server.py`: WebSocket 服务端，处理客户端连接和转录请求
- `web_client.py`: WebSocket 客户端，采集麦克风音频并发送到服务器
- `cache/`: 转录音频缓存目录
- `checkpoints/`: 模型存储目录
- `examples/`: 示例音频文件目录，用于测试和演示系统功能
- `register_db/`: 发言人注册音频库，用于存放注册用户的说话人样本
- `preheat_audio.wav`: 模型预热音频文件

## 核心功能

### 1. 声音增强

在转录前对音频进行预处理，提升信噪比和音质，改善转录准确性。

**工作原理**:
- 使用 ClearVoice 的 MossFormer2_SE_48K 模型进行语音增强，去除背景噪音
- 使用 pyloudnorm 进行响度归一化，统一音频音量
- 自动处理不同采样率（内部转换为 48kHz 处理）

**配置参数** (`config.py`):
- `speech_enhance.enable`: 是否启用声音增强 (默认 `True`)
- `speech_enhance.model_name`: 增强模型名称 (默认 `"MossFormer2_SE_48K"`)
- `speech_enhance.target_lufs`: 目标响度值，单位 LUFS (默认 `-16.0`)
- `speech_enhance.true_peak_limit`: 真峰值限制，单位 dBTP (默认 `-1.0`)

> **注意**: 音频长度小于 0.4 秒时，将仅进行简单的峰值归一化以保证处理稳定性。

### 2. 实时转录

基于 funasr 模型实现流式转录，支持以下特性：

- **自动语言检测**: 支持自动识别音频语言类型
- **逆文本规范化**: 自动将数字、日期等转换为标准文本格式
- **VAD 智能合并**: 通过 `merge_vad` 和 `merge_length_s` 参数合并相邻语音片段，提升长句转录准确性
- **句子时间戳**: 提供每个句子的起止时间信息，支持精确的音频定位
- **上下文连续性**: 通过音频缓冲区管理保持流式转录的上下文连续性

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

下载 `ERes2NetV2` 模型到 `checkpoints/` 目录

```bash
cd checkpoints

modelscope download --model iic/ClearerVoice-Studio MossFormer2_SE_48K/last_best_checkpoint --local_dir .
modelscope download --model iic/ClearerVoice-Studio MossFormer2_SE_48K/last_best_checkpoint.pt --local_dir .
modelscope download --model iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common --local_dir ./ERes2NetV2_w24s4ep4
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
