FROM megrez:pytorch-2.8.0_cuda-12.8_python-3.12_ubuntu-22.04

WORKDIR /app

# 设置非交互模式 + 快速安装
ENV DEBIAN_FRONTEND=noninteractive

# 更新源并安装 Opus 系统库
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libopus0 libopus-dev && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -c "import ctypes.util; print(ctypes.util.find_library('opus'))"

COPY requirements-server.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY preheat_audio.wav .
COPY speaker_recognize.py .
COPY transcriptor.py .
COPY web_server.py .

EXPOSE 6002

ENTRYPOINT ["python", "web_server.py"]
