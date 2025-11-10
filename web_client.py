import queue
import pyaudio
import opuslib
import base64
import json
import threading
import websocket

SAMPLING_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FRAME_SIZE = 320  # 每 320 采样点为 1 帧
AUDIO_DATA_SIZE = 50    # 每 50 帧为 1 秒，每秒 16000 采样点
RECV_TIMEOUT = 3        # 接收结果超时时间，单位：秒


class WebClient():
    def __init__(self, url = "ws://localhost:6002"):
        self.frames = []
        self.audio_fifo = queue.Queue()
        self.recv_fifo = queue.Queue()

        self.opus_encoder = opuslib.Encoder(SAMPLING_RATE, AUDIO_CHANNELS, opuslib.APPLICATION_VOIP)
        self.ws = websocket.WebSocketApp(url, on_message=self.on_message, on_open=self.on_open)
        print("Client Init")

    def in_callback(self, in_data, frame_count, time_info, status):
        opus_audio = self.opus_encoder.encode(in_data, frame_size=AUDIO_FRAME_SIZE)
        header = len(opus_audio).to_bytes(2, 'big')
        self.frames.append(header + opus_audio)

        if len(self.frames) >= AUDIO_DATA_SIZE:
            self.audio_fifo.put(b"".join(self.frames))
            self.frames = []

        return (in_data, pyaudio.paContinue)

    def on_open(self, ws):
        print("Client connected")

        send_thread = threading.Thread(target=self.on_audio_process, args=(ws,))
        send_thread.daemon = True
        send_thread.start()

    def on_audio_process(self, ws):
        print("On handle audio fifo thread")

        request = {
            "audio_base64": "",
            "last_speaker": "guest",
            "last_sentence": "",
            "last_transcript": "",
            "last_buffer_base64": ""
        }

        while True:
            opus_audio = self.audio_fifo.get()
            audio_base64 = base64.b64encode(opus_audio).decode("utf-8")
            request["audio_base64"] = audio_base64

            message = json.dumps(request)
            ws.send(message)

            # 获取结果，并更新 request
            try:
                result_dict = self.recv_fifo.get(timeout=RECV_TIMEOUT)  # 设置超时时间为2秒
                request["last_speaker"] = result_dict.get("speaker")
                request["last_sentence"] = result_dict.get("sentence")
                request["last_transcript"] = result_dict.get("transcript")
                request["last_buffer_base64"] = result_dict.get("buffer_base64")
            except queue.Empty:
                print(f"Receive result timeout, no data received within {RECV_TIMEOUT} seconds.")
                continue

    def on_message(self, ws, message):
        result_dict = json.loads(message)

        try:
            if result_dict.get("final"):
                print("\r\033[K", end="", flush=True)
                print(f"{result_dict.get('speaker')}: {result_dict.get('sentence')}")
                print(result_dict.get("transcript"), end="", flush=True)
            else:
                print("\r\033[K", end="", flush=True)
                print(result_dict.get("transcript"), end="", flush=True)
        except websocket.WebSocketConnectionClosedException:
            print("receive result end")
            ws.close()

        self.recv_fifo.put(result_dict)


if __name__ == "__main__":
    client = WebClient("ws://localhost:6002")

    pa = pyaudio.PyAudio()
    stream_in = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLING_RATE,
        input=True,
        frames_per_buffer=AUDIO_FRAME_SIZE,
        stream_callback=client.in_callback
    )

    stream_in.start_stream()
    client.ws.run_forever()
