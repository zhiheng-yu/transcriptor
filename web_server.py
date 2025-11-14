import websocket_server
import base64
import opuslib_next
import json
import numpy as np

from transcriptor import Transcriptor

SAMPLING_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FRAME_SIZE = 320


class WebServer:
    def __init__(self):
        self.transcriptor = Transcriptor()
        self.opus_decoder = opuslib_next.Decoder(SAMPLING_RATE, AUDIO_CHANNELS)
        self.opus_encoder = opuslib_next.Encoder(SAMPLING_RATE, AUDIO_CHANNELS, opuslib_next.APPLICATION_VOIP)

        # 创建 WebSocket 服务器
        self.ws = websocket_server.WebsocketServer(host='0.0.0.0', port=6002)
        self.ws.set_fn_new_client(self.on_new_client)
        self.ws.set_fn_client_left(self.on_client_left)
        self.ws.set_fn_message_received(self.on_message)

        print("Server Init")

    def on_new_client(self, client, server):
        print(f"New client connected: {client['address'][0]+':'+str(client['address'][1])}")

    def on_client_left(self, client, server):
        print(f"Client disconnected: {client['address'][0]+':'+str(client['address'][1])}")

    def encode_opus(self, audio_data):
        opus_list = []

        # 检查音频数据长度，如果小于一帧或为空，直接返回空字节串
        if len(audio_data) < AUDIO_FRAME_SIZE:
            print(f"数据长度小于一帧: {len(audio_data)}, 返回空字节串")
            return b""

        for i in range(len(audio_data) // AUDIO_FRAME_SIZE):
            chunk = audio_data[i*AUDIO_FRAME_SIZE:(i+1)*AUDIO_FRAME_SIZE]

            print(f"编码: {len(chunk)}")
            opus_audio = self.opus_encoder.encode(chunk.tobytes(), frame_size=AUDIO_FRAME_SIZE)
            print(f"编码后: {len(opus_audio)}")
            header = len(opus_audio).to_bytes(2, 'big')
            opus_list.append(header + opus_audio)

        return b"".join(opus_list)

    def decode_opus(self, opus_audio):
        pcm_list = []

        while len(opus_audio) >= 2:  # 至少要有长度头
            # 读取长度头
            packet_len = int.from_bytes(opus_audio[:2], 'big')
            total_packet_size = 2 + packet_len  # 头 + 数据

            if len(opus_audio) < total_packet_size:
                # 数据不完整，等待更多数据
                print(f"数据不完整，等待更多数据: {len(opus_audio)}")
                break

            # 提取完整包
            opus_packet = opus_audio[2:total_packet_size]
            opus_audio = opus_audio[total_packet_size:]  # 移除已处理部分

            # 解码
            try:
                print(f"解码: {len(opus_packet)}")
                pcm = self.opus_decoder.decode(opus_packet, frame_size=AUDIO_FRAME_SIZE)
                print(f"解码后: {len(pcm)}")
                pcm_list.append(pcm)
            except Exception as e:
                print(f"解码失败: {e}, 数据长度: {len(opus_packet)}")
                continue  # 跳过损坏包

        return b"".join(pcm_list)

    # 处理客户端消息
    def on_message(self, client, server, message):
        try:
            resquest = json.loads(message)

            audio_data = np.frombuffer(
                self.decode_opus(base64.b64decode(resquest["audio_base64"])),
                dtype=np.int16
            )
            audio_f32 = audio_data.astype(np.float32) / 32768.0

            last_speaker = resquest["last_speaker"]
            last_sentence = resquest["last_sentence"]
            last_transcript = resquest["last_transcript"]
            last_buffer = np.frombuffer(
                self.decode_opus(base64.b64decode(resquest["last_buffer_base64"])),
                dtype=np.int16
            )
            last_buffer_f32 = last_buffer.astype(np.float32) / 32768.0

            final, speaker, sentence, transcript, new_buffer_f32 = self.transcriptor.inference(
                audio_f32, last_speaker, last_sentence, last_transcript, last_buffer_f32)

            new_buffer_i16 = (new_buffer_f32 * 32768.0).astype(np.int16)

            inference_result = {
                "final": final,
                "speaker": speaker,
                "sentence": sentence,
                "transcript": transcript,
                "buffer_base64": base64.b64encode(self.encode_opus(new_buffer_i16)).decode("utf-8")
            }

            server.send_message(client, json.dumps(inference_result, ensure_ascii=False, indent=4))
        except Exception as e:
            print(f"Warning processing message: {e}")


if __name__ == "__main__":
    server = WebServer()
    server.ws.run_forever()
