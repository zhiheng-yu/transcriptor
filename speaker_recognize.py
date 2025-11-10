from modelscope.pipelines import pipeline

from config import Config


class SpeakerVerifier:
    def __init__(self):
        sv_config = Config.models['speaker_verifier']
        self.sv_pipeline = pipeline(task='speaker-verification', model=sv_config['path'])

        self.registered_speaker = {}

    def compare(self, audio1, audio2, thr=0.5):
        return self.sv_pipeline([audio1, audio2], thr=thr)

    def verify(self, audio1, audio2, thr=0.5):
        result = self.compare(audio1, audio2, thr=0.5)
        return result >= thr

    def register_speaker(self, speaker_id, audio):
        self.registered_speaker[speaker_id] = audio

    def match_speaker(self, audio, thr=0.3):
        if len(self.registered_speaker) == 0:
            return "guest"

        match_scores = {}
        for speaker_id, registered_audio in self.registered_speaker.items():
            similarity = self.compare(audio, registered_audio, thr=thr)
            # print(similarity)
            match_scores[speaker_id] = similarity['score']

        match_speaker_id, max_value = max(match_scores.items(), key=lambda item: item[1])
        if max_value >= thr:
            return match_speaker_id
        else:
            return "guest"


if __name__ == '__main__':
    speaker_verifier = SpeakerVerifier()

    speaker_verifier.register_speaker('speaker1', './examples/speaker1_a_cn_16k.wav')
    speaker_verifier.register_speaker('speaker2', './examples/speaker2_a_cn_16k.wav')

    print("match to:", speaker_verifier.match_speaker('./examples/speaker1_b_cn_16k.wav'))
