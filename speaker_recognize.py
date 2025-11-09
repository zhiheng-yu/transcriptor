from modelscope.pipelines import pipeline


class SpeakerVerifier:
    def __init__(self):
        self.sv_pipeline = pipeline(
            task='speaker-verification',
                model='iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common',
                model_revision='v1.0.1'
        )

        self.registered_speaker = {}

    def compare(self, audio1, audio2, thr=0.5):
        return self.sv_pipeline([audio1, audio2], thr=thr)

    def verify(self, audio1, audio2, thr=0.5):
        result = self.compare(audio1, audio2, thr=0.5)
        return result >= thr

    def register_speaker(self, speaker_id, audio):
        self.registered_speaker[speaker_id] = audio

    def match_speaker(self, audio, thr=0.3):
        match_scores = {}
        for speaker_id, registered_audio in self.registered_speaker.items():
            similarity = self.compare(audio, registered_audio, thr=thr)
            match_scores[speaker_id] = similarity['score']

        match_speaker_id, max_value = max(match_scores.items(), key=lambda item: item[1])
        if max_value >= thr:
            return match_speaker_id
        else:
            return "guest"


if __name__ == '__main__':
    speaker_verifier = SpeakerVerifier()

    speaker_verifier.register_speaker('liu', './register_db/liu.wav')
    speaker_verifier.register_speaker('lu', './register_db/lu.wav')
    speaker_verifier.register_speaker('yu', './register_db/yu.wav')

    print("match to:", speaker_verifier.match_speaker('./examples/liu_example.wav'))
    print("match to:", speaker_verifier.match_speaker('./examples/lu_example.wav'))
    print("match to:", speaker_verifier.match_speaker('./examples/yu_example.wav'))
