from asr_pipeline.audio import resample_audio
import numpy as np

# Two promises tested here: 1. Does it leave a correct sample rate unaltered and 2. Does it correctly resample?
def test_resample_audio_same_rate():
    audio=np.array([0.1,0.2,0.3])
    result=resample_audio(
        audio,
        sample_rate=16000,
        target_rate=16000
    )
    np.testing.assert_array_equal(result,audio)

def test_resample_audio_different_rate():
    audio=np.zeros(8000)
    result=resample_audio(
        audio,
        sample_rate=8000,
        target_rate=16000
    )
    assert len(result)==16000
