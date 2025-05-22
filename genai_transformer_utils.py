from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

music = synthesiser(
    "Generate HD quality of the natural rain sound with the thunderstong in the background",
    forward_params={"do_sample": True},
)

# scipy.io.mp3.write(
#     "rain_out.mp3", rate=music["sampling_rate"], data=music["audio"]
# )
scipy.io.wavfile.write("rain_out.wav", rate=music["sampling_rate"], data=music["audio"])
