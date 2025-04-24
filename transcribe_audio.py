import whisper
import numpy as np
import io
import tempfile

model = whisper.load_model("turbo")

def transcribe_audio(audio_bytes):
    """
    Transcribes audio from byte data.

    Args:
        audio_bytes:  The audio data in bytes.

    Returns:
        str: The transcribed text, or None on error.
    """
    try:
        # Write the bytes to a temporary file so that whisper.load_audio can read it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_file_path = temp_audio_file.name

        # Load the audio data from the temporary file
        audio = whisper.load_audio(temp_file_path)
        audio = whisper.pad_or_trim(audio)

        # Create the mel spectrogram and move to the same device as the model
        #mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # Decode the audio with default options
        #options = whisper.DecodingOptions()
        #result = model.decode(mel, options)
        result = model.transcribe(audio, language="en")
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
