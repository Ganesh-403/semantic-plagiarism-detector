import os

class MultimediaScanner:
    """Ingests mp3, mp4, and wav files and transcribes them using local Whisper model."""
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None

    def load_model(self):
        if self.model is None:
            import whisper
            self.model = whisper.load_model(self.model_size)

    def transcribe_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        self.load_model()
        result = self.model.transcribe(file_path, word_timestamps=True)
        return result
