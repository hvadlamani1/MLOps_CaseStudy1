import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app

class TestApp(unittest.TestCase):
    def setUp(self):
        # reset resources before test
        app.model = None
        app.processor = None
        app.atc_translator = None

    @patch('app.sf.read')
    @patch('app.load_resources')
    def test_transcribe_audio(self, mock_load_resources, mock_sf_read):
        """
        Test the transcribe_audio functionality by mocking model inference.
        """
        # Setup Mock Audio
        # Create a dummy audio array 
        dummy_audio = np.zeros(16000, dtype=np.float32)
        mock_sf_read.return_value = (dummy_audio, 16000)

        # Setup Mock Resources
        app.model = MagicMock()
        app.processor = MagicMock()
        app.atc_translator = MagicMock()
        app.device = "cpu"
        app.torch_dtype = torch.float32

        # Mock Processor behavior
        # processor(speech, ...) returns an object with .input_features
        mock_features = MagicMock()
        mock_features.to.return_value = mock_features # Mock .to(device)
        app.processor.return_value.input_features = mock_features

        # Mock Model behavior
        # model.generate returns generated_ids
        mock_generated_ids = MagicMock()
        app.model.generate.return_value = mock_generated_ids

        # Mock Processor decode
        app.processor.batch_decode.return_value = ["Delta 123, climb and maintain level 300"]

        # Mock ATC Translator behavior
        # It has .tokenizer.apply_chat_template
        app.atc_translator.tokenizer.apply_chat_template.return_value = "dummy prompt"
        # The pipeline itself is callable
        app.atc_translator.return_value = [{'generated_text': "Delta 123, please climb to 30,000 feet."}]

        # execution
        generator = app.transcribe_audio("dummy_path.wav", use_local_model=True)
        results = list(generator)
        final_transcription_update, final_translation_update = results[-1]

        # Assertions
        self.assertEqual(final_transcription_update['value'], "Delta 123, climb and maintain level 300")
        self.assertEqual(final_translation_update['value'], "Delta 123, please climb to 30,000 feet.")
        
        # Verify calls
        mock_load_resources.assert_called()
        mock_sf_read.assert_called_with("dummy_path.wav")
        app.model.generate.assert_called()

if __name__ == '__main__':
    unittest.main()
