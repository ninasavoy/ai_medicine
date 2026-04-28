"""
Respiratory Sound Diagnosis Predictor

Use a trained CNN model to predict respiratory diagnoses from spectrograms or audio files.
"""

from pathlib import Path
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt


class DiagnosisPredictor:
    """Predictor for respiratory sound diagnoses."""
    
    def __init__(self, model_path, encoder_path, target_shape=(128, 128)):
        """
        Initialize predictor with model and encoder.
        
        Args:
            model_path: Path to trained model (.h5)
            encoder_path: Path to label encoder (.pkl)
            target_shape: Expected input shape for spectrograms
        """
        self.model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.target_shape = target_shape
    
    def load_spectrogram(self, filepath):
        """Load and normalize spectrogram."""
        try:
            spec = np.load(filepath)
            # Resize if needed
            if spec.shape != self.target_shape:
                from scipy import ndimage
                zoom_factors = (
                    self.target_shape[0] / spec.shape[0],
                    self.target_shape[1] / spec.shape[1]
                )
                spec = ndimage.zoom(spec, zoom_factors, order=1)
            
            # Normalize
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            spec = spec[np.newaxis, ..., np.newaxis]  # Add batch and channel
            return spec
        except Exception as e:
            print(f'Error loading spectrogram {filepath}: {e}')
            return None
    
    def predict_from_spectrogram(self, spectrogram_path):
        """
        Predict diagnosis from spectrogram file.
        
        Args:
            spectrogram_path: Path to .npy spectrogram file
            
        Returns:
            (diagnosis, confidence, probabilities_dict)
        """
        spec = self.load_spectrogram(spectrogram_path)
        if spec is None:
            return None, None, None
        
        # Predict
        probs = self.model.predict(spec, verbose=0)
        predicted_class = np.argmax(probs[0])
        confidence = probs[0][predicted_class]
        diagnosis = self.label_encoder.classes_[predicted_class]
        
        # Create probability dict
        probs_dict = {
            self.label_encoder.classes_[i]: float(probs[0][i])
            for i in range(len(self.label_encoder.classes_))
        }
        
        return diagnosis, float(confidence), probs_dict
    
    def predict_from_audio(self, audio_path, sr=22050, n_mels=128, 
                          n_fft=1024, hop_length=512, fmin=50, fmax=4000):
        """
        Predict diagnosis from audio file by converting to spectrogram.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency
            
        Returns:
            (diagnosis, confidence, probabilities_dict)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr)
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                hop_length=hop_length, fmin=fmin, fmax=fmax
            )
            
            # Convert to dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to target shape
            from scipy import ndimage
            zoom_factors = (
                self.target_shape[0] / mel_spec_db.shape[0],
                self.target_shape[1] / mel_spec_db.shape[1]
            )
            mel_spec_db = ndimage.zoom(mel_spec_db, zoom_factors, order=1)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            spec = mel_spec_db[np.newaxis, ..., np.newaxis]
            
            # Predict
            probs = self.model.predict(spec, verbose=0)
            predicted_class = np.argmax(probs[0])
            confidence = probs[0][predicted_class]
            diagnosis = self.label_encoder.classes_[predicted_class]
            
            # Create probability dict
            probs_dict = {
                self.label_encoder.classes_[i]: float(probs[0][i])
                for i in range(len(self.label_encoder.classes_))
            }
            
            return diagnosis, float(confidence), probs_dict
        
        except Exception as e:
            print(f'Error processing audio {audio_path}: {e}')
            return None, None, None
    
    def visualize_prediction(self, spectrogram_or_audio_path, is_audio=False, save_path=None):
        """
        Visualize spectrogram and prediction.
        
        Args:
            spectrogram_or_audio_path: Path to spectrogram or audio file
            is_audio: Whether input is audio file
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Make prediction
        if is_audio:
            diagnosis, confidence, probs_dict = self.predict_from_audio(spectrogram_or_audio_path)
            
            # Load and plot spectrogram
            y, sr = librosa.load(spectrogram_or_audio_path, sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            im = axes[0].imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_ylabel('Frequency Bins')
            axes[0].set_xlabel('Time Frames')
            plt.colorbar(im, ax=axes[0])
        else:
            diagnosis, confidence, probs_dict = self.predict_from_spectrogram(spectrogram_or_audio_path)
            
            # Load and plot spectrogram
            spec = np.load(spectrogram_or_audio_path)
            im = axes[0].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_ylabel('Frequency Bins')
            axes[0].set_xlabel('Time Frames')
            plt.colorbar(im, ax=axes[0])
        
        axes[0].set_title('Spectrogram')
        
        # Plot probabilities
        if probs_dict:
            classes = list(probs_dict.keys())
            probs = list(probs_dict.values())
            colors = ['green' if c == diagnosis else 'blue' for c in classes]
            
            axes[1].barh(classes, probs, color=colors, alpha=0.7)
            axes[1].set_xlabel('Probability')
            axes[1].set_title(f'Prediction: {diagnosis}\nConfidence: {confidence:.2%}')
            axes[1].set_xlim([0, 1])
            
            for i, v in enumerate(probs):
                axes[1].text(v + 0.01, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Figure saved to {save_path}')
        
        plt.show()
        
        return diagnosis, confidence, probs_dict


def main():
    parser = argparse.ArgumentParser(
        description='Predict respiratory sound diagnosis from spectrogram or audio'
    )
    parser.add_argument(
        'input_file',
        help='Path to spectrogram (.npy) or audio file'
    )
    parser.add_argument(
        '--model',
        default='models/diagnosis_classifier.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--encoder',
        default='models/label_encoder.pkl',
        help='Path to label encoder'
    )
    parser.add_argument(
        '--audio',
        action='store_true',
        help='Treat input as audio file (if not specified, assumes spectrogram)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization'
    )
    parser.add_argument(
        '--save',
        help='Save visualization to file'
    )
    
    args = parser.parse_args()
    
    # Load predictor
    project_root = Path(__file__).parent
    model_path = project_root / args.model
    encoder_path = project_root / args.encoder
    
    if not model_path.exists() or not encoder_path.exists():
        print(f'Error: Model or encoder not found')
        print(f'  Model: {model_path}')
        print(f'  Encoder: {encoder_path}')
        return
    
    predictor = DiagnosisPredictor(str(model_path), str(encoder_path))
    
    # Make prediction
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f'Error: Input file not found: {input_file}')
        return
    
    is_audio = args.audio or input_file.suffix in ['.wav', '.mp3', '.ogg', '.flac']
    
    print(f'Processing: {input_file}')
    
    if is_audio:
        diagnosis, confidence, probs = predictor.predict_from_audio(str(input_file))
    else:
        diagnosis, confidence, probs = predictor.predict_from_spectrogram(str(input_file))
    
    if diagnosis is None:
        print('Prediction failed')
        return
    
    print(f'\nPrediction Results:')
    print(f'  Diagnosis: {diagnosis}')
    print(f'  Confidence: {confidence:.2%}')
    print(f'\n  Probabilities:')
    for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f'    {class_name}: {prob:.2%}')
    
    if args.visualize:
        predictor.visualize_prediction(str(input_file), is_audio=is_audio, save_path=args.save)


if __name__ == '__main__':
    main()
