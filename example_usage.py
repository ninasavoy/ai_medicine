"""
Exemplo de uso da classificação de diagnóstico

Este script demonstra como usar o modelo treinado para fazer predições.
"""

from pathlib import Path
from predict_diagnosis import DiagnosisPredictor

def example_predict_from_spectrogram():
    """Exemplo: Predizer diagnóstico a partir de um espectrograma."""
    
    project_root = Path(__file__).parent
    model_path = project_root / 'models' / 'diagnosis_classifier.h5'
    encoder_path = project_root / 'models' / 'label_encoder.pkl'
    
    # Verificar se modelo existe
    if not model_path.exists() or not encoder_path.exists():
        print('Modelo não encontrado. Execute diagnosis_classifier.ipynb primeiro.')
        return
    
    # Inicializar preditor
    predictor = DiagnosisPredictor(str(model_path), str(encoder_path))
    
    # Encontrar um espectrograma para testar
    spec_dir = project_root / 'processed_audio' / 'spectrograms'
    if spec_dir.exists():
        spec_files = list(spec_dir.glob('*.npy'))
        
        if spec_files:
            spec_file = spec_files[0]
            print(f'\n=== Predição a partir de Espectrograma ===')
            print(f'Arquivo: {spec_file.name}')
            
            # Fazer predição
            diagnosis, confidence, probs = predictor.predict_from_spectrogram(str(spec_file))
            
            if diagnosis:
                print(f'\nResultado:')
                print(f'  Diagnóstico: {diagnosis}')
                print(f'  Confiança: {confidence:.2%}')
                print(f'\n  Probabilidades para cada classe:')
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f'    {class_name}: {prob:.2%}')
                
                # Visualizar
                print(f'\nGerando visualização...')
                predictor.visualize_prediction(str(spec_file), save_path='prediction_example.png')


def example_predict_from_audio():
    """Exemplo: Predizer diagnóstico a partir de um arquivo de áudio."""
    
    project_root = Path(__file__).parent
    model_path = project_root / 'models' / 'diagnosis_classifier.h5'
    encoder_path = project_root / 'models' / 'label_encoder.pkl'
    
    # Verificar se modelo existe
    if not model_path.exists() or not encoder_path.exists():
        print('Modelo não encontrado. Execute diagnosis_classifier.ipynb primeiro.')
        return
    
    # Inicializar preditor
    predictor = DiagnosisPredictor(str(model_path), str(encoder_path))
    
    # Encontrar um arquivo de áudio para testar
    audio_dir = project_root / 'processed_audio' / 'audio'
    if audio_dir.exists():
        audio_files = list(audio_dir.glob('*.wav'))
        
        if audio_files:
            audio_file = audio_files[0]
            print(f'\n=== Predição a partir de Áudio ===')
            print(f'Arquivo: {audio_file.name}')
            
            # Fazer predição
            diagnosis, confidence, probs = predictor.predict_from_audio(str(audio_file))
            
            if diagnosis:
                print(f'\nResultado:')
                print(f'  Diagnóstico: {diagnosis}')
                print(f'  Confiança: {confidence:.2%}')
                print(f'\n  Probabilidades para cada classe:')
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f'    {class_name}: {prob:.2%}')
                
                # Visualizar
                print(f'\nGerando visualização...')
                predictor.visualize_prediction(str(audio_file), is_audio=True, save_path='audio_prediction_example.png')


if __name__ == '__main__':
    print('Exemplos de uso do clasificador de diagnóstico')
    print('=' * 60)
    
    # Tentar predição com espectrograma
    example_predict_from_spectrogram()
    
    # Tentar predição com áudio
    example_predict_from_audio()
    
    print('\n' + '=' * 60)
    print('Exemplos completados!')
