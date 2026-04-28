#!/usr/bin/env python
"""
Validação Rápida de Dependências e Setup

Execute este script para verificar se tudo está configurado corretamente.
"""

import sys
from pathlib import Path


def check_python_version():
    """Verificar versão do Python."""
    print("✓ Verificando versão do Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} OK")
        return True
    else:
        print(f"  ✗ Python 3.8+ requerido. Você tem {version.major}.{version.minor}")
        return False


def check_dependencies():
    """Verificar dependências instaladas."""
    print("\n✓ Verificando dependências...")
    
    deps = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'librosa': 'Librosa',
        'scipy': 'SciPy',
        'tensorflow': 'TensorFlow'
    }
    
    missing = []
    
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name} instalado")
        except ImportError:
            print(f"  ✗ {name} NÃO instalado")
            missing.append(module)
    
    if missing:
        print(f"\n  ⚠️  Instale dependências faltantes com:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_project_structure():
    """Verificar estrutura de diretórios."""
    print("\n✓ Verificando estrutura do projeto...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        'notebooks',
        'processed_audio',
    ]
    
    required_files = [
        'process_audio.py',
        'enrich_metadata.py',
        'predict_diagnosis.py',
        'example_usage.py',
        'README.md'
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ Diretório '{dir_name}' encontrado")
        else:
            print(f"  ✗ Diretório '{dir_name}' NÃO encontrado")
            all_ok = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✓ Arquivo '{file_name}' encontrado")
        else:
            print(f"  ✗ Arquivo '{file_name}' NÃO encontrado")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Verificar se dados existem."""
    print("\n✓ Verificando dados...")
    
    project_root = Path(__file__).parent
    metadata_path = project_root / 'processed_audio' / 'metadata.csv'
    
    if metadata_path.exists():
        print(f"  ✓ metadata.csv encontrado")
        
        # Tentar carregar e contar linhas
        try:
            import pandas as pd
            df = pd.read_csv(metadata_path)
            print(f"    • {len(df)} linhas")
            print(f"    • Colunas: {', '.join(df.columns[:5].tolist())}{' ...' if len(df.columns) > 5 else ''}")
            
            if 'diagnosis' in df.columns:
                print(f"    ✓ Coluna 'diagnosis' encontrada")
                diagnoses = df['diagnosis'].nunique()
                print(f"    • {diagnoses} diagnósticos únicos")
            else:
                print(f"    ⚠️  Coluna 'diagnosis' NÃO encontrada")
            
            return True
        except Exception as e:
            print(f"    ✗ Erro ao ler metadata.csv: {e}")
            return False
    else:
        print(f"  ✗ metadata.csv NÃO encontrado")
        print(f"    → Execute process_audio.py primeiro")
        return False


def check_spectrograms():
    """Verificar se espectrogramas existem."""
    print("\n✓ Verificando espectrogramas...")
    
    project_root = Path(__file__).parent
    spec_dir = project_root / 'processed_audio' / 'spectrograms'
    
    if spec_dir.exists():
        spec_files = list(spec_dir.glob('*.npy'))
        if spec_files:
            print(f"  ✓ Pasta 'spectrograms' encontrada")
            print(f"    • {len(spec_files)} arquivos .npy")
            
            # Tentar carregar um
            try:
                import numpy as np
                sample = np.load(spec_files[0])
                print(f"    • Shape: {sample.shape}")
                return True
            except Exception as e:
                print(f"    ✗ Erro ao carregar espectrogram: {e}")
                return False
        else:
            print(f"  ✗ Nenhum arquivo .npy em spectrograms/")
            return False
    else:
        print(f"  ✗ Diretório 'spectrograms' NÃO encontrado")
        print(f"    → Execute process_audio.py primeiro")
        return False


def check_models():
    """Verificar se modelos treinados existem."""
    print("\n✓ Verificando modelos treinados...")
    
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    
    if models_dir.exists():
        model_file = models_dir / 'diagnosis_classifier.h5'
        encoder_file = models_dir / 'label_encoder.pkl'
        
        if model_file.exists():
            print(f"  ✓ Modelo encontrado: {model_file.name}")
            print(f"    • Tamanho: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  ⚠️  Modelo NÃO encontrado")
            print(f"    → Execute diagnosis_classifier.ipynb primeiro")
            return False
        
        if encoder_file.exists():
            print(f"  ✓ Encoder encontrado: {encoder_file.name}")
            return True
        else:
            print(f"  ⚠️  Encoder NÃO encontrado")
            return False
    else:
        print(f"  ⚠️  Diretório 'models' NÃO encontrado (normal se não treinou ainda)")
        return False


def main():
    """Executar todas as verificações."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      Validação de Setup - AI Medicine Diagnóstico       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependências", check_dependencies),
        ("Estrutura do Projeto", check_project_structure),
        ("Dados (metadata.csv)", check_data_files),
        ("Espectrogramas", check_spectrograms),
        ("Modelos Treinados", check_models),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Erro ao verificar {name}: {e}")
            results.append((name, False))
    
    # Resumo
    print("\n" + "═" * 60)
    print("RESUMO")
    print("═" * 60)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print("═" * 60)
    
    # Status final
    all_ok = all(result for _, result in results)
    
    if all_ok:
        print("\n✅ Tudo pronto! Execute:")
        print("   jupyter notebook notebooks/diagnosis_classifier.ipynb")
    else:
        essential = results[0][1] and results[1][1] and results[2][1]
        if essential:
            print("\n⚠️  Você precisa preparar os dados primeiro:")
            print("   1. Coloque arquivos de áudio em uma pasta 'data'")
            print("   2. Execute: python process_audio.py")
            print("   3. Execute: python enrich_metadata.py")
            print("   4. Execute novamente: python validation.py")
        else:
            print("\n❌ Há problemas críticos. Verifique acima.")
    
    return 0 if all_ok or essential else 1


if __name__ == '__main__':
    sys.exit(main())
