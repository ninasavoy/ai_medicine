# AI Medicine - Respiratory Sound Diagnosis Classifier

Classificação de diagnósticos respiratórios (COPD, Pneumonia, URTI, etc.) a partir de sons pulmonares usando CNN.

## Pipeline

```
Audio (.wav) → Mel Spectrogram → CNN 2D → Diagnóstico
```

**Fluxo:**
1. Segmentação de áudios em ciclos respiratórios
2. Conversão para Mel Spectrograms (128x128)
3. Treinamento de CNN com 4 blocos convolucionais
4. Agregação por paciente (média de probabilidades)
5. Predição de diagnóstico final

## Quick Start

### Treinar Modelo
```bash
jupyter notebook notebooks/diagnosis_classifier.ipynb
```

### Fazer Predição
```bash
# Com spectrogram
python predict_diagnosis.py processed_audio/spectrograms/sample.npy --visualize

# Com áudio direto
python predict_diagnosis.py audio.wav --audio --visualize
```

## Estrutura

```
ai_medicine/
├── notebooks/
│   ├── eda.ipynb                  # Análise exploratória
│   └── diagnosis_classifier.ipynb # Treinamento do modelo
├── processed_audio/
│   ├── spectrograms/              # Espectrogramas (.npy)
│   ├── audio/                     # Áudios processados
│   └── metadata.csv               # Metadados
├── models/
│   ├── diagnosis_classifier.h5    # Modelo treinado
│   └── label_encoder.pkl          # Encoder de classes
├── process_audio.py               # Processamento de áudio
├── enrich_metadata.py             # Enriquecer metadados
└── predict_diagnosis.py           # Fazer predições
```

## Modelo

**Arquitetura CNN:**
- 4 blocos de convolução (32 → 64 → 128 → 256 filtros)
- Batch Normalization + Dropout em cada bloco
- MaxPooling2D entre blocos
- Dense layers (512 → 256 → num_classes)

**Treinamento:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Split: 60% train, 20% val, 20% test

## Métricas

- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC Curves (One vs Rest)
- Test Accuracy

## Considerações

- **Data Leakage:** Split por paciente (não por ciclo)
- **Desbalanceamento:** Classes desbalanceadas no dataset
- **Limitações:** Quantidade limitada de dados, presença de ruído
3. **Padronização do áudio**: mono, `22050 Hz`, normalização de amplitude e duração fixa de `5s` por ciclo.
4. **Geração padronizada de Mel-spectrograma** com shape final fixo `128x128` e normalização para `[0,1]`.
5. **Persistência por paciente** em `.npy` + geração de `metadata.csv` no nível de ciclo.
6. **Enriquecimento do metadata** com:
	- diagnóstico por paciente (`diagnosis.txt`),
	- coluna `duration`,
	- split `train/validation/test` por paciente,
	- validações finais de consistência (diagnóstico único/split único por paciente e paths válidos).

**Resumo prático:** hoje a base já está pronta e consistente para treino supervisionado com prevenção explícita de leakage por paciente.

## Complemento — EDA (etapas, motivos e resultados)

As etapas executadas no notebook de EDA e seus objetivos são:

1. **Carga e checagem inicial**
	- **Motivo:** confirmar caminhos e leitura correta do metadata.
	- **Resultado:** dataset carregado e ambiente validado.

2. **Visão geral do dataset**
	- **Motivo:** medir volume de ciclos/pacientes e faltantes.
	- **Resultado:** estatísticas gerais e mapa de valores ausentes.

3. **Distribuição de labels**
	- **Motivo:** analisar prevalência de `crackles`, `wheezes` e `diagnosis`.
	- **Resultado:** identificação de desbalanceamento de classes.

4. **Verificação de split e leakage**
	- **Motivo:** garantir que paciente não aparece em múltiplos splits.
	- **Resultado:** validação da separação por paciente para treino/validação/teste.

5. **Análise de ciclos**
	- **Motivo:** entender duração dos ciclos e ciclos por paciente.
	- **Resultado:** distribuição de duração e variabilidade entre pacientes.

6. **Inspeção visual de espectrogramas**
	- **Motivo:** validar qualidade visual e possíveis padrões por classe.
	- **Resultado:** amostras coerentes com o pipeline de pré-processamento.

7. **Validação avançada de qualidade**
	- **Motivo:** detectar problemas silenciosos de dados.
	- **Resultado:** checagens de arquivos faltantes/corrompidos, shape, faixa de valores, duplicatas e consistência de split.

8. **Análise de separabilidade (PCA)**
	- **Motivo:** observar sinal discriminativo inicial no espaço de features.
	- **Resultado:** projeções 2D para análise qualitativa entre diagnósticos e labels respiratórios.

9. **Detecção de outliers**
	- **Motivo:** identificar ciclos/pacientes extremos que podem enviesar treino.
	- **Resultado:** lista de outliers por duração e pacientes com número extremo de ciclos.

10. **Síntese de insights**
	- **Motivo:** transformar EDA em decisões de modelagem.
	- **Resultado:** direcionamento para balanceamento, robustez e avaliação por paciente.

## Complemento — Próximos passos com base nos resultados

1. Treinar baseline por ciclo com métricas robustas (F1 macro, matriz de confusão por classe).
2. Consolidar avaliação principal por paciente (agregação de probabilidades/features por paciente).
3. Mitigar desbalanceamento com `class weights`, `focal loss` e possíveis estratégias de reamostragem.
4. Definir política de tratamento de outliers e repetir EDA após limpeza para medir impacto.
5. Evoluir arquitetura (regularização com dropout, ajustes de hiperparâmetros e data augmentation em espectrograma/áudio).
6. Consolidar reprodutibilidade (seeds, versionamento de artefatos e rastreio de experimentos).
7. Expandir interpretabilidade e análise de erro para guiar melhorias por classe diagnóstica.



