# AI Medicine - Respiratory Sound Diagnosis Classifier

Classificação de diagnósticos respiratórios (COPD, Pneumonia, URTI, etc.) a partir de sons pulmonares usando CNN.


### Sprint 1 — Definição e Exploração Inicial

- Definição do escopo: classificação de doenças respiratórias a partir de sons pulmonares
- Escolha da abordagem de converter áudio em espectrogramas para alimentar uma CNN — decisão que permite tratar o problema de áudio como classificação de imagem
- Geração dos primeiros espectrogramas a partir dos arquivos de áudio brutos
- Implementação do pipeline inicial de pré-processamento de áudio

### Sprint 2 — Construção do Dataset

- Análise exploratória dos dados (EDA): distribuição de classes, duração dos áudios, qualidade dos registros
- Definição do formato final do dataset (espectrogramas rotulados por condição respiratória)
- Implementação de uma CNN básica como baseline — sem otimizações, apenas para validar o pipeline de ponta a ponta (Fizemos outro modelo de baseline)

### Sprint 3 — Primeiros Resultados e Agregação

- CNN Baseline
- Decisão de agregar predições por paciente (em vez de por fragmento de áudio), tornando o resultado clinicamente mais interpretável
- Ajustes na arquitetura da CNN com base nos resultados iniciais
- Avaliação das primeiras métricas (acurácia, F1-score, matriz de confusão)

### Sprint 4 — Refinamento do Modelo <-- Estamos aqui

- Ajuste de hiperparâmetros (learning rate, batch size, número de camadas)
- Tratamento do desbalanceamento de classes: decisão entre técnicas como oversampling, undersampling ou pesos por classe na loss function
- Melhorias na arquitetura (ex: adição de Dropout, BatchNorm, ou transfer learning)

### Sprint 5 — Avaliação Final

- Avaliação final com métricas consolidadas por paciente
- Análise crítica dos resultados e visualizações (curvas ROC, mapas de ativação, etc.)
- Documentação completa e conclusões do projeto

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


## Estrutura

```
ai_medicine/
├── notebooks/
│   ├── eda.ipynb                  # Análise exploratória
│   └── diagnosis_classifier.ipynb # testando um modelo
├── processed_audio/
│   ├── spectrograms/              # Espectrogramas (.npy)
│   ├── audio/                     # Áudios processados
│   └── metadata.csv               # Metadados
├── process_audio.py               # Processamento de áudio (Criando espectrogramas)
├── enrich_metadata.py             # Enriquecer metadados (split, diagnostico etc)
└── predict_diagnosis.py           # testando
```

## Modelo

**Arquitetura CNN:**
- Em construcao

**Treinamento:**
- A decidir

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
	- split `train/validation/test` por paciente e usando stratify,
	- validações finais de consistência (diagnóstico único/split único por paciente e paths válidos).


## EDA (etapas, motivos e resultados)

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

11. **Modelo baseline (EDA)**
	- Extraimos features simples dos espectrogramas (estatisticas e/ou downsample) e treinamos modelos classicos.
	- Avaliacao em nivel de ciclo e agregacao por paciente (media das probabilidades) para obter o diagnostico final.
	- Comparacao entre Logistic Regression e Random Forest usando F1 macro por paciente para selecionar o melhor modelo.
	- Visualizacoes: distribuicao de classes, matrizes de confusao e confianca por paciente.

## Baseline CNN (TensorFlow)

- CNN simples em espectrogramas (128x128) com 3 blocos Conv + MaxPool, Flatten, Dense e Dropout.
- Treino com `SparseCategoricalCrossentropy` e pesos de classe para lidar com desbalanceamento.
- Metricas por epoca: loss, accuracy e F1 macro no conjunto de validacao.
- Avaliacao final por paciente com agregacao de probabilidades (media por `patient_id`).





