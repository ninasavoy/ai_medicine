# ai_medicine


## Evolução semanal
### Sprint 1 — Definição e Exploração Inicial

- Definição do escopo: classificação de doenças respiratórias a partir de sons pulmonares
- Escolha da abordagem de converter áudio em espectrogramas para alimentar uma CNN — decisão que permite tratar o problema de áudio como classificação de imagem
- Geração dos primeiros espectrogramas a partir dos arquivos de áudio brutos
- Implementação do pipeline inicial de pré-processamento de áudio

### Sprint 2 — Construção do Dataset

- Análise exploratória dos dados (EDA): distribuição de classes, duração dos áudios, qualidade dos registros
- Definição do formato final do dataset (espectrogramas rotulados por condição respiratória)
- Implementação de uma CNN básica como baseline — sem otimizações, apenas para validar o pipeline de ponta a ponta

Sprint 3 — Primeiros Resultados e Agregação

- Decisão de agregar predições por paciente (em vez de por fragmento de áudio), tornando o resultado clinicamente mais interpretável
- Ajustes na arquitetura da CNN com base nos resultados iniciais
- Avaliação das primeiras métricas (acurácia, F1-score, matriz de confusão)

### Sprint 4 — Refinamento do Modelo

- Ajuste de hiperparâmetros (learning rate, batch size, número de camadas)
- Tratamento do desbalanceamento de classes: decisão entre técnicas como oversampling, undersampling ou pesos por classe na loss function
- Melhorias na arquitetura (ex: adição de Dropout, BatchNorm, ou transfer learning)

### Sprint 5 — Avaliação Final

- Avaliação final com métricas consolidadas por paciente
- Análise crítica dos resultados e visualizações (curvas ROC, mapas de ativação, etc.)
- Documentação completa e conclusões do projeto

## Pipeline

1. Segmentação dos Áudios

Os áudios são divididos em ciclos respiratórios individuais utilizando as anotações fornecidas no dataset:

- Início do ciclo
- Fim do ciclo

Cada ciclo passa a ser uma amostra independente.

2. Conversão para Espectrogramas

Cada ciclo de áudio é transformado em um Mel Spectrogram, que representa o sinal no domínio tempo-frequência.

Etapas:

- Carregamento do áudio
- Recorte do ciclo
- Geração do espectrograma
- Conversão para escala logarítmica (dB)

Esses espectrogramas são utilizados como entrada da CNN.

3. Extração de Features com CNN

Uma rede neural convolucional é utilizada para extrair padrões relevantes dos espectrogramas.

A CNN aprende automaticamente características como:

- padrões de frequência
- intensidade do som
- variações temporais

Saída da CNN:

Vetor de features (representação do ciclo)

4. Agregação por Paciente

Como cada paciente possui vários ciclos, é necessário combinar essas informações.

Método utilizado:

- Média das features dos ciclos

Alternativas possíveis:

- Média das probabilidades
- Votação
- Modelos sequenciais (não utilizado nesta versão)

5. Classificação Final

Após a agregação, cada paciente é representado por um único vetor.
Esse vetor é utilizado para prever o diagnóstico:

Classes possíveis incluem:

- COPD
- Pneumonia
- URTI
- Healthy
- entre outras

O classificador final pode ser:

- MLP (rede densa)
- ou camada final da própria CNN

## Considerações Importantes
### Data Leakage

A divisão entre treino e teste é feita por paciente, não por ciclo, para evitar vazamento de informação.

### Desbalanceamento

O dataset possui classes desbalanceadas (ex: muitos casos de COPD).

Soluções utilizadas:

- Class weights
- Avaliação com métricas além de accuracy

### Limitações

- Nem todo ciclo contém informação suficiente sobre a doença
- Quantidade de dados limitada
- Presença de ruído nos áudios

## Resumo da Arquitetura
Áudio → Segmentação → Espectrograma → CNN → Features → Agregação → Diagnóstico