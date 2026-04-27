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

---

## Complemento — Pipeline implementado até o momento

Além da visão geral acima, o pipeline atualmente implementado nos scripts inclui:

1. **Pareamento automático de arquivos** (`.wav` + `.txt`) com validações de integridade.
2. **Leitura robusta das anotações de ciclo** (`start`, `end`, `crackles`, `wheezes`) e descarte seguro de linhas inválidas.
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
3. Mitigar desbalanceamento com `class weights`, possíveis estratégias de reamostragem e/ou `focal loss`.
4. Definir política de tratamento de outliers e repetir EDA após limpeza para medir impacto.
5. Evoluir arquitetura (regularização, ajustes de hiperparâmetros e data augmentation em espectrograma/áudio).
6. Consolidar reprodutibilidade (seeds, versionamento de artefatos e rastreio de experimentos).
7. Expandir interpretabilidade e análise de erro para guiar melhorias por classe diagnóstica.