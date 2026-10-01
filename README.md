# Checkpoint 5 — Previsão de consumo de veículos

Projeto de Data Science & Statistical Computing — FIAP, 2026.
Comparação de Random Forest, XGBoost e LightGBM na base Auto MPG, com Grid Search, Optuna e interface Streamlit.

## Identificação e links

| Integrante | RM |
| --- | --- |
| Rafael Joda | **PREENCHER** |

Acrescentar os demais integrantes, se houver. O grupo pode ter até cinco alunos.

- GitHub: **PENDENTE DE PUBLICAÇÃO**
- Streamlit: **PENDENTE DE PUBLICAÇÃO**
- Preencher também `URL_GITHUB` e `URL_STREAMLIT` no Exercício 7 do notebook.

## Problema

Estimar a eficiência em MPG (milhas por galão americano) a partir de cilindros, cilindrada,
potência, peso, aceleração, ano-modelo e origem. Quanto maior o MPG, maior a distância por
unidade de combustível. É regressão supervisionada, com valores medidos de consumo como alvo.
O aplicativo mostra também km/L, usando `MPG × 0.425143707`; a métrica de treinamento continua em MPG.

## Dados e atribuição

Quinlan, R. (1993). **Auto MPG**. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5859H
Página: https://archive.ics.uci.edu/dataset/9/auto+mpg
Arquivo oficial: https://archive.ics.uci.edu/static/public/9/auto+mpg.zip

398 observações, nove colunas (sete atributos, alvo e nome do carro), anos-modelo 1970–1982.
A distribuição oficial possui seis ausências em potência. `?` é convertido em valor ausente;
a mediana é aprendida exclusivamente no treino de cada fold. Não se removem essas linhas.
A cópia original está em `data/auto-mpg.data`, acompanhada de `auto-mpg.names`.
Licença dos dados informada pela fonte: **CC BY 4.0**. Os dados são redistribuídos sem alteração;
a preparação acontece no notebook. SHA-256: `48b830e11feee5572525f8f1691ddb9d38d3d7b7063edcd8fca672c2a5e17d8d`.

## Protocolo

1. Reservar 20% antes da exploração: **318 treino / 80 teste**, semente 42.
2. Explorar e tomar decisões apenas no treino; excluir `car_name` dos preditores.
3. Usar pipeline com imputação pela mediana e one-hot encoding de origem.
4. Comparar os modelos nos mesmos cinco folds KFold embaralhados, semente 42.
5. Executar 8 combinações Grid Search e **20 trials Optuna por modelo**.
6. Escolher pelo menor MAE médio de CV, sem consultar o teste; em empate, menor desvio-padrão e tempo de ajuste.
7. Ajustar em todo o treino, avaliar o teste uma vez e exportar o pipeline completo.

MAE é a métrica principal; RMSE e R² são auxiliares. O desvio-padrão da CV descreve
variação entre folds, não intervalo de confiança. Há uma referência adicional constante
(média de treino por fold), com MAE CV de aproximadamente 6,702 MPG.

## Resultados executados

| Modelo | Estratégia | MAE treino | MAE CV | DP CV |
| --- | --- | ---: | ---: | ---: |
| XGBoost | Optuna | 0.495 | 1.948 | 0.152 |
| LightGBM | GridSearch | 0.792 | 1.956 | 0.258 |
| LightGBM | Baseline | 0.589 | 1.958 | 0.274 |
| XGBoost | Baseline | 0.938 | 1.959 | 0.090 |
| XGBoost | GridSearch | 1.243 | 1.993 | 0.113 |
| LightGBM | Optuna | 0.753 | 2.011 | 0.230 |
| Random Forest | Optuna | 0.763 | 2.013 | 0.104 |
| Random Forest | Baseline | 0.770 | 2.033 | 0.107 |
| Random Forest | GridSearch | 0.772 | 2.035 | 0.108 |

**Selecionado:** XGBoost / Optuna.

| Métrica | Teste reservado |
| --- | ---: |
| MAE | 1.7419 MPG |
| RMSE | 2.3784 MPG |
| R² | 0.8948 |

A diferença de CV para o segundo colocado é de apenas 0.0075 MPG.
Há sobreajuste: MAE de treino 0.495 versus CV 1.948.
A curva melhora com mais dados e o teste permanece na mesma ordem de erro da validação.
Não interpretamos o ranking como superioridade estatisticamente comprovada.
Os estudos usam orçamentos e espaços distintos; os tempos medidos valem para o ambiente da execução.

## Instalar e abrir

Ambiente verificado: **Python 3.12**. Dentro da pasta extraída:

```bash
python -m venv .venv
```

No Windows, ative com `.venv\Scripts\activate`; no macOS/Linux, com `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O pipeline já está treinado em `artifacts/pipeline.joblib`. Não é preciso retreinar para abrir o app.
Use as versões fixadas: carregar um artefato com versões incompatíveis pode alterar o comportamento.

## Reproduzir o treinamento

Abra `Checkpoint05_RF_XGBoost_LightGBM.ipynb` no Jupyter/Colab e execute todas as células,
ou rode:

```bash
python treinar.py
python verificar_paridade.py
```

`treinar.py` executa as células sequencialmente, interrompe se houver erro e salva saídas no notebook.
Os modelos, resultados e gráficos são regenerados. O notebook baixa a base oficial se não encontrar
`data/auto-mpg.data`; com o projeto completo, usa a cópia incluída.
O treinamento realiza centenas de ajustes pequenos. A duração depende da máquina.
O teste de paridade verifica as entradas e saídas de cinco exemplos pela API AppTest do Streamlit,
incluindo o valor exibido arredondado. Resultado em `results/paridade_streamlit.json`.

## Publicar no GitHub e Streamlit

1. Criar um repositório GitHub, por exemplo `checkpoint05-auto-mpg`.
2. Enviar **o conteúdo da pasta extraída**, incluindo `data`, `artifacts`, `results`, notebook,
   `app.py`, scripts e `requirements.txt`. Não enviar apenas o ZIP.
3. No [Streamlit Community Cloud](https://share.streamlit.io/), conectar o GitHub e criar uma aplicação.
4. Selecionar repositório, branch e arquivo de entrada `app.py`; usar Python 3.12 nas configurações disponíveis.
5. Publicar e testar um exemplo de consistência na interface hospedada.
6. Registrar os links reais aqui e no notebook. A hospedagem ainda não foi realizada nesta entrega de arquivos.

Documentação oficial: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Arquivos principais

| Caminho | Conteúdo |
| --- | --- |
| `Checkpoint05_RF_XGBoost_LightGBM.ipynb` | Sete exercícios, código, gráficos e interpretações com saídas reais |
| `app.py` | Interface Streamlit |
| `treinar.py` | Execução sequencial do notebook e geração dos artefatos |
| `verificar_paridade.py` | Verificação notebook–interface nos cinco exemplos |
| `requirements.txt` | Dependências fixadas |
| `artifacts/` | Pipeline e metadados, parâmetros e exemplos |
| `data/` | Base original e descrição da UCI |
| `results/` | Tabelas dos nove modelos, buscas, folds, gráficos, split e previsões |
| `ROTEIRO_APRESENTACAO.md` | Roteiro de até dez minutos e perguntas para estudo |
| `CHECKLIST_ENTREGA.md` | Requisitos concluídos e pendências |

## Limitações

Base pequena e histórica; não há validação para carros atuais, outros combustíveis ou uso rodoviário.
Fatores como manutenção e modo de condução não são observados. O mesmo esquema de CV seleciona
hiperparâmetros e modelo, gerando potencial otimismo; o teste reservado fornece a avaliação final.
Uma partição não garante estabilidade em outras amostras. Importância de variáveis não implica causalidade.
Não foram calculados intervalos de previsão individuais.
