# Roteiro de apresentação — até 10 minutos

Tempo planejado: **9 minutos**, deixando 1 minuto de margem.
Divida os blocos entre os integrantes depois de preencher os nomes e RMs.
Os trechos abaixo são apoio: ensaie explicando os gráficos, sem ler o notebook inteiro.

## 0:00–1:00 · Problema e dados

“Nosso projeto prevê a eficiência de combustível de veículos. A pergunta é: conhecendo
características como peso, potência, cilindrada e ano, quanto esse veículo percorre por galão?
Utilizamos a Auto MPG, uma base pública da UCI com 398 registros históricos.
O alvo é MPG: quanto maior o valor, mais eficiente é o carro. Como prevemos um número,
o problema é de regressão. A proposta é acadêmica e não foi validada para carros atuais.”

**Mostrar:** definição do problema e dicionário de dados do Exercício 1.

## 1:00–2:00 · Preparação e exploração

“A potência tinha valores ausentes, representados por interrogação. Convertimos para ausentes
e usamos a mediana, calculada dentro do pipeline a cada treinamento.
Mantivemos os carros com potência desconhecida. Excluímos o nome do carro dos preditores
e codificamos a origem em categorias separadas.
O gráfico de peso mostra associação negativa com MPG. Os carros mais pesados tendem a ser
menos eficientes nessa amostra. Os gráficos por ano também mostram mudança na eficiência,
mas essas relações não provam causalidade.”

**Mostrar:** gráfico peso–MPG e boxplots por ano. Explique uma relação concreta observada.

## 2:00–3:00 · Protocolo experimental

“Reservamos 80 registros para teste antes de explorar os dados. Os 318 restantes foram usados
no desenvolvimento, com validação cruzada de cinco folds iguais para todos os modelos.
A métrica principal é MAE, que indica o erro absoluto médio em MPG. Também usamos RMSE e R².
Imputação e codificação são aprendidas apenas no treino de cada fold. Isso impede vazamento
das informações usadas para validar. O teste ficou reservado até a escolha final.”

**Mostrar:** protocolo do Exercício 3 e tabela dos folds.

## 3:00–4:00 · Três modelos iniciais

“Random Forest combina árvores treinadas com aleatoriedade. XGBoost e LightGBM usam boosting,
em que árvores sucessivas trabalham para melhorar as previsões do conjunto.
Começamos com configurações explícitas e comparamos treino, validação, dispersão e tempo.
Também calculamos uma referência simples que sempre prevê a média: ela teve MAE de cerca
de 6,7 MPG. Os três algoritmos ficaram perto de 2 MPG na validação.”

**Mostrar:** tabela dos baselines. Não diga que R² significa percentual de acertos.

## 4:00–5:10 · Grid Search e Optuna

“Aplicamos as duas estratégias aos três algoritmos. O Grid Search avaliou oito combinações
por algoritmo. O Optuna executou vinte tentativas por algoritmo, propondo configurações
a partir dos resultados anteriores. Foram utilizados os mesmos folds e a mesma métrica.
Os parâmetros controlam número e complexidade das árvores, taxa de aprendizado e regularização.
Os orçamentos e espaços são diferentes, então não podemos afirmar que um método sempre é melhor.
No LightGBM, por exemplo, Optuna não superou a grade nem o modelo inicial.”

**Mostrar:** histórico do Optuna e tabela das buscas. Os tempos estão registrados em segundos.

## 5:10–6:20 · Escolha e generalização

“Comparamos nove configurações. Pela regra de menor MAE médio, o escolhido foi
XGBoost com Optuna, com 1.948 MPG na validação.
A diferença para o segundo foi só 0.0075 MPG,
então não é uma superioridade comprovada. Existe sobreajuste: o erro no treino ficou
em 0.495, bem abaixo da validação. A curva mostra melhora quando aumentamos
a quantidade de dados, mas o gap continua. Não é o padrão de underfitting, em que ambos
os erros seriam altos. Mantivemos a escolha feita antes de abrir o teste.”

**Mostrar:** comparação dos nove modelos, curva de aprendizado e importância das variáveis.
A maior importância média foi de **model_year**. Ela mede contribuição preditiva, não causalidade.

## 6:20–7:20 · Teste final e casos concretos

“Ajustamos o modelo escolhido em todo o treino e avaliamos uma vez os 80 registros reservados.
O MAE foi 1.742 MPG, o RMSE 2.378 e o R² 0.895.
O erro ficou na mesma ordem de grandeza da validação. Isso apoia a generalização para essa
população histórica, sem garantir o resultado em outras populações.
Nos cinco exemplos, mostramos tanto um erro muito pequeno quanto um caso difícil.
O maior erro foi no VW Pickup: real de 44 MPG, previsão de aproximadamente 36 MPG.
Isso mostra por que a média de erro não é garantia para cada veículo.”

**Mostrar:** tabela das cinco observações, sem esconder o pior caso.

## 7:20–8:30 · Demonstrar o Streamlit

1. Abrir o aplicativo e selecionar o primeiro exemplo, AMC Concord D/L.
2. Clicar em **Estimar consumo**.
3. Mostrar previsão em MPG, conversão para km/L e valor real do exemplo.
4. Apontar a diferença notebook–interface igual a zero dentro da tolerância numérica.
5. Abrir a aba de comparação para mostrar os nove resultados.

“O aplicativo carrega exatamente o pipeline salvo no notebook. Não refaz o treinamento
nem calcula outra preparação. Verificamos os cinco exemplos com o AppTest do Streamlit,
comparando a previsão calculada e o valor exibido. A mesma entrada mantém a mesma saída.”

## 8:30–9:00 · Conclusão

“O projeto compara três algoritmos com um protocolo comum e duas buscas por algoritmo.
O resultado final tem erro médio de 1.742 MPG no teste reservado, mas a base é pequena,
histórica e não representa veículos atuais. Em uma próxima etapa, buscaríamos dados mais
recentes e avaliação adequada ao uso real. O repositório contém código, dados atribuídos,
dependências e o pipeline usado pela interface.”

## Perguntas para estudar antes da apresentação

- **Por que MAE?** É interpretável em MPG e não amplifica erros grandes tanto quanto RMSE.
- **O que é vazamento?** Usar informação do teste ou da validação para aprender etapas do treino.
- **Por que pipeline?** Ajusta imputação/codificação somente nas partições permitidas e mantém a inferência consistente.
- **Por que não normalizar?** Árvores fazem divisões por limiares e não dependem de distâncias entre atributos.
- **Por que excluir nome?** É um rótulo textual de alta cardinalidade; o objetivo usa especificações disponíveis.
- **Qual a diferença entre treino, validação e teste?** Treino ajusta; CV orienta desenvolvimento; teste avalia a escolha congelada.
- **Optuna sempre melhora?** Não. Depende do espaço, orçamento e ruído da validação; nossos resultados mostram isso.
- **R² de 0.895 é 89.5% de acerto?** Não. R² é uma comparação do erro quadrático com a referência baseada na média.
- **Tem overfitting?** Sim, há evidências pelo gap; a curva e o teste contextualizam seu impacto.
- **Por que o teste foi melhor que a CV?** A dificuldade das amostras varia e o modelo final usa mais dados de treino que cada fold.
- **A importância prova causa?** Não, mede contribuição à previsão e pode ser afetada por correlação entre atributos.
- **Podemos usar no Captur atual?** Não foi validado para essa população; os dados são históricos.

Antes de apresentar: preencher identificação, publicar os links reais e ensaiar a demonstração.
