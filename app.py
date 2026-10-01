"""Interface de inferência: usa exclusivamente o pipeline exportado no notebook."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MPG_PARA_KM_L = 0.425143707
st.set_page_config(page_title='Auto MPG · Consumo de veículos', layout='wide')


@st.cache_resource
def carregar():
    modelo = joblib.load(ROOT / 'artifacts' / 'pipeline.joblib')
    meta = json.loads((ROOT / 'artifacts' / 'metadata.json').read_text())
    return modelo, meta


try:
    modelo, meta = carregar()
except FileNotFoundError:
    st.error('Artefatos ausentes. Execute o notebook completo ou python treinar.py antes de abrir o aplicativo.')
    st.stop()

st.caption('FIAP 2026  /  CHECKPOINT 05  /  REGRESSÃO')
st.title('Quanto esse carro percorre?')
st.write('Explore a relação entre as características de um veículo e sua eficiência de combustível.')
st.info('Demonstração com veículos de 1970–1982. Os resultados não foram validados para carros atuais, '
        'outros combustíveis ou consumo rodoviário.')

simular, comparar, metodo = st.tabs(['Simular consumo', 'Comparar modelos', 'Entender o projeto'])
with simular:
    casos = meta['casos_consistencia']

    def preencher_caso():
        caso = casos[st.session_state.get('example_idx', 0)]
        for campo in ['cylinders','displacement','weight','acceleration','origin']:
            st.session_state[campo] = int(caso[campo]) if campo in ['cylinders','origin'] else float(caso[campo])
        st.session_state['year'] = int(caso['model_year']) + 1900
        st.session_state['hp_missing'] = caso['horsepower'] is None
        st.session_state['horsepower'] = float(caso['horsepower']) if caso['horsepower'] is not None else 100.0
        st.session_state.pop('prediction_mpg', None)

    if 'weight' not in st.session_state:
        preencher_caso()
    st.selectbox('Carregar um exemplo do teste', options=list(range(len(casos))),
                 format_func=lambda i: f"Caso {i+1} · {casos[i]['car_name']}",
                 key='example_idx', on_change=preencher_caso)
    st.caption('Os exemplos foram reservados antes do desenvolvimento. Você pode editar as características abaixo.')

    with st.form('simulacao'):
        a,b,c = st.columns(3)
        with a:
            st.selectbox('Cilindros', [3,4,5,6,8], key='cylinders')
            st.number_input('Cilindrada (pol³)', min_value=1.0, max_value=1000.0, step=1.0, key='displacement')
            st.number_input('Potência (hp)', min_value=1.0, max_value=1000.0, step=1.0, key='horsepower')
            st.checkbox('Potência desconhecida', key='hp_missing',
                        help='O pipeline usa a mediana de potência aprendida no treino.')
        with b:
            st.number_input('Peso (lb)', min_value=100.0, max_value=10000.0, step=10.0, key='weight')
            st.number_input('Aceleração 0–60 mph (s)', min_value=1.0, max_value=60.0, step=0.1,
                            format='%.1f', key='acceleration')
        with c:
            st.number_input('Ano-modelo', min_value=1970, max_value=1982, step=1, key='year')
            st.selectbox('Origem', [1,2,3], key='origin',
                         format_func=lambda x: {1:'EUA',2:'Europa',3:'Japão'}[x])
        submitted = st.form_submit_button('Estimar consumo', type='primary')

    if submitted:
        entrada = {c: st.session_state[c] for c in ['cylinders','displacement','weight','acceleration','origin']}
        entrada['horsepower'] = np.nan if st.session_state.hp_missing else st.session_state.horsepower
        entrada['model_year'] = st.session_state.year - 1900
        quadro = pd.DataFrame([entrada], columns=meta['features'])
        previsao = float(modelo.predict(quadro)[0])
        st.session_state['prediction_mpg'] = previsao
        st.session_state['prediction_input'] = entrada

    if 'prediction_mpg' in st.session_state:
        previsao = st.session_state.prediction_mpg
        x,y,z = st.columns(3)
        x.metric('Previsão (MPG)', f'{previsao:.3f}')
        y.metric('Equivalente (km/L)', f'{previsao * MPG_PARA_KM_L:.2f}')
        z.metric('MAE no teste (MPG)', f"{meta['metricas_teste']['mae']:.3f}")
        st.caption('MAE é o erro absoluto médio nos dados de teste; não é um intervalo de confiança para este carro. '
                   'Quanto maior o MPG ou km/L, maior a eficiência.')
        fora = [c for c,l in meta['limites_treino'].items()
                if not pd.isna(st.session_state.prediction_input[c]) and
                not l['min'] <= st.session_state.prediction_input[c] <= l['max']]
        if fora:
            st.warning('Valores fora do intervalo observado no treino: '+', '.join(fora)+'. A previsão exige cautela.')
        caso = casos[st.session_state.example_idx]
        corresponde = all(
            (pd.isna(st.session_state.prediction_input[c]) and caso[c] is None) or
            (caso[c] is not None and np.isclose(st.session_state.prediction_input[c],caso[c]))
            for c in meta['features'])
        if corresponde:
            erro_paridade = abs(previsao-caso['previsto_mpg'])
            st.success(f"Exemplo original: real {caso['real_mpg']:.2f} MPG; "
                       f"previsão do notebook {caso['previsto_mpg']:.3f} MPG. "
                       f"Diferença interface–notebook: {erro_paridade:.10f} MPG.")

with comparar:
    st.subheader('Três algoritmos, nove configurações')
    st.write(f"Selecionado por validação cruzada: **{meta['modelo']} · {meta['estrategia']}**.")
    tabela = pd.read_csv(ROOT / 'results' / 'comparacao_9_modelos.csv')
    st.dataframe(tabela[['modelo','estrategia','mae_treino','mae_cv','dp_mae_cv','gap_mae','tempo_busca_s']].round(4),
                 hide_index=True, width='stretch')
    st.image(str(ROOT/'results'/'08_comparacao_final.png'), width='stretch')
    st.caption('A escolha usa apenas validação cruzada. O teste não participa do ranking.')
    m1,m2,m3=st.columns(3)
    m1.metric('MAE final (MPG)',f"{meta['metricas_teste']['mae']:.3f}")
    m2.metric('RMSE final (MPG)',f"{meta['metricas_teste']['rmse']:.3f}")
    m3.metric('R² final',f"{meta['metricas_teste']['r2']:.3f}")

with metodo:
    st.subheader('Uma comparação com regras comuns')
    st.markdown('''
    - **Base:** Auto MPG, UCI, 398 registros históricos.
    - **Separação:** 318 registros de treino e 80 reservados para teste, semente 42.
    - **Validação:** cinco folds iguais para todos os algoritmos.
    - **Preparação:** potência ausente recebe a mediana do treino; origem recebe one-hot encoding.
    - **Ajustes:** Grid Search e 20 trials Optuna para cada algoritmo.
    - **Aplicativo:** carrega o pipeline exportado do notebook e não retreina o modelo.
    ''')
    st.image(str(ROOT/'results'/'09_curva_aprendizado.png'),width='stretch')
    st.markdown('[Fonte e atribuição: Quinlan, R. (1993), Auto MPG — UCI](https://doi.org/10.24432/C5859H). '
                'Dados sob CC BY 4.0.')
    st.caption('Rafael Joda · Projeto acadêmico de Data Science & Statistical Computing')
