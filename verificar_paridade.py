"""Testa a interface real do Streamlit nos cinco casos registrados no notebook."""
from pathlib import Path
import json
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


def main():
    meta = json.loads((ROOT/'artifacts'/'metadata.json').read_text())
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    assert not app.exception, app.exception
    linhas=[]
    for i,caso in enumerate(meta['casos_consistencia']):
        app.selectbox(key='example_idx').set_value(i).run()
        app.button[0].click().run()
        assert not app.exception, app.exception
        previsto=float(app.session_state['prediction_mpg'])
        esperado=caso['previsto_mpg']
        assert np.isclose(previsto,esperado,rtol=0,atol=1e-8), (previsto,esperado)
        exibido=next(m.value for m in app.metric if m.label=='Previsão (MPG)')
        assert abs(float(exibido)-esperado)<=0.00050001
        linha={'caso':i+1,'id_linha':caso['id_linha'],'notebook_mpg':esperado,
               'interface_mpg':previsto,'interface_exibido':exibido,
               'diferenca_absoluta':abs(previsto-esperado),'status':'OK'}
        linhas.append(linha)
        print(linha)
    (ROOT/'results'/'paridade_streamlit.json').write_text(json.dumps(linhas,indent=2))
    print('OK: os cinco casos produziram a mesma previsão no notebook e na interface Streamlit.')


if __name__=='__main__':
    main()
