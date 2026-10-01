"""Execute todas as células em ordem e salve o notebook com as saídas reais.

Alternativa a Run All no Jupyter/Colab, sem necessidade de abrir um servidor.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    path = root / 'Checkpoint05_RF_XGBoost_LightGBM.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from matplotlib_inline.backend_inline import configure_inline_support
    configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
    count = 0
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'Executando célula de código {count} (posição {i})...', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        cell.outputs = outputs
        cell.execution_count = count
        nbformat.write(notebook, path)
        if not result.success:
            print(captured.stdout)
            print(captured.stderr)
            raise RuntimeError(f'Erro na célula {i}; execução interrompida.') from (result.error_in_exec or result.error_before_exec)
        if captured.stdout:
            print(captured.stdout[:2000], flush=True)
    nbformat.validate(notebook)
    print('Execução completa: todas as células concluídas e notebook salvo.', flush=True)


if __name__ == '__main__':
    main()
