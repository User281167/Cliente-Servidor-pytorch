# Cliente-Servidor PyTorch

Proyecto de pruebas con PyTorch.

## Instalacion con `uv`

Desde la raiz del proyecto:

```bash
uv sync
```

Crea entorno virtual e instala las dependencias declaradas en `pyproject.toml`.

## Instalacion con `pip`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### MNIST local

```bash
uv run python -m mnist.local.main
python -m mnist.local.main
```
