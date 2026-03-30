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
python -m mnist.local
```

### Mnist Distributed Data Parallel (DDP)
```bash
python -m mnist.ddp --rank 0 --world-size 2 --master-addr 127.0.0.1
python -m mnist.ddp --rank 1 --world-size 2 --master-addr 127.0.0.1
```

### CIFAR-10 local

```bash
python -m cifar10.local
```

### CIFAR-10 Distributed Data Parallel (DDP)

```bash
# Terminal 1
python -m cifar10.ddp --rank 0 --world-size 2 --master-addr 127.0.0.1

# Terminal 2
python -m cifar10.ddp --rank 1 --world-size 2 --master-addr 127.0.0.1
```

Flags disponibles:
- `--conv`: Usar red convolucional
- `--rgb`: Usar entrada RGB
