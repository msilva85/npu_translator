# Traductor offline usando modelo en NPU AMD

### Requerimientos:

FastFlowLM (FLM): https://fastflowlm.com/docs/
Python 3.12 o superior.

En este ejemplo se utilizo Debian trixie con los drivers AMD y la NPU firmware.
Tambien en la documentación oficial es compatible con Windows.

Una vez instalado y configurado FLM debes tener descargado el modelo. 

En consola: 
flm run translategemma:4b

Clona el repositorio.
Inicializa el entorno.
instala las dependencias python de requirements.txt.

### Codigo:

mkdir -p translategemma-portal/static && cd translategemma-portal
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### Levanta el aplicativo:

uvicorn main:app --host 0.0.0.0 --port 8080 --reload
