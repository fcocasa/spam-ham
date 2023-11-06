# Spam-Ham

Es un proyecto creado para difrenciar mensajes estilo Spam y Ham.

Proyecto pasado en el modelo pre-entrenado de:

https://huggingface.co/skandavivek2/spam-classifier

## Qué instalar

pip3 install transformers

pip3 install wheel

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Re Entrenar

pip3 install datasets 

pip3 install numpy

pip3 install evaluate

pip3 install json

pip3 install pandas

### Limpiar Texto

pip3 install nltk

pip3 install -U spacy

python3 -m spacy download en_core_web_lg

## Ejecutar

En el archivo *data.json* se encunentran los parametros a configurar.
```
{
    "csv":"archivo.csv",
    "csv_a_limpiar":"dataset/spam_ham_dataset.csv",
    "model":"skandavivek2/spam-classifier",
    "use_tensor":0,
    "model_to_save":"modelo",
    "example_file":"spam-ham.txt"
}
```
 - **csv**: El archivo que se lee para entrenar el modelo, debe tener dos columnas, una con nombre **text** y otra con nombre **label**
 - **csv_a_limpiar**: Existe un archivo que se descargo de kaggle que se limpio, aca esta la direccion de dicho archivo
 - **model**: Que modelo se lee, puede ser tanto local como remoto (usar una direccion de HugginFace)
 - **model_to_save**: Una vez entrenado el modelo, se guarda en esta carpeta
 - **use_tensor**: En caso de usar un modelo local (archivo .safetensors), debe estar en 1 (True), sino 0 (False)
 - **example_file**: Archivo de texto al que se le realizará una prediccion

### Comando

#### Para Ejecutar una prediccion

python3 spam.py

#### Para Entrenar el modelo

python3 train_custom_data.py

#### Limpiar Texto

python3 clean_dataset.py

## Recomendacion

Crear un ambiente virtual

### Instalar

python3 -m venv /path/to/new/virtual/environment

Sustituir venv por cualquier otro nombre

### Activar

. venv/bin/activate
