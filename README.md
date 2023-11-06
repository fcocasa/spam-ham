# Spam-Ham

Es un proyecto creado para difrenciar mensajes estilo Spam y Ham.

Proyecto pasado en el modelo pre-entrenado de:

https://huggingface.co/skandavivek2/spam-classifier?text=Hello%2C%0A%0AWe+contacted+you+at+the+end+of+October+asking+for+feedback+on+the+ICANN78+ccNSO+Members+Meeting.++So+far+22+people+have+responded.++We+would+be+very+grateful+if+you+could+complete+our+short+survey+%283+minutes+long%29+to+help+us+understand+what+went+well+%28and+not+so+well%29.%0A%0AHow+do+you+give+your+feedback%3F+Go+to+https%3A%2F%2Fwww.surveymonkey.com%2Fr%2FICANN78ccNSOMM%0A%0AThe+online+survey+will+be+open+until+Wednesday%2C+8+November+2023%2C+23%3A59+UTC.%0A%0AYour+feedback+is+valuable+input+for+the+ccNSO+Meeting+Programme+Committee+as+they+prepare+for+ICANN79+in+March+%2724.%0A%0AThank+you+very+much%21%0A%0ABest+regards%2C

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
