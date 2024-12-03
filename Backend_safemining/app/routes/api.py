from flask import Flask, request, send_file
import pandas as pd
from io import BytesIO
from app.model.modelBETO import RiesgoMineriaModel
from flask_cors import CORS



# Crear una instancia del modelo (asegúrate de que el modelo esté preentrenado)
model = RiesgoMineriaModel()

# Crear la aplicación Flask
app = Flask(__name__)

# Habilitar CORS
CORS(app)

# Endpoint para recibir y procesar el archivo CSV
@app.route("/procesar_csv/", methods=["POST"])
def procesar_csv():
    # Verificar si el archivo fue enviado
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    # Leer el archivo CSV recibido
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error al leer el archivo CSV: {str(e)}", 400

    # Verificar si la columna 'text' existe
    if 'text' not in df.columns:
        return "El archivo CSV no contiene una columna 'text'", 400

    # Crear una lista para almacenar las alertas generadas
    alertas_generadas = []

    # Procesar cada transcripción y predecir el nivel de riesgo
    for transcripcion in df['text']:
        alerta = model.predict(transcripcion)
        alertas_generadas.append(alerta)

    # Crear un DataFrame con las alertas generadas
    alertas_df = pd.DataFrame(alertas_generadas)

    # Guardar las predicciones en un archivo CSV
    output_csv = BytesIO()
    alertas_df.to_csv(output_csv, index=False)
    output_csv.seek(0)

    # Devolver el archivo CSV con las predicciones
    return send_file(output_csv, mimetype="text/csv", as_attachment=True, download_name="alertas_generadas.csv")

if __name__ == "__main__":
    # Correr la aplicación Flask
    app.run(debug=True)
