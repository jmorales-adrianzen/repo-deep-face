import azure.functions as func
import logging
import base64
import numpy as np
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import json

# Configura logging
logging.basicConfig(level=logging.INFO)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="analyze")
def analyze(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # 1. Parsear el JSON de entrada
        req_body = req.get_json()
        logging.info("Solicitud recibida")
        
        img1_base64 = req_body.get("imagen")
        actions=("age", "gender", "emotion", "race")

        # 2. Validar entradas
        if not img1_base64 or not actions:
            logging.warning("Faltan imagen o acciones en la solicitud")
            return func.HttpResponse(
                json.dumps({"error": "Se requiere 'image1' o 'action' en Base64"}),
                status_code=400,
                mimetype="application/json"
            )

        # 3. Convertir imágenes
        logging.info("Procesando imágenes...")
        img1_array = base64_to_image(img1_base64)

        # 4. Comparar rostros con DeepFace
        logging.info(f"Comparando con modelo {actions}...")      
        result = DeepFace.analyze(
            img_path=img1_array,  # Pasamos el array directamente
            actions=actions,
            detector_backend="retinaface",
            enforce_detection=False            
        )        

        logging.info("Análisis completado")
        
        # Mapeo directo de campos (sin cálculos intermedios)
        response = format_raw_deepface_response(result)

        #logging.info(f"Resultado: {response['verified']} (Distancia: {result['distance']:.4f})")
        
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
    except ValueError as e:
        logging.error(f"Error de validación: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error interno: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "Error interno del servidor"}),
            status_code=500,
            mimetype="application/json"
        )

        
def base64_to_image(base64_str: str) -> np.ndarray:
    try:
        # Elimina el prefijo si existe (ej: "data:image/jpeg;base64,")
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_bytes))
        return np.array(img.convert("RGB"))
    
    except Exception as e:
        logging.error(f"Error decodificando Base64: {str(e)}")
        raise ValueError(f"Formato de imagen inválido: {str(e)}")         
    
    
def format_raw_deepface_response(analysis_result):
    """
    Formatea los resultados de DeepFace.analyze() considerando la estructura correcta.
    Args:
        analysis_result: Resultado de DeepFace.analyze() (lista de diccionarios)
    Returns:
        Diccionario formateado
    """
    if not analysis_result or not isinstance(analysis_result, list):
        return {"analysis": {}}

    # Extraer el primer resultado si existe
    first_result = analysis_result[0] if len(analysis_result) > 0 else {}

    # Verificar que sea un diccionario antes de acceder
    if not isinstance(first_result, dict):
        return {"analysis": {}}

    try:
        formatted = {
            "analysis": {
                "age": int(first_result.get('age', 0)),
                "gender": {
                    "dominant": first_result.get('dominant_gender', 'unknown'),
                    "confidence": {
                        "Man": float(first_result.get('gender', {}).get('Man', 0.0)),
                        "Woman": float(first_result.get('gender', {}).get('Woman', 0.0))
                    }
                },
                "emotion": {
                    "dominant": first_result.get('dominant_emotion', 'neutral'),
                    "confidence": {
                        emotion: float(score) 
                        for emotion, score in first_result.get('emotion', {}).items()
                    }
                },
                "race": {
                    "dominant": first_result.get('dominant_race', 'unknown'),
                    "confidence": {
                        race.replace(" ", "_"): float(score) 
                        for race, score in first_result.get('race', {}).items()
                    }
                },
                "face_region": {
                    "x": first_result.get('region', {}).get('x', 0),
                    "y": first_result.get('region', {}).get('y', 0),
                    "w": first_result.get('region', {}).get('w', 0),
                    "h": first_result.get('region', {}).get('h', 0),
                    "confidence": float(first_result.get('face_confidence', 0.0))
                }
            }
        }
        return formatted
    except Exception as e:
        print(f"Error al formatear: {str(e)}")
        return {"analysis": {}}    