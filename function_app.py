import azure.functions as func
import logging
import base64
import numpy as np
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import json
import sys
import os

# Asegura que el entorno vea los paquetes instalados en .python_packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.python_packages/lib/site-packages')))

# Configura logging
logging.basicConfig(level=logging.INFO)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.route(route="verifymetodo")
def verifymetodo(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # 1. Parsear el JSON de entrada
        req_body = req.get_json()
        logging.info("Solicitud recibida")
        
        img1_base64 = req_body.get("image1")
        img2_base64 = req_body.get("image2")
        model_name = req_body.get("model", "VGG-Face")  # Default: VGG-Face
        threshold = float(req_body.get("threshold", 0.4))  # Default: 0.4

        # 2. Validar entradas
        if not img1_base64 or not img2_base64:
            logging.warning("Faltan imágenes en la solicitud")
            return func.HttpResponse(
                json.dumps({"error": "Se requieren 'image1' y 'image2' en Base64"}),
                status_code=400,
                mimetype="application/json"
            )

        # 3. Convertir imágenes
        logging.info("Procesando imágenes...")
        img1_array = base64_to_image(img1_base64)
        img2_array = base64_to_image(img2_base64)

        # 4. Comparar rostros con DeepFace
        logging.info(f"Comparando con modelo {model_name}...")
        result = DeepFace.verify(
            img1_path=img1_array,
            img2_path=img2_array,
            model_name=model_name,
            distance_metric="cosine",
            detector_backend="ssd",  # Más preciso que "ssd"
            enforce_detection=False
        )

        # 5. Formatear respuesta
        response = {
            "verified": bool(result["distance"] <= threshold),
            "confidence": float(1 - result["distance"]),
            "metrics": {
                "distance": float(result["distance"]),
                "threshold": threshold,
                "model": model_name,
                "detector": "ssd",
                "processing_time": result["time"]
            },
            "faces": {
                "image1_detected": bool(result["facial_areas"]["img1"]),
                "image2_detected": bool(result["facial_areas"]["img2"])
            }
        }

        logging.info(f"Resultado: {response['verified']} (Distancia: {result['distance']:.4f})")
        
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