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