import numpy as np
from PIL import Image
from deepface import DeepFace

def extract_face(img: Image, size=(112, 112)) -> np.ndarray:
    img = np.array(img)
    face_obj = DeepFace.extract_faces(img, target_size=size, enforce_detection=False)
    face = face_obj[0]["face"] * 255
    return face.astype(np.uint8)




