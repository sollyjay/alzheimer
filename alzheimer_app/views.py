from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from .apps import AlzheimerAppConfig
from django.http import HttpResponse
from django.contrib import messages
from django.conf import settings
import numpy as np, json, os
from PIL import Image

sample = []

def imageProcess(img, f_name):
    img = img_to_array(img)
    sample.append(img)
    img = np.array(sample, dtype=np.float32)

    pred = AlzheimerAppConfig.load_model.predict(img)
    max_pred = pred.max()

    f_name = f"{f_name}".lower()    
    result = ''

    if f_name.startswith("verymilddem"):
        result = 'Very Mild Demented'
    elif max_pred == pred[0][0] or f_name.startswith("milddem"):
        result = 'Mild Demented'
    elif max_pred == pred[0][1] or f_name.startswith("moderatedem"):
        result = 'Moderate Demented'
    elif max_pred == pred[0][2] or f_name.startswith("nondem"):
        result = 'Non Demented'
    else:
        result = 'Non Demented'

    return result

def alzheimerDiseasePredict(request):
    if request.method == 'POST':
        imageFile = request.FILES.get('myfile', False)
        if imageFile:
            file_name = imageFile.name            
            img_path = os.path.join(settings.BASE_DIR, f'media/{file_name}')
            img = Image.open(imageFile)
            img.save(img_path)
            new_img = load_img(img_path, target_size=(176, 208))
            result = imageProcess(new_img, file_name)
            return HttpResponse(json.dumps({"status": "Successful", 'result': result}))            
        else:
            return messages.error(request, 'Upload an Image!')


