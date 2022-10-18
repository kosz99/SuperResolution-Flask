from flask import Flask, request, send_file
import torch
from model import SuperResolutionNet
import io
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision

app = Flask(__name__)
model = SuperResolutionNet(upscale_factor=3)
model.load_state_dict(torch.load("weights.pth", map_location=torch.device('cpu')))
model.eval()

to_tensor = torchvision.transforms.ToTensor()


@app.route('/superresolution', methods=['POST'])
def transform():
    
    if request.method=="POST":
        file = request.files['file']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        img_ycbcr = image.convert('YCbCr')
        img_y, img_cb, img_cr = img_ycbcr.split()
        img_y = to_tensor(img_y).unsqueeze(0)
        with torch.no_grad():
            img_out_y = model(img_y)
        img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
        final_img = Image.merge(
            "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
            ]).convert("RGB")
        
        #image_bytes = io.BytesIO()
        final_img.save("output.jpeg")
        #image_bytes = image_bytes.getvalue()
        #print(image_bytes)
        


    return send_file("output.jpeg", mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0') 