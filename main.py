from fastapi import FastAPI, UploadFile, File
import uvicorn
import pymongo
import os, shutil, io
import cv2
from configuration import configuration_model
from PIL import Image
from starlette.responses import StreamingResponse
from starlette.responses import FileResponse


from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.visualizer import Visualizer, MetadataCatalog

app = FastAPI()

dbAtlas = pymongo.MongoClient(
    "mongodb+srv://user:user@cluster0.2jv3l.mongodb.net/warehouse_01?retryWrites=true&w=majority")
db = dbAtlas.get_database('warehouse_01')
records = db.object_detection_01

classes = ['Pallet Jacks', 'Rolling Ladder', 'Wire Mesh', 'Bulk Box', 'Totes',
           'Dump Hopper', 'Bin', 'Yard Ramp']

warehouse_metadata = MetadataCatalog.get("experiment1/train").set(thing_classes=classes)
print('metadata...', warehouse_metadata)


@app.get('/')
def index():
    return {'Hello': 'World'}


@app.post('/upload-image')
def create_upload_files(image: UploadFile = File(...)):
    # form = await request.form()
    print('Hello World')
    print('something.....', image.filename)
    print('abccc....', image)
    # with open(form.values(), 'r') as f:
    #     xyz = f.write()
    temp_file = _save_file_to_disk(image, path='original_image', save_as=image.filename)
    #
    img = cv2.imread('original_image/'+image.filename)
    # # print('image name....', self.window.filename)
    print('image shape is.....', img.shape)

    predictor = configuration_model()

    output = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=warehouse_metadata, scale=0.5)
    out = visualizer.draw_instance_predictions(output['instances'].to('cpu'))
    img_out = Image.fromarray(out.get_image()[:, :, ::-1])

    prediction= output['instances'].pred_classes.numpy()
    dict_list= list(set(prediction))
    dict_ = {}
    dict_list = [2, 4, 3, 5, 1, 6]

    for name_ in dict_list:
        dict_[classes[name_]] = 0
    print('dict_ value.....', dict_)
    prediction = [2, 3, 2, 4, 3, 2]
    for name_ in prediction:
        dict_[classes[name_]] = dict_[classes[name_]] + 1
    image_1 = {'Image': image.filename}
    #
    dict_['Image'] = image.filename
    records.insert_one(dict_)
    # print('predicted.......', out, 'and.....', img_out)
    # print('
    cv2.imwrite(f'predicted_image/{image.filename}', out.get_image()[:, :, ::-1])
    #
    # # temp_file = _save_file_to_disk(out.get_image()[:, :, ::-1], path='static/assets/img/predicted_image', save_as=image.filename+'_pred')
    # #xyzz = FileResponse("static/assets/img/predicted_image/pred.jpg")
    # # pil_img= Image.open(BytesIO(form.values()[0]))
    # # print('filename is.....', xyz)
    # dict_.pop('_id')
    return StreamingResponse(io.BytesIO(out.get_image()[:, :, ::-1].tobytes()), media_type="image/png")


def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=15000, reload=True)
