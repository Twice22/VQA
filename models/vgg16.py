from keras.preprocessing import image
from keras.models import Model
import numpy as np

## VGG16 model
from keras.applications.vgg16 import VGG16, preprocess_input
vgg16_model = VGG16(weights='imagenet', include_top=True)

# get only the layer corresponding to the features
vgg16_model_extractfeatures = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('fc2').output)


## Resnet50 model
from keras.applications.resnet50 import ResNet50, preprocess_input
resnet_model = ResNet50(weights='imagenet', include_top=True)

# get only the layer corresponding to the features
resnet_model_extractfeatures = Model(inputs=resnet_model.input, outputs=resnet_model.get_layer('flatten_1').output)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract_features(img_path, model):
    x = preprocess_input(path_to_tensor(img_path))
    fc2_features = model.predict(x)
    return fc2_features