import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image


# Additional function Peak Signal to Noise Ratio
def psnr(orig, pred):
    # cast the target images to integer
    orig = orig * 255.0
    orig = tf.cast(orig, tf.uint8)
    orig = tf.clip_by_value(orig, 0, 255)
    # cast the predicted images to integer
    pred = pred * 255.0
    pred = tf.cast(pred, tf.uint8)
    pred = tf.clip_by_value(pred, 0, 255)
    # return the psnr
    return tf.image.psnr(orig, pred, max_val=255)


def ShowPhoto(model, src_img, one_photo):
    input_size = 128

    img_lab = cv2.cvtColor(src_img, cv2.COLOR_RGB2Lab)
    img_lab = img_lab/255.0
    # resize image to network input size
    img_lab_rs = cv2.resize(img_lab, (input_size, input_size))
    gray = img_lab_rs[:, :, 0]  # pull out L channel

    output = model.predict(np.expand_dims(gray, axis=0), verbose=False)

    # gathering image chanels
    new_img = np.zeros((input_size, input_size, 3))
    new_img[:, :, 0] = gray
    new_img[:, :, 1:] = output[0]
    new_img = 255*new_img
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_Lab2RGB)

    # lab to rgb
    y_img = img_lab_rs[:, :, 1:]
    lab_img = np.zeros((input_size, input_size, 3))
    lab_img[:, :, 0] = gray
    lab_img[:, :, 1:] = y_img
    lab_img = 255*lab_img
    lab_img = lab_img.astype(np.uint8)
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2RGB)

    if one_photo:
        st.image(new_img, width=500)
    else:
        image_list = []
        image_list.append(rgb_img)
        image_list.append(gray)
        image_list.append(new_img)

        captions = ['Base image', 'Grayscale image', 'Predicted image']
        st.image(image_list, caption=captions, clamp=True, width=225)


def ShowImages(model, base_img, num=1, one_photo=False):
    if type(base_img) == list:
        arr_idx = np.random.randint(0, len(base_img), num)
        for idx in arr_idx:
            img_path = os.path.join(directory, base_img[idx])
            src_img = cv2.imread(img_path)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            ShowPhoto(model, src_img, one_photo)
    else:
        ShowPhoto(model, base_img, one_photo)


model = keras.models.load_model(
    'colorizer_model', custom_objects={"psnr": psnr})

directory = "test_dataset"
test_files_list = [f for f in os.listdir(directory) if f.endswith(
    ".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

st.title('Black-and-white photos colorizer')
st.caption('This is neural network that can receive black and white image and return colorize image.\n\
    For now, it returns only 128x128 pixels images .\n\
    Model works better with photos of human faces, so it is preferable to use it for such photos.\n')
st.header('Model validation')


photos_amount = st.number_input(
    'Insert a number of photos to be colorized', min_value=1, max_value=10)

if st.button('Colorize some random photos'):
    ShowImages(model, test_files_list, photos_amount)

st.header('Selecting a photo for coloring')

uploaded_photo = st.file_uploader(
    "Select a photo", type=['png', 'jpg', 'jpeg'])
if st.button('Colorize selected photo'):
    if uploaded_photo is not None:
        image = Image.open(uploaded_photo)
        ShowImages(model, np.array(image))

if st.button('Get only predicted photo'):
    if uploaded_photo is not None:
        image = Image.open(uploaded_photo)
        ShowImages(model, np.array(image), 1, True)
