import streamlit as st
import tempfile
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import cv2

st.title("Keras models")
st.write("Select the desired model")

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type=['png','jpeg','jpg'])
    models = st.radio(
    "Select the desired model",
    ('Boston hous', 'Cifar10', 'Cifar100', 'Fashion MNIST', 'MNIST'))

#------------           FASHION MNIST        -------------------

if models == 'Fashion MNIST':
    st.title("Fashion MNIST")
    st.write(f'Can identify these things ➡', 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


    model = tf.keras.models.load_model('fashion_mnist.h5')

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = cv2.imread(uploaded_file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
        pre = np.argmax(model.predict(np.expand_dims(img,0)))
        label = {
                    0:'T-shirt/top',
                    1:'Trouser',
                    2:'Pullover',
                    3:'Dress',
                    4:'Coat',
                    5:'Sandal',
                    6:'Shirt',
                    7:'Sneaker',
                    8:'Bag',
                    9:'Ankle boot'
        }
        cap = label[pre]
        st.title(f"This is {cap}")
        st.image(image, caption=f"Model predict: {cap}")

#------------           CIFAR 10        -------------------

elif models == 'Cifar10':
    st.title("Cifar10")
    st.write(f'Can identify these things ➡ airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |')

    model = tf.keras.models.load_model('cifar10.h5')

    if uploaded_file is not None:
        image = Image.open(uploaded_file.name)
        img = cv2.imread(uploaded_file.name)
        img = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
        pre = np.argmax(model.predict(np.expand_dims(img,0)))
        label = {
                    0:'airplane',
                    1:'automobile',
                    2:'bird',
                    3:'cat',
                    4:'deer',
                    5:'dog',
                    6:'frog',
                    7:'horse',
                    8:'ship',
                    9:'truck'
        }
        cap = label[pre]
        st.title(f"This is {cap}")
        st.image(image, caption=f"Model predict: {cap}")

#------------           CIFAR 100        -------------------

elif models == 'Cifar100':
    st.title("Cifar100")
    st.write(f'Can identify these things ➡','apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl',
        'boy','bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 
        'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 
        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 
        'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 
        'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    model = tf.keras.models.load_model('cifar100.h5')

    if uploaded_file is not None:
        image = Image.open(uploaded_file.name)
        img = cv2.imread(uploaded_file.name)
        img = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
        pre = np.argmax(model.predict(np.expand_dims(img,0)))
        label = {
                0: 'apple',
                1: 'aquarium_fish',
                2: 'baby',
                3: 'bear',
                4: 'beaver',
                5: 'bed',
                6: 'bee',
                7: 'beetle',
                8: 'bicycle',
                9: 'bottle',
                10: 'bowl',
                11: 'boy',
                12: 'bridge',
                13: 'bus',
                14: 'butterfly',
                15: 'camel',
                16: 'can',
                17: 'castle',
                18: 'caterpillar',
                19: 'cattle',
                20: 'chair',
                21: 'chimpanzee',
                22: 'clock',
                23: 'cloud',
                24: 'cockroach',
                25: 'couch',
                26: 'cra',
                27: 'crocodile',
                28: 'cup',
                29: 'dinosaur',
                30: 'dolphin',
                31: 'elephant',
                32: 'flatfish',
                33: 'forest',
                34: 'fox',
                35: 'girl',
                36: 'hamster',
                37: 'house',
                38: 'kangaroo',
                39: 'keyboard',
                40: 'lamp',
                41: 'lawn_mower',
                42: 'leopard',
                43: 'lion',
                44: 'lizard',
                45: 'lobster',
                46: 'man',
                47: 'maple_tree',
                48: 'motorcycle',
                49: 'mountain',
                50: 'mouse',
                51: 'mushroom',
                52: 'oak_tree',
                53: 'orange',
                54: 'orchid',
                55: 'otter',
                56: 'palm_tree',
                57: 'pear',
                58: 'pickup_truck',
                59: 'pine_tree',
                60: 'plain',
                61: 'plate',
                62: 'poppy',
                63: 'porcupine',
                64: 'possum',
                65: 'rabbit',
                66: 'raccoon',
                67: 'ray',
                68: 'road',
                69: 'rocket',
                70: 'rose',
                71: 'sea',
                72: 'seal',
                73: 'shark',
                74: 'shrew',
                75: 'skunk',
                76: 'skyscraper',
                77: 'snail',
                78: 'snake',
                79: 'spider',
                80: 'squirrel',
                81: 'streetcar',
                82: 'sunflower',
                83: 'sweet_pepper',
                84: 'table',
                85: 'tank',
                86: 'telephone',
                87: 'television',
                88: 'tiger',
                89: 'tractor',
                90: 'train',
                91: 'trout',
                92: 'tulip',
                93: 'turtle',
                94: 'wardrobe',
                95: 'whale',
                96: 'willow_tree',
                97: 'wolf',
                98: 'woman',
                99: 'worm'
        }
        cap = label[pre]
        st.title(f"This is {cap}")
        st.image(image, caption=f"Model predict: {cap}")

#------------           MNIST        -------------------

elif models == 'MNIST':
    st.title("MNIST")
    st.write(f'The model can recognize the following numbers ➡ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9')

    model = tf.keras.models.load_model('mnist.h5')

    if uploaded_file is not None:
        image = Image.open(uploaded_file.name)
        img = cv2.imread(uploaded_file.name,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
        pre = np.argmax(model.predict(np.expand_dims(img,0)))
        st.title(f'This is {pre}')
        st.image(image, caption=f"Model predict: {pre}")


#------------           BOSTON HOUS        -------------------

elif models == 'Boston hous':
    quetion=0
    st.title("Boston hous price")
    inputs_text = ["per capita crime rate by town","proportion of residential land zoned for lots over 25,000 sq.ft",
    "proportion of non-retail business acres per town","Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
    "nitric oxides concentration (parts per 10 million)","average number of rooms per dwelling","proportion of owner-occupied units built prior to 1940",
    "weighted distances to five Boston employment centres","index of accessibility to radial highways","full-value property-tax rate per $10,000",
    "pupil-teacher ratio by town","1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
    "% lower status of the population"]
    inputs = []
    
   
    

    for i in inputs_text:
        number = st.number_input(i)
        inputs.append(number)
    #st.write('The current number is ', inputs)
    if st.button('Run'):
        quetion = 1

    if quetion == 1:
        model = tf.keras.models.load_model('boston_hous_model.h5',compile=False)
        pre = model.predict(np.expand_dims(np.array(inputs),0))
        st.write("Model predict: ",pre.item())

else:
    st.write("Faylni yuklab bo'lmadi")
