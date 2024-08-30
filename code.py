import streamlit as st

def app():
    with st.sidebar:
        st.markdown("# Sections")
        st.sidebar.markdown("""<a href="#step-1-importing-libraries" style="font-size: 18px; color: black; text-decoration: none;">Step 1: Importing Libraries</a>""", unsafe_allow_html=True)
        st.sidebar.markdown("""<a href="#step-2-feature-extraction-using-vgg16-model" style="font-size: 18px; color: black; text-decoration: none;">Step 2: Feature extraction using VGG16 model</a>""", unsafe_allow_html=True)
        st.sidebar.markdown("""<a href="#step-3-preprocessing-caption-data" style="font-size: 18px; color: black; text-decoration: none;">Step 3: Preprocessing caption data</a>""", unsafe_allow_html=True)
        st.sidebar.markdown("""<a href="#step-4-splitting-the-dataset-for-training-and-splitting" style="font-size: 18px; color: black; text-decoration: none;">Step 4: Splitting the dataset for training and splitting</a>""", unsafe_allow_html=True)
        st.sidebar.markdown("""<a href="#step-5-setting-up-image-caption-generator-model" style="font-size: 18px; color: black; text-decoration: none;">Step 5: Setting up Image caption generator model</a>""", unsafe_allow_html=True)
        st.sidebar.markdown("""<a href="#step-6-train-the-image-caption-generator-model" style="font-size: 18px; color: black; text-decoration: none;">Step 6: Train the image caption generator model</a>""", unsafe_allow_html=True)
        st.sidebar.markdown("""<a href="#step-7-testing-the-model" style="font-size: 18px; color: black; text-decoration: none;">Step 7: Testing the model</a>""", unsafe_allow_html=True)
    st.code("""
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


    """)
    st.markdown("## Step-2 Feature extraction using VGG16 model")
    st.markdown("#### Loading base directory")
    st.code("BASE_DIR = os.getcwd() ")
    st.markdown("#### Loading VGG16 model")
    st.code(""" 

model = VGG16()

model = Model(inputs=model.inputs, outputs=model.layers[-2].output)#we only need the previous layer for extracting features

print(model.summary())

""")
    st.markdown("#### Extracting features of images into dictionary")
    st.code("""
Extract features from images
features = {}
directory = os.path.join(BASE_DIR, 'Images')
for img_name in tqdm(os.listdir(directory)):
    
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature


""")

    st.markdown("## Step-3 Preprocessing caption data")
    st.markdown("#### Loading caption data")
    st.code("""
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

""")
    st.markdown("#### Splitting image id and caption")
    st.code("""
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

""")
    st.markdown("#### Caption preprocessing function")
    st.code("""
# Preprocess text data
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i] #Extracting one caption from the list
            caption = caption.lower() #Converting into lowercase letters
            caption = re.sub('[^A-Za-z]', ' ', caption) #Replace all the characters other than [^a-z] into an empty space. (re-regular expression)
            caption = re.sub('\\s+', ' ', caption) #Removing extra spaces
            caption = 'begin ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' end'
            #Filtering list of words that are of small length to remove noise or impurities and adding startseq and endseq
            captions[i] = caption

""")
    st.markdown("#### Storing the processed captions into a single list")
    st.code("""
            
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

""")
    st.markdown("#### Tokenizing the text and setting the vocabulary size")
    st.code("""
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
            
# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length

""")
    st.markdown("## Step-4 Splitting the dataset for training and splitting")
    st.code("""
image_ids = list(mapping.keys()) #converting the image id's into list to get the number of unique images stored.
split = int(len(image_ids) * 0.90) #90% Split
train = image_ids[:split] #90% Training data
test = image_ids[split:] #10% Testing data

            
""")
    st.markdown("#### Data generator function")
    st.code("""

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield ((np.array(X1, dtype=np.float32), np.array(X2, dtype=np.int32)), np.array(y, dtype=np.float32))
                X1, X2, y = list(), list(), list()
                n = 0


""")
    st.markdown("## Step-5 Setting up Image caption generator model")
    st.code("""
            
# image feature layers
inputs1 = Input(shape=(4096,))#Input layer 
#Refer to VGG16 model it contain 4096 feature.
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1) #fe1 is the input to the dense layer
#number of neurons = 256

# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) #256 is the embedding dimension
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')


""")
    st.image("model.png",caption="Image Caption Generator Model")
    st.markdown("#### Setting the type configuration for data generator")
    st.code("""
import tensorflow as tf
output_signature = (
    (tf.TensorSpec(shape=(None, 4096), dtype=tf.float32), tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
)

            

""")
    st.markdown("#### Creating the dataset")
    st.code("""
def create_dataset(train, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size),
        output_signature=output_signature
    )
    return dataset
""")
    st.markdown("## Step-6 Train the image caption generator model")
    st.code("""

epochs = 15
batch_size = 64
steps = len(train) // batch_size

# Create dataset
dataset = create_dataset(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)

for epoch in range(epochs):
    # Create data generator for this epoch
    # generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    
    # Fit the model for one epoch
    model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=1)

""")
    st.markdown("#### Saving the trained model")
    st.code("""

 model.save('image_caption_generator.h5')
""")
    
    st.markdown("## Step-7 Testing the model")
    st.markdown("#### Loading the weights of the trained model")
    st.code("""
model.load_weights('image_caption_generator.h5')
""")
    st.markdown("#### Index to word converted Function")
    st.code("""
def index_to_word(integer, tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
""")
    st.markdown("#### Image caption sequence generator function")
    st.code("""

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'begin'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yword = model.predict([image, sequence], verbose=0)
        yword = np.argmax(yword)
        word = index_to_word(yword, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

""")
    st.markdown("#### Image Caption Generation and Visualization")
    st.code("""
from PIL import Image
import matplotlib.pyplot as plt
import os

def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    image_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(image_path)
    
    captions = mapping[image_id]
    print("Actual Caption:\n")
    for caption in captions:
        print(caption)
   
    # Display the image
    plt.imshow(image)
    plt.axis('off') 
    plt.show()

    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print("\nPredicted Caption:\n")
    print(y_pred)
    

""")
    
    st.code("""
    generate_caption("1002674143_1b742ab4b8.jpg")
""")