import streamlit as st
import os #for handling the files
import re
import pickle #Storing numpy features (#means image features)
import numpy as np
from tqdm.notebook import tqdm #giving us a UI for how much data is processed till now (#getting an estimation of overall process)
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input #Extracting and preprocessing data
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

def app():
    with open('features.pkl','rb') as f:
        features = pickle.load(f)
    with open('captions.txt', 'r') as f:
        next(f) #Used to skip the first line
        captions_doc = f.read()
    

    mapping = {}
    for line in tqdm(captions_doc.split('\n')): #tqdm is used to show a progress bar 100%
        tokens = line.split(',') #the image name and description is seperated by a comma.
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:] #Seperating image name and caption in caption.txt
        
        image_id = image_id.split('.')[0] #take the first element in list formed by splitting using'.'
        # It is used to extract the image id alone without the format name.
        
        caption = " ".join(caption) #Combining list into a single string.
        if image_id not in mapping:
            mapping[image_id] = [] #Adding the image id that is the key alone to the dictionary.
        mapping[image_id].append(caption) #Mapping image id with the caption

    # Preprocess text data
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i] #Extracting one caption from the list
            caption = caption.lower() #Converting into lowercase letters
            caption = re.sub('[^A-Za-z]', ' ', caption) #Replace all the characters other than [^a-z] into an empty space. (re-regular expression)
            caption = re.sub('\\s+', ' ', caption) #Removing extra spaces
            caption = 'begin ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' end'
            #Filtering list of words that are of small length to remove noise or impurities and adding startseq and endseq
            captions[i] = caption

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = 8425

    max_length = max(len(caption.split()) for caption in all_captions)


    # encoder model
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


    model.load_weights('image_caption_generator.h5')

    def index_to_word(integer, tokenizer):
        for word,index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    #generate captions for an image
    def predict_caption(model,image,tokenizer,max_length):
        in_text = 'begin'
        for i in range(max_length):
            #convert sequence into and integer
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence],max_length)
            #Now predicting next word
            yword = model.predict([image,sequence],verbose=0)

            yword = np.argmax(yword) #Return the word the most probability

            #converting the index value to word
            word = index_to_word(yword,tokenizer)

            #stop if word not found
            if word is None:
                break

            #Append the word to the existing word
            in_text += ' '+word

            if word == 'end':
                break
        return in_text
        
    # def feature_extractor():
    #     model = VGG16()
    #     model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    #     return model
    
    # def extract_features(image, feature_extractor):
    #     image = image.resize((224, 224)) 
    #     img_array = img_to_array(image)
    #     img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    #     img_array = preprocess_input(img_array)
    #     feature = feature_extractor.predict(img_array, verbose=0)
    #     return feature

    st.title('Image Caption Generator')

    # File uploader for PNG images
    img_file_buffer = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])

    # Check if an image is uploaded
    if img_file_buffer is not None:
        # Read and open the image
        image = Image.open(img_file_buffer)
        image_name = img_file_buffer.name
        image_id = image_name.split('.')[0]


        # Display the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')  # Hide axis
        st.pyplot(fig)

        # feature_model = feature_extractor()
        # image_features = extract_features(image,feature_model)
        y_pred = predict_caption(model, features[image_id], tokenizer, max_length)

        captions = mapping[image_id]
        st.markdown("### Actual Caption: ")
        for caption in captions:
            st.write(caption)



        # Show predicted caption
        st.markdown("### Predicted Caption: ")
        st.write(y_pred)
