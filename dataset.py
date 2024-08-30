import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def app():
    # # Example data (replace with actual dataset)
    # images = ['1000268201_693b08cb0e.jpg', '1001773457_577c3a7d70.jpg']  # List of image paths
    # captions = [
    #     ['A child in a pink dress is climbing up a set of stairs in an entry way .', 'A girl going into a wooden building .', 'A little girl climbing into a wooden playhouse .', 'A little girl climbing the stairs to her playhouse .', 'A little girl in a pink dress going into a wooden cabin .'],
    #     ['A black dog and a spotted dog are fighting', 'A black dog and a tri-colored dog playing with each other on the road .', 'A black dog and a white dog with brown spots are staring at each other in the street .', 'Two dogs of different breeds looking at each other on the road .', 'Two dogs on pavement moving toward each other .'],
    # ]

    # # Number of images
    # num_images = len(images)

    # # Number of captions per image
    # num_captions = [len(caption_list) for caption_list in captions]

    # # Caption lengths
    # caption_lengths = [len(caption.split()) for caption_list in captions for caption in caption_list]

    # # Displaying the number of captions per image as a histogram
    # st.subheader('Number of Captions per Image')
    # fig_num_captions = plt.figure(figsize=(10, 5))
    # plt.hist(num_captions, bins=range(1, max(num_captions) + 2), align='left')
    # plt.title('Number of Captions per Image')
    # plt.xlabel('Number of Captions')
    # plt.ylabel('Frequency')
    # st.pyplot(fig_num_captions)

    # # Displaying the caption lengths as a histogram
    # st.subheader('Caption Length Distribution')
    # fig_caption_lengths = plt.figure(figsize=(10, 5))
    # plt.hist(caption_lengths, bins=range(1, max(caption_lengths) + 2), align='left')
    # plt.title('Caption Length Distribution')
    # plt.xlabel('Caption Length (words)')
    # plt.ylabel('Frequency')
    # st.pyplot(fig_caption_lengths)

    # # Displaying images with captions
    # st.subheader('Images with Captions')
    # for i, image_path in enumerate(images):
    #     st.image(image_path, caption='\n'.join(captions[i]), use_column_width=True)

    # hai = 6+5
    # st.write(hai)

    # Display code using st.code()
   


    img_file_buffer = st.file_uploader('Upload a PNG image')
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)