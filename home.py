import streamlit as st

def app():
    st.title("Image Caption Generation")
    st.markdown("---")
    st.write("""
             


### Overview

Image caption generation is the task of generating a textual description for a given image. This combines computer vision and natural language processing to produce meaningful captions that describe the content of the image.

   """)

    st.markdown("""
             ---
### How It Works 
                 """)
    st.image("working2.webp",caption='Working of Caption generator')
    st.markdown("""
#### 1. **Image Feature Extraction**:         
- **Model Used**: We use the **VGG16 model**, a **pre-trained convolutional neural network**, to **extract features** from images.
- **Process**: The image is passed through the **VGG16 network**, and the **features** from **one of the last layers** (usually a dense layer) are **extracted**. These features represent the visual content of the image in a high-dimensional space.
""")
    st.image("vgg16_2.webp",caption = "VGG16 Architecture")
    st.markdown("""
                #### 2. **Sequence Processing**:
    - **Model Used**: An encoder-decoder architecture is employed for sequence processing.
    - **Encoder**: The extracted image features are fed into an encoder, typically an **LSTM (Long Short-Term Memory) network**, which processes the sequence of features.
    - **Embedding**: An embedding layer is used to convert words into vectors, allowing the model to understand and process textual data.
    - **Decoder**: The decoder is another **LSTM network** that generates the caption word by word. It takes the encoded image features and generates a sequence of words (the caption).
    """)
    st.image("lstm.png",caption = "LSTM-Architecture")
    st.markdown("""
#### 3. **Training**:
- **Dataset**: The model is trained on a large dataset of images and their corresponding captions. Popular datasets include MS COCO, **Flickr8k**, and Flickr30k.
- **Loss Function**: A common loss function for training image captioning models is **categorical cross-entropy**, which measures the accuracy of the predicted caption compared to the ground truth caption.
- **Optimization**: The model is optimized using algorithms like Adam or SGD to minimize the loss function.

---

### Features

- **Upload Image**: Users can upload an image from their device.
- **Generate Caption**: Upon uploading, the model processes the image and generates a descriptive caption.
- **Save and Share**: Users can save the generated captions and share them on social media platforms.
---
                """)
    st.markdown("## Other Models")
    st.markdown("""
                
### Sequence Processing Models

1. **GRU (Gated Recurrent Unit)**
    - GRU is a type of recurrent neural network (RNN) that combines the input and forget gates into a single update gate. It's known for its simpler architecture compared to LSTM, making it more computationally efficient for tasks requiring sequential data processing.
2. **Transformer**
    - Transformers rely solely on attention mechanisms to draw global dependencies between input and output sequences. They excel in tasks like machine translation and text generation by capturing long-range dependencies more effectively than traditional RNNs.
---
### Image Feature Extraction Models

1. **ResNet (Residual Network)**
    - ResNet introduced residual learning to deep convolutional neural networks (CNNs), allowing training of very deep networks (e.g., ResNet-50, ResNet-101) by using skip connections to mitigate the vanishing gradient problem. It's widely used in image classification, object detection, and image segmentation tasks.
2. **Inception (GoogLeNet)**
    - Inception modules, as used in GoogLeNet, utilize multiple filter sizes within the same layer to capture diverse features at different scales. This architecture enhances feature extraction capabilities while maintaining computational efficiency, making it suitable for tasks requiring detailed feature extraction from images.
3. **MobileNet**
    - MobileNet is designed for mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the computational complexity of standard CNNs while maintaining high accuracy in tasks like image classification and object detection on resource-constrained devices.
---
### Combined Models (Sequence and Image)

1. **BERT (Bidirectional Encoder Representations from Transformers)**
    - BERT is a transformer-based model pre-trained on vast amounts of text data. It revolutionized natural language processing (NLP) by capturing bidirectional context and achieving state-of-the-art results in tasks like question answering and sentiment analysis.
2. **ViT (Vision Transformer)**
    - ViT applies the transformer architecture directly to image data, treating images as sequences of patches. By leveraging self-attention mechanisms, ViT achieves competitive performance in image classification tasks, demonstrating the potential of transformers beyond traditional CNNs.

             """)