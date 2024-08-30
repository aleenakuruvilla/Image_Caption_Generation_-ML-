import streamlit as st

def app():
    st.title("Deep Learning Concepts")
    st.markdown("---")
    st.markdown("## Key Terms and Definitions")

    st.markdown("#### 1. ReLU (Rectified Linear Unit)")
    st.image("relu.png",width=500)
    st.markdown("- **Definition**: ReLU is an activation function defined as $f(x) = \max(0, x)$. It replaces all negative values in the input with zero while keeping positive values unchanged.")
    st.markdown("- **Why It's Used**: ReLU introduces non-linearity into the model, enabling it to learn complex patterns efficiently. It helps mitigate the vanishing gradient problem in deep learning.")

    st.markdown("---")
    st.markdown("#### 2. Softmax")
    st.image("softmax.webp")
    st.markdown("- **Definition**: Softmax is an activation function that converts a vector of values into a probability distribution. It is defined as $\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j} e^{x_j}}$, where $x$ is the input vector.")
    st.markdown("- **Why It's Used**: Softmax is used in the output layer of classification models to generate probabilities for different classes. In image captioning, it assigns probabilities to words in the vocabulary.")
    st.markdown("---")
    
    st.markdown("#### 3. Categorical Cross-Entropy")
    st.image("categorical-cross-entropy.png")
    st.markdown("- **Definition**: Categorical cross-entropy is a loss function used for multi-class classification. It measures the difference between the true label distribution and the predicted probability distribution. It is defined as $L = -\\sum_{i} y_i \\log(p_i)$, where $y_i$ is the true label and $p_i$ is the predicted probability.")
    st.markdown("- **Why It's Used**: It helps train models by penalizing incorrect predictions and guiding them towards correct class distributions. In image captioning, it aids in generating accurate captions by comparing predicted and actual words.")
    st.markdown("---")
   
    
    st.markdown("#### 4. Optimizer='adam'")
    st.markdown("- **Definition**: Adam (Adaptive Moment Estimation) is an optimization algorithm that computes adaptive learning rates for each parameter. It combines advantages of AdaGrad and RMSProp.")
    st.markdown("- **Why It's Used**: Adam is computationally efficient and adjusts learning rates dynamically, facilitating faster convergence in training deep learning models with large datasets.")
    st.markdown("---")
    
    st.markdown("#### 5. Embedding Layer")
    st.markdown("- **Definition**: An embedding layer maps high-dimensional categorical data (like words) into a lower-dimensional continuous vector space. It captures semantic relationships between categories.")
    st.markdown("- **Why It's Used**: In image captioning, an embedding layer converts words into vectors, enhancing the model's ability to process and generate meaningful textual descriptions.")
    st.markdown("---")
   
    st.markdown("#### 6. LSTM (Long Short-Term Memory)")
    st.image("lstm.webp")
    st.markdown("- **Definition**: LSTM is a type of recurrent neural network (RNN) designed to learn long-term dependencies. It includes memory cells to store information over time.")
    st.markdown("- **Why It's Used**: LSTMs excel in handling sequential data, maintaining context over sequences of words. In image captioning, they generate coherent captions by remembering important features of the input image.")
    st.markdown("---")
    
    st.markdown("#### 7. VGG16")
    st.image("vgg16.webp")
    st.markdown("- **Definition**: VGG16 is a convolutional neural network architecture with 16 layers, known for its simplicity and depth. It uses small (3x3) convolutional filters.")
    st.markdown("- **Why It's Used**: VGG16 is effective in image feature extraction due to its ability to capture detailed visual features. In image captioning, it extracts meaningful representations from input images to aid in generating accurate captions.")

