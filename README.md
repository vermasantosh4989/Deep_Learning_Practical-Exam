::: {align="center"}
# 🧠 Deep Learning Practical Project --- Set C

### From Artificial Neural Networks to Transformer-Based Sentiment Analysis

**Five beginner-friendly Deep Learning practicals using Python, Jupyter
Notebook, TensorFlow/Keras, OpenCV, and NLP**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Jupyter](https://img.shields.io/badge/Environment-Jupyter%20Notebook-F37626?logo=jupyter)
![Project](https://img.shields.io/badge/Project-Educational-success)
:::

------------------------------------------------------------------------

## 📌 Project at a Glance

  -----------------------------------------------------------------------
  Item                                Details
  ----------------------------------- -----------------------------------
  Project                             Deep Learning Practical Exam ---
                                      Set C

  Practical count                     5

  Tools                               Python, Jupyter Notebook,
                                      TensorFlow/Keras, OpenCV,
                                      scikit-learn

  Topics                              ANN, image processing, CNN, RNN,
                                      Transformers

  Datasets                            Wine, sample images, MNIST, IMDB

  Goal                                Learn the Deep Learning workflow
                                      through notebook-based exercises
  -----------------------------------------------------------------------

> **Note:** Results depend on the dataset split, model settings, random
> seed, and software environment. Fill in the results table with the
> values from your own notebook runs; no performance values are invented
> here.

## 🎯 Project Objectives

-   Practise data preparation, model building, training, and evaluation.
-   Understand ANN, CNN, RNN, and Transformer architectures.
-   Apply common image-processing techniques with OpenCV.
-   Visualise model performance and inspect prediction errors.
-   Save and reload models where applicable.
-   Keep each practical documented in a reproducible Jupyter Notebook.

## 🗂️ Practical Overview

  -----------------------------------------------------------------------
  Step                    Practical               Main concepts
  ----------------------- ----------------------- -----------------------
  **1**                   ANN --- Wine            DataFrame inspection,
                          Classification          scaling, train/test
                                                  split, Dense layers,
                                                  accuracy, confusion
                                                  matrix

  **2**                   OpenCV --- Image        Colour spaces, masking,
                          Processing              morphology, edge
                                                  detection, face
                                                  detection

  **3**                   CNN --- MNIST Digit     Normalisation, ANN
                          Classification          baseline, convolution,
                                                  pooling, feature maps,
                                                  error analysis

  **4**                   RNN --- IMDB Sentiment  Tokenised reviews,
                          Analysis                padding, embeddings,
                                                  SimpleRNN, binary
                                                  classification

  **5**                   Transformer --- IMDB    Positional embeddings,
                          Sentiment Analysis      self-attention,
                                                  Transformer Encoder,
                                                  evaluation and
                                                  inference
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🧪 Step-by-Step Practicals

## Step 1 --- Artificial Neural Network (ANN): Wine Classification

### Objective

Build an ANN to classify samples from the Wine dataset into one of three
classes.

### Workflow

1.  Import libraries and load the Wine dataset into a pandas DataFrame
    named `df`.
2.  Inspect dataset shape, columns, data types, and class distribution.
3.  Separate features and target labels.
4.  Split the data into training and testing sets.
5.  Scale features using `StandardScaler`.
6.  Encode the target labels for multiclass classification.
7.  Build and train a Keras ANN using Dense layers and ReLU activation.
8.  Use a three-neuron softmax output layer to predict the class.
9.  Evaluate the model with test metrics and a confusion matrix.
10. Save the trained model.

### Concepts

-   Dense neural-network layers
-   Weights, biases, and activation functions
-   Feature scaling and label encoding
-   Multiclass classification
-   Accuracy and confusion-matrix analysis

### Key learning

ANNs learn patterns in structured numerical data by adjusting trainable
weights and biases during training.

------------------------------------------------------------------------

## Step 2 --- OpenCV: Image Processing and Computer Vision

### Objective

Explore image-processing operations and a basic face-detection workflow.

### Workflow

1.  Install and import OpenCV, NumPy, Matplotlib, and any required
    image-data package.
2.  Load and display sample images.
3.  Inspect image dimensions, shape, and data type.
4.  Convert between colour spaces such as BGR/RGB and HSV.
5.  Create a colour mask to isolate a selected colour range.
6.  Apply morphological operations to clean up a mask.
7.  Use Sobel operators to explore image edges.
8.  Run a Haar-cascade face detector when the required cascade file is
    available.
9.  Compare face-detection settings such as `minNeighbors`.
10. Display or save processed outputs as required.

### Techniques

-   Image reading and display
-   BGR/RGB and HSV conversion
-   Colour-based segmentation
-   Erosion and dilation
-   Morphological operations
-   Sobel edge detection
-   Haar-cascade face detection

### Key learning

Image processing transforms or analyses pixel values to reveal useful
visual information. Detection methods identify patterns that match a
detector's features.

**Troubleshooting:** If `ModuleNotFoundError: No module named 'skimage'`
appears, run `%pip install scikit-image` in a notebook cell. If an image
path does not exist, use a provided sample image or upload an image and
update the path.

------------------------------------------------------------------------

## Step 3 --- Convolutional Neural Network (CNN): MNIST Digit Classification

### Objective

Classify handwritten digits from 0 to 9 and compare a CNN with an ANN
baseline.

### Workflow

1.  Load the MNIST dataset.
2.  Display example images and labels.
3.  Normalise pixel values and prepare image dimensions.
4.  Build and evaluate a simple ANN baseline.
5.  Build a CNN using convolution and pooling layers.
6.  Train the CNN and use callbacks such as EarlyStopping where
    configured.
7.  Plot training and validation accuracy/loss.
8.  Generate a confusion matrix and classification report.
9.  Inspect feature maps and misclassified digits.
10. Save the trained CNN model.

### Concepts

-   Image normalisation
-   ANN versus CNN
-   Convolution filters and feature maps
-   Max pooling
-   Overfitting and early stopping
-   Confusion matrices and classification metrics

### Key learning

Convolutional layers learn spatial patterns such as edges and shapes.
Pooling reduces spatial dimensions, and classification layers use
extracted features to predict a digit.

------------------------------------------------------------------------

## Step 4 --- Recurrent Neural Network (RNN): IMDB Sentiment Analysis

### Objective

Classify movie reviews as positive or negative using a SimpleRNN-based
model.

### Workflow

1.  Load the IMDB movie-review dataset.
2.  Inspect encoded reviews and sentiment labels.
3.  Pad or truncate sequences to a fixed length.
4.  Convert token IDs into dense vectors using an Embedding layer.
5.  Build a model with `Embedding`, `SimpleRNN`, and a sigmoid output
    layer.
6.  Train the model and monitor validation performance.
7.  Evaluate the model on held-out test data.
8.  Create a classification report, confusion matrix, and ROC curve
    where included.
9.  Predict sentiment for example reviews.
10. Save the trained model.

### Model outline

`Input tokens → Embedding → SimpleRNN → Dense(sigmoid)`

### Concepts

-   Sequential text data
-   Token IDs and padding
-   Word embeddings
-   Recurrent hidden states
-   Binary cross-entropy
-   Accuracy, precision, recall, F1-score, and ROC-AUC

### Key learning

An RNN processes tokens sequentially and maintains a hidden state that
can capture information from earlier tokens. Long-term dependencies can
be challenging for basic RNNs.

------------------------------------------------------------------------

## Step 5 --- Transformer: IMDB Sentiment Analysis

### Objective

Build a small Transformer Encoder model to classify IMDB reviews as
positive or negative.

### Workflow

1.  Load the IMDB dataset and prepare fixed-length token sequences.
2.  Create token and positional embeddings.
3.  Apply a Transformer Encoder with multi-head self-attention and a
    feed-forward network.
4.  Use residual connections and layer normalisation in the encoder
    block.
5.  Pool the encoded token representations.
6.  Add dropout and a sigmoid output layer.
7.  Train the model and monitor validation loss.
8.  Evaluate test loss, accuracy, and ROC-AUC.
9.  Predict sentiment for example reviews.
10. Save and reload the model, ensuring custom layers are serialisable.

### Model outline

`Input tokens → Token + Position Embeddings → Transformer Encoder → Global Average Pooling → Dropout → Dense(sigmoid)`

### Concepts

-   Token embeddings and positional information
-   Self-attention and multi-head attention
-   Feed-forward networks
-   Residual connections and layer normalisation
-   Binary sentiment classification
-   Model saving and custom-layer serialisation

### Key learning

Self-attention allows each token representation to incorporate
information from other tokens. Unlike a standard RNN, a Transformer can
process sequence positions in parallel during training.

### Implementation notes

-   Use the same `MAX_LEN` and `VOCAB_SIZE` during training and
    inference.
-   Custom layers such as `TokenAndPositionEmbedding` should accept
    Keras configuration arguments through `**kwargs` and implement
    `get_config()`.
-   When loading a saved model, provide custom layers or functions in
    `custom_objects` if they are not automatically registered.
-   A confidence score is not proof that a prediction is correct.

------------------------------------------------------------------------

## 📊 Results and Evaluation

Fill in this table using the results printed by your notebooks.

  -----------------------------------------------------------------------
  Practical               Evaluation to record    Actual result
  ----------------------- ----------------------- -----------------------
  Step 1 --- ANN Wine     Test accuracy and       Add your result
                          confusion matrix        

  Step 2 --- OpenCV       Processed-image outputs Add your observations
                          and detection           
                          observations            

  Step 3 --- CNN MNIST    Test accuracy,          Add your result
                          classification report,  
                          confusion matrix        

  Step 4 --- RNN IMDB     Test accuracy,          Add your result
                          precision, recall,      
                          F1-score, ROC-AUC       

  Step 5 --- Transformer  Test accuracy, ROC-AUC, Add your result
  IMDB                    classification report   
  -----------------------------------------------------------------------

Do not compare models using accuracy alone. Use suitable metrics and
keep the test set separate from training and model selection.

## 🖼️ Screenshots and Visualisations

Create a `screenshots/` folder in your repository, add your actual
notebook screenshots, and update the filenames below as needed.

``` text
screenshots/
├── step1_ann_wine_results.png
├── step2_opencv_outputs.png
├── step3_cnn_training_curves.png
├── step3_cnn_confusion_matrix.png
├── step4_rnn_training_curves.png
├── step4_rnn_confusion_matrix.png
├── step5_transformer_training_curves.png
└── step5_transformer_confidence_chart.png
```

Example Markdown to display a screenshot:

``` markdown
![CNN Training Curves](screenshots/step3_cnn_training_curves.png)
```

## 🧰 Technologies Used

-   **Python** --- programming language
-   **Jupyter Notebook** --- interactive practical environment
-   **NumPy and pandas** --- data handling
-   **Matplotlib** --- plots and visualisations
-   **scikit-learn** --- datasets, preprocessing, data splitting, and
    evaluation
-   **TensorFlow / Keras** --- ANN, CNN, RNN, and Transformer models
-   **OpenCV** --- image processing and computer vision
-   **scikit-image** --- sample images for the OpenCV practical, where
    used

## ⚙️ Installation and Usage

### 1. Create a virtual environment (optional but recommended)

``` bash
python -m venv .venv
```

On Windows:

``` bash
.venv\Scripts\activate
```

### 2. Install the main dependencies

``` bash
python -m pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python scikit-image jupyter
```

Install any additional package required by a specific notebook in the
same Python environment used by the active Jupyter kernel.

### 3. Launch Jupyter Notebook

``` bash
jupyter notebook
```

### 4. Run the practicals

Open each `.ipynb` file and execute its cells from top to bottom. Some
cells may download datasets or model assets and require an internet
connection on first use.

**Compatibility note:** TensorFlow support depends on the Python version
and operating system. If installation or DLL loading fails, consult the
official TensorFlow installation guidance and use a supported
Python/TensorFlow combination.

## 📁 Suggested Repository Structure

``` text
DeepLearning-Set-C/
│
├── README.md
├── DeepLearning_SetC_Step1_ANN_Wine.ipynb
├── DeepLearning_SetC_Step2_OpenCV.ipynb
├── DeepLearning_SetC_Step3_CNN_MNIST.ipynb
├── DeepLearning_SetC_Step4_RNN_IMDB.ipynb
├── DeepLearning_SetC_Step5_Transformers_IMDB.ipynb
│
├── screenshots/
│   ├── step1_ann_wine_results.png
│   ├── step2_opencv_outputs.png
│   ├── step3_cnn_training_curves.png
│   ├── step4_rnn_training_curves.png
│   └── step5_transformer_training_curves.png
│
└── models/
    ├── ann_wine_model.h5
    ├── cnn_mnist_model.h5
    ├── rnn_imdb_sentiment_model.keras
    └── transformer_imdb_sentiment_model.keras
```

This is a **suggested** structure. Include only files that actually
exist in your repository. Large model files and datasets may be better
stored outside Git or managed with Git LFS.

## 🔍 Key Learnings

1.  **ANN:** Dense layers learn relationships between structured
    numerical features.
2.  **OpenCV:** Image-processing techniques transform images and
    highlight useful visual patterns.
3.  **CNN:** Convolution and pooling learn spatial features for image
    classification.
4.  **RNN:** Recurrent layers process sequences and use context from
    previous tokens.
5.  **Transformer:** Self-attention models relationships among tokens
    without sequential recurrence.

## ⚠️ Limitations and Future Improvements

-   Record reproducible metrics and training settings for every model.
-   Use validation data for model selection and reserve the test set for
    final evaluation.
-   Tune model size, learning rate, batch size, and sequence length
    where appropriate.
-   Add error analysis for misclassified images and incorrect sentiment
    predictions.
-   Compare the small Transformer with a pretrained model only if task
    requirements and available compute permit it.
-   Save dependency versions in a `requirements.txt` file for
    reproducibility.

## ✅ Final Conclusion

These five practicals progress through important Deep Learning
approaches: ANN for structured data, OpenCV for image processing, CNN
for image classification, RNN for sequential text, and Transformers for
attention-based language modelling. Together, they demonstrate data
preparation, model building, training, evaluation, visualisation, and
model persistence in a notebook-based workflow.

------------------------------------------------------------------------

## 👨‍💻 Author

**Santosh Verma**

*Deep Learning Practical Project --- Set C*

::: {align="center"}
**Made for learning, experimentation, and practical understanding of
Deep Learning.**
:::
