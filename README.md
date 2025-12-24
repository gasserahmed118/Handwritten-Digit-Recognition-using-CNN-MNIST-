🧠 Handwritten Digit Recognition using CNN (MNIST)

📌 Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset. The model is trained on grayscale images of size 28×28 pixels and achieves high accuracy through effective feature extraction using convolutional layers. The project demonstrates a complete deep learning workflow from data preprocessing to model evaluation and real-world prediction.

🎯 Objectives
   1. Build an end-to-end CNN model for digit classification
   2. Accurately recognize handwritten digits (0–9)
   3. Visualize training performance and classification results
   4.Predict digits from manually created images

📂 Dataset
   1. Dataset Name: MNIST Handwritten Digits
   2. Training Samples: 42,000
   3. Test Samples: 28,000
   4. Image Size: 28×28 (grayscale)
   5. Classes: 10 (digits 0–9)

🏗 Project Structure

    📦 mnist-cnn-digit-recognition
     ┣ 📜 README.md
     ┣ 📓 MNIST_CNN.ipynb
     ┣ 📁 data
     ┃ ┣ 📄 train.csv
     ┃ ┗ 📄 test.csv
     ┣ 📁 results
     ┃ ┣ 📊 confusion_matrix.png
     ┃ ┣ 📊 training_accuracy.png
     ┃ ┗ 📊 training_loss.png
     ┗ 📁 models

🔎 Exploratory Data Analysis (EDA)

     1. Dataset shape inspection
     2. Null value checking
     3. Label distribution analysis
     4. Sample digit visualization

    Data validation confirmed:
     1. No missing values
     2. Balanced label distribution

🧩 Data Preprocessing

    1. Normalized pixel values to range [0,1]
    2. Reshaped images to (28, 28, 1) for CNN input
    3. Converted labels to one-hot encoded vectors
    4. Split training data into training and validation sets (90% / 10%)     


🧠 CNN Model Architecture
   1. Conv2D + ReLU (feature extraction)
   2. MaxPooling (spatial downsampling)
   3. Dropout (overfitting prevention)
   4. Flatten
   5. Dense layers
   6. Softmax output layer

Optimizer: RMSprop

Loss Function: Categorical Cross-Entropy

Metric: Accuracy    

     
🚀 Model Training
   1. Batch size: 85
   2. Epochs: 30
   3. Learning rate reduction using ReduceLROnPlateau
   4. Data augmentation used to improve generalization

📊 Evaluation & Visualization

    1. Training and validation accuracy/loss curves
    2. Confusion matrix for multi-class classification
    3. Error analysis of misclassified digits
    4. Visualization of most confusing predictions


📈 Key Results
   1. Achieved >99% accuracy on validation data
   2. CNN effectively learned spatial features
   3. Strong generalization to manually drawn digits
   4. Error analysis highlights visually similar digits


🛠 Technologies Used
   1. Python
   2. TensorFlow / Keras
   3. NumPy
   4. Pandas
   5. Matplotlib
   6. OpenCV
   7. Scikit-learn
   8. Jupyter Notebook    
