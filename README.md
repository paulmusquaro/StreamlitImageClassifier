# StreamlitImageClassifier


###### Image Classification with Streamlit

This project provides a web-based interface to classify images using pre-trained deep learning models. The application uses **Streamlit** for the frontend and supports two models: a simple CNN and a fine-tuned VGG16 model.

---

## Features

- Upload images in JPG, JPEG, or PNG format.
- Choose between two classification models:
  - **CNN**: A Convolutional Neural Network trained on the Fashion MNIST dataset.
  - **VGG16**: A fine-tuned VGG16 model for enhanced accuracy.
- Display:
  - Uploaded image preview.
  - Predicted class and its probability.
  - Probability distribution for all classes.
- Interactive charts for visualizing class probabilities.

---

## Getting Started

### Prerequisites

- **Python 3.9** or later.
- Docker (optional, for containerized deployment).

---

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/paulmusquaro/StreamlitImageClassifier.git
cd StreamlitImageClassifier
```

#### 2. Set up the environment

##### Using Conda:

To make the project reproducible and ensure smooth package management, this project uses Conda as a package and environment manager. Below are the steps to set up the environment:


1. **Install Conda**:
If you haven't installed Conda yet, you can download it from the official [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) websites. Anaconda is a larger distribution with more pre-installed packages, while Miniconda is a smaller, minimal version. Choose whichever suits your needs.

2. **Create a new environment:** Open your terminal and run the following command to create a new Conda environment with Python 3.9:

    ```bash
    conda create --name new_conda_env python=3.9
    ```

3. **Activate the environment:** Once the environment is created, activate it by running:

    ```bash
    conda activate new_conda_env
    ```

4. **Install required packages (Jupyter, NumPy, MatPlotLib, Tensorflow, Keras, Streamlit and Pillow)**

    ```bash
    conda install jupyter numpy matplotlib tensorflow keras streamlit pillow
    ```

5. **Run Jupyter Notebook**

    ```bash
    jupyter notebook
    ```



##### Using Docker:
1. Build the Docker image:
```bash
docker build -t streamlit_app:v1.0 .
```
2. Run the container:
```bash
docker-compose up
```

---

### Running the Application

#### Using Python:
Start the Streamlit app:
```bash
streamlit run main.py
```
The app will be available at `http://localhost:8501`.

#### Using Docker:
After running `docker-compose up`, visit `http://localhost:8501` in your browser.

---

## File Structure

```
.
├── main.py            # Main Streamlit application
├── requirements.txt   # Python dependencies
├── models/            # Directory containing pre-trained models
├── Dockerfile         # Dockerfile for containerized deployment
├── docker-compose.yml # Docker Compose configuration
└── README.md          # Project documentation
```

---

## Models

1. **CNN**:
   - Input size: 28x28 grayscale images.
   - Trained on: Fashion MNIST dataset.
   - Model file: `models/fashion_mnist_cnn_model.h5`.

2. **VGG16**:
   - Input size: 32x32 RGB images.
   - Fine-tuned for fashion classification.
   - Model file: `models/vgg16_fashion_mnist_fine_tuned.h5`.

---

## Environment Variables

- `KMP_DUPLICATE_LIB_OK=TRUE`: Required to avoid runtime errors related to TensorFlow.

---

## Deployment Notes

The `Dockerfile` is based on `python:3.9-slim` to minimize the image size. The `docker-compose.yaml` exposes port `8501` for the Streamlit app and ensures the container restarts automatically.


---

## Troubleshooting

- **Error loading models**: Ensure the model files exist in the `models/` directory and match the filenames in `main.py`.
- **TensorFlow compatibility issues**: Verify that the TensorFlow version in `requirements.txt` is compatible with your Python environment.

