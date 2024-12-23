# Age and Gender Detection using UTKFace Dataset

This project implements an age and gender detection model using the UTKFace dataset. The model is built using TensorFlow and Keras.

## Dataset

The UTKFace dataset is used for training and testing the model. It contains over 20,000 face images with annotations for age, gender, and ethnicity. The dataset can be downloaded from [here](https://susanqq.github.io/UTKFace/).

**Note:** Due to the size of the dataset, it is recommended to download it and place it in your Google Drive before running the code. This is done by setting the `fldr` variable in the provided code.

## Model Architecture

The model uses a convolutional neural network (CNN) architecture with the following layers:

1.  **Input Layer:** Takes image inputs of size 48x48 with 3 color channels (RGB).
2.  **Convolutional Layers:** Several Conv2D layers with ReLU activation and Batch Normalization are used to extract features from the images. Each Conv2D layer is followed by a Dropout layer to prevent overfitting and a MaxPooling2D layer to reduce the dimensionality.
3.  **Flatten Layer:** Flattens the output of the last MaxPooling2D layer for feeding into fully connected layers.
4.  **Fully Connected Layers:**
    *   Two Dense layers with ReLU activation are used, followed by a Dropout layer to each.
    *   Two output layers: one for gender classification (sigmoid activation) and one for age regression (ReLU activation).
5.  **Model Compilation:** The model is compiled with `Adam` optimizer and `binary_crossentropy` loss for gender classification, `mae` for age regression, and both are tracked with `accuracy`.

## Dependencies

*   tensorflow
*   opencv-python
*   numpy
*   matplotlib
*   scikit-learn
*   seaborn

Install the dependencies using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn
content_copy
download
Use code with caution.
Markdown
Usage

Clone the Repository:

git clone <repository-url>
cd <repository-name>
content_copy
download
Use code with caution.
Bash

Prepare the Dataset:

Download the UTKFace dataset and place the image folder in your Google Drive.

Update the fldr variable in the notebook to reflect the path of your UTKFace dataset folder in Google Drive.

fldr="/content/drive/MyDrive/UTKFace"
content_copy
download
Use code with caution.
Python

Run the Jupyter Notebook:

Open the Age_Detection(UTKFace_Dataset).ipynb notebook, in Colab or your preferred environment, with GPU acceleration enabled.

Run the cells sequentially.

The notebook will:
* Load and preprocess the dataset.
* Split the dataset into training and test sets.
* Define and train the model.
* Evaluate the model performance.
* Save the trained model with the name: Age_sex_detection.keras.
* Generate visualizations for the training history, age prediction and gender distribution.
* Test the model on a sample image.

Results

The notebook provides the following performance metrics:

Age R2 Score: R-squared value of the age prediction, indicating the fitness of the age prediction.

Model Loss and Accuracy Plots: Visualize how loss and accuracy change over training epochs.

Confusion matrix: Shows the prediction performance on the test data, how well the trained model does in classifying different ages.

Sample Predictions: Displays the age and gender predicted for a sample image.

Notes

The model is trained with 100 epochs, you can experiment with different number of epochs.

The default batch size is set to 64.

Early stopping is implemented to prevent overfitting.

The model is saved using keras.

The test_image function can be used to test the model on any image in the dataset by passing the respective index of the image.

Future Work

Implement data augmentation techniques to improve model performance.

Experiment with different model architectures and hyperparameters.

Explore other optimization algorithms.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

content_copy
download
Use code with caution.
