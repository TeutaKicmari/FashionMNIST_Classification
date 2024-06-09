from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('fashion_mnist_cnn_model.h5')

# Define the folder containing images
folder_path = r"C:\Users\Lenovo\Desktop\Imazhet"

# Define class names for prediction
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Collect all image filenames
images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# Determine the number of images and rows needed for a 3-pairs-per-row layout
num_images = len(images)
num_rows = (num_images + 2) // 3  # Calculate rows needed for groups of three

# Create figure with subplots
fig, axs = plt.subplots(nrows=num_rows, ncols=6, figsize=(15, num_rows * 3))  # 6 columns because 2 per image pair

for idx, filename in enumerate(images):
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path)

    # Resize image to 28x28 for the model input
    image_resized = image.resize((28, 28))

    # Convert to grayscale
    image_gray = image_resized.convert('L')

    # Invert and normalize pixel values to match Fashion MNIST preprocessing
    image_array = np.array(image_gray)
    image_array_normalized = (255 - image_array) / 255.0

    # Reshape for the model input and make prediction
    image_for_model = image_array_normalized.reshape(1, 28, 28, 1)  # Reshape to match CNN input shape
    prediction = model.predict(image_for_model)
    predicted_class = class_names[np.argmax(prediction)]

    # Calculate row and column index for subplots
    row_idx = idx // 3
    col_idx = (idx % 3) * 2  # Each image pair takes up 2 columns

    # Display original image
    ax = axs[row_idx, col_idx]
    ax.imshow(image, cmap='gray')
    ax.set_title('Original')
    ax.axis('off')
    for spine in ax.spines.values():  # Add border to subplot
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Display processed image and prediction
    ax = axs[row_idx, col_idx + 1]
    ax.imshow(image_array_normalized, cmap='gray')
    ax.set_title(f'Processed: {predicted_class}')
    ax.axis('off')
    for spine in ax.spines.values():  # Add border to subplot
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

plt.tight_layout()
plt.show()
