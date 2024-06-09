import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the data
train_data = pd.read_csv(r"C:\Users\Lenovo\Desktop\fashion-mnist_train.csv")
test_data = pd.read_csv(r"C:\Users\Lenovo\Desktop\fashion-mnist_test.csv")

# Separate features and target labels
X_train = train_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
y_test = test_data['label'].values

# Normalize the pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Save the model
model.save('fashion_mnist_cnn_model.h5')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

# Predict classes
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

# Classification report
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_report = classification_report(actual_classes, predicted_classes, target_names=class_names)
print("Classification Report:\n", class_report)

# Visualization of the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Visualization of the predictions
indices = np.random.choice(range(len(X_test)), 10, replace=False)

plt.figure(figsize=(10, 5))
for i, index in enumerate(indices):
    ax = plt.subplot(2, 5, i + 1)
    img = X_test[index].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {class_names[predicted_classes[index]]}\nActual: {class_names[actual_classes[index]]}')
    plt.axis('off')
plt.show()
