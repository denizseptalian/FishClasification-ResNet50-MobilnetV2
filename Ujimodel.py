import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

# Load the model
model = tf.keras.models.load_model("MobileNet_model.h5")

# Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "path_to_test_data",  # Replace with the path to your test dataset
    image_size=(224, 224),  # Adjust size if needed
    batch_size=32,
    shuffle=False
)

# Extract labels and file paths
true_labels = np.concatenate([y.numpy() for _, y in test_dataset])

# Predict on the test dataset
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Model Accuracy on Test Dataset: {accuracy:.2%}")
