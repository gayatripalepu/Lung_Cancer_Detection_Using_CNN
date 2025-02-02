from flask import Flask, render_template, request,send_from_directory, session, redirect, url_for
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Save the file to the uploads folder
    filename = file.filename
    file.save(os.path.join('uploads', filename))
    session['filename'] = filename  # Store the filename in the session
    
    class_names = {0: 'Lung benign tissue', 1: 'Lung adenocarcinoma', 2:'Lung squamous cell carcinoma'}
    model = tf.keras.models.load_model('lungmodel.h5')
    # Load the test image
    test_image_path = f'uploads/{filename}'
    test_image = Image.open(test_image_path)
    
    # Preprocess the image

    img = image.load_img(test_image_path, target_size=(256, 256))  # Adjust target_size based on your model's input shape
    img_array = image.img_to_array(img)
    # Get the input shape of the first layer
    input_shape = model.input_shape  # or model.layers[0].input_shape

    # Extract the number of channels from the last dimension
    num_channels = input_shape[-1]
    # Convert RGB image to grayscale if the model expects grayscale input
    #if model.layers[0].input_shape[-1] == 1:
    if num_channels == 1:
        img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255  # Normalize the image data

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Map predicted class to actual class label based on your dataset
    
    class_labels = {0: 'Bengin cases', 1: 'Malignant cases', 2: 'Normal cases'}  # Update with your class labels
    predicted_class_label = class_labels[predicted_class[0]]

    print("Predicted class for the image: ", predicted_class_label)
    print("Predicted Class:", predicted_class_label)
    session['text'] = predicted_class_label  # Store the text in the session
    return redirect(url_for('view_image'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/view_image')
def view_image():
    filename = session.get('filename')
    text = session.get('text')
    if not filename:
        return 'No image uploaded'
    
        

    return render_template('viewimage.html', filename=filename, text=text)


if __name__ == '__main__':
    app.run(debug=True)
