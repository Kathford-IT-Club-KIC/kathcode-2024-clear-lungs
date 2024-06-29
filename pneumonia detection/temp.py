from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import mysql.connector
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from my_module import YourModelClass  # Import your PyTorch model class

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'webdev',
    'database': 'pneumonia_detection'
}

# Load the PyTorch model
model_path = 'your_pytorch_model.pth'  # Update with your PyTorch model path
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Define the model architecture and load state dictionary
if 'model' in checkpoint:
    model = checkpoint['model']
else:
    model = YourModelClass()  # Instantiate your model class
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()  # Set model to evaluation mode

# Set upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Ensure 3 channels
])

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Route for handling the index redirecting to login
@app.route('/')
def index():
    return redirect(url_for('login'))

# Route for handling login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['user'] = email
            return redirect(url_for('main'))
        else:
            error = 'Invalid email or password'
            return render_template('login.html', error=error)
    
    return render_template('login.html')

# Route for handling main page
@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    return render_template('main.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return render_template('main.html', result='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('main.html', result='No selected file')
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict pneumonia using PyTorch model
        img = Image.open(filepath).convert('L')  # Ensure image is grayscale
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(torch.device('cpu'))  # Ensure model runs on CPU
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.sigmoid(output).item()
        
        result = 'Pneumonia Detected' if prediction > 0.5 else 'No Pneumonia'
        
        return render_template('main.html', result=result)

# Route for serving uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
