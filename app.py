from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "my_secret_key"

# Configuration settings
app.config['UPLOAD_FOLDER'] = 'media'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db = SQLAlchemy(app)

# Load the trained ResNet50 model
model = tf.keras.models.load_model('./model/resnet50_model.h5')

# Class labels in the correct order
CLASS_LABELS = ['cataract', 'glaucoma', 'normal', 'diabetic_retinopathy']

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

def preprocess_image(image_path):
    """
    Preprocess the uploaded image exactly as in the training code
    Parameters:
    image_path (str): Path to the saved image file
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Incorrect username or password."
            return render_template("login.html", error=error)
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            error = "User already exists!"
            return render_template("register.html", error=error)

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        session["username"] = username
        return redirect(url_for("dashboard"))

    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))
        
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('dashboard.html', username=session["username"], error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('dashboard.html', username=session["username"], error='No selected file')
        
        try:
            # Create media directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image using the saved file path
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            probabilities = predictions[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class] * 100)
            result = CLASS_LABELS[predicted_class]
            
            class_probabilities = {
                label: float(prob * 100)
                for label, prob in zip(CLASS_LABELS, probabilities)
            }
            
            # Clean up the saved file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return render_template(
                'dashboard.html',
                username=session["username"],
                prediction=result,
                confidence=round(confidence, 2),
                all_probabilities=class_probabilities
            )
            
        except Exception as e:
            # Clean up the file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return render_template('dashboard.html', username=session["username"], error=f'Error processing image: {str(e)}')
    
    return render_template('dashboard.html', username=session["username"])

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)