import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Secret key configuration
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'media'

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Custom Jinja filter for JSON parsing
@app.template_filter('from_json')
def from_json_filter(value):
    if value:
        return json.loads(value)
    return {}

# Load the trained model from local file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'resnet50_model.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure resnet50_model.h5 is in the model/ directory.")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

# Class labels in the correct order
CLASS_LABELS = ['cataract', 'glaucoma', 'normal', 'diabetic_retinopathy']

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if uploaded file has an allowed image extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)
    patients = db.relationship('Patient', backref='doctor', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())
    eye_analyses = db.relationship('EyeAnalysis', backref='patient', lazy=True, cascade='all, delete-orphan')

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def get_analysis(self, eye_side):
        """Get analysis for specific eye (left or right)"""
        return EyeAnalysis.query.filter_by(patient_id=self.id, eye_side=eye_side).first()

class EyeAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    eye_side = db.Column(db.String(5), nullable=False)  # 'left' or 'right'
    image_filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=True)  # Result of analysis
    confidence = db.Column(db.Float, nullable=True)  # Confidence percentage
    all_probabilities = db.Column(db.Text, nullable=True)  # JSON string with all class probabilities
    analyzed_at = db.Column(db.DateTime, nullable=True)  # When analysis was performed

    @property
    def is_analyzed(self):
        return self.prediction is not None

def preprocess_image(image_path):
    """
    Preprocess the uploaded image exactly as in the training code
    Parameters:
    image_path (str): Path to the saved image file
    """
    img = load_img(image_path, target_size=(224, 224))  # Loads as RGB
    img_array = img_to_array(img)
    img_array = img_array[..., ::-1]  # Convert RGB to BGR (matching training with cv2.imread)
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

    # Get current doctor (user)
    doctor = User.query.filter_by(username=session["username"]).first()

    # Get all patients for this doctor
    patients = Patient.query.filter_by(doctor_id=doctor.id).order_by(Patient.created_at.desc()).all()

    return render_template("dashboard.html", username=session["username"], patients=patients)

@app.route("/patients/create", methods=["GET", "POST"])
def create_patient():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")

        if not first_name or not last_name:
            flash("First name and last name are required", "error")
            return render_template("create_patient.html", username=session["username"])

        # Get current doctor
        doctor = User.query.filter_by(username=session["username"]).first()

        # Create new patient
        new_patient = Patient(
            doctor_id=doctor.id,
            first_name=first_name.strip(),
            last_name=last_name.strip()
        )
        db.session.add(new_patient)
        db.session.commit()

        flash(f"Patient {new_patient.full_name} created successfully!", "success")
        return redirect(url_for("patient_detail", patient_id=new_patient.id))

    return render_template("create_patient.html", username=session["username"])

@app.route("/patients/<int:patient_id>")
def patient_detail(patient_id):
    if "username" not in session:
        return redirect(url_for("login"))

    # Get current doctor
    doctor = User.query.filter_by(username=session["username"]).first()

    # Get patient (ensure it belongs to this doctor)
    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor.id).first_or_404()

    # Get analyses for both eyes
    left_eye = patient.get_analysis('left')
    right_eye = patient.get_analysis('right')

    return render_template("patient_detail.html",
                         username=session["username"],
                         patient=patient,
                         left_eye=left_eye,
                         right_eye=right_eye)

@app.route("/patients/<int:patient_id>/upload/<eye_side>", methods=["POST"])
def upload_eye_image(patient_id, eye_side):
    if "username" not in session:
        return redirect(url_for("login"))

    if eye_side not in ['left', 'right']:
        flash("Invalid eye side", "error")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    # Get current doctor
    doctor = User.query.filter_by(username=session["username"]).first()

    # Get patient (ensure it belongs to this doctor)
    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor.id).first_or_404()

    if 'file' not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    if not allowed_file(file.filename):
        flash("Invalid file format. Please upload an image (PNG, JPG, JPEG, BMP, TIFF, WEBP)", "error")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    try:
        # Create patient directory
        patient_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'patients', str(patient_id))
        os.makedirs(patient_dir, exist_ok=True)

        # Save with standardized filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{eye_side}.{ext}"
        filepath = os.path.join(patient_dir, filename)
        file.save(filepath)

        # Check if analysis record exists for this eye
        eye_analysis = patient.get_analysis(eye_side)

        if eye_analysis:
            # Update existing record
            eye_analysis.image_filename = filename
            eye_analysis.prediction = None  # Reset analysis
            eye_analysis.confidence = None
            eye_analysis.all_probabilities = None
            eye_analysis.analyzed_at = None
        else:
            # Create new analysis record
            eye_analysis = EyeAnalysis(
                patient_id=patient.id,
                eye_side=eye_side,
                image_filename=filename
            )
            db.session.add(eye_analysis)

        db.session.commit()
        flash(f"{eye_side.capitalize()} eye image uploaded successfully!", "success")

    except Exception as e:
        flash(f"Error uploading image: {str(e)}", "error")

    return redirect(url_for("patient_detail", patient_id=patient_id))

@app.route("/patients/<int:patient_id>/analyze/<eye_side>", methods=["POST"])
def analyze_eye(patient_id, eye_side):
    if "username" not in session:
        return redirect(url_for("login"))

    if eye_side not in ['left', 'right']:
        flash("Invalid eye side", "error")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    # Get current doctor
    doctor = User.query.filter_by(username=session["username"]).first()

    # Get patient (ensure it belongs to this doctor)
    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor.id).first_or_404()

    # Get eye analysis record
    eye_analysis = patient.get_analysis(eye_side)

    if not eye_analysis:
        flash(f"No image uploaded for {eye_side} eye", "error")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    try:
        # Get image path
        patient_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'patients', str(patient_id))
        filepath = os.path.join(patient_dir, eye_analysis.image_filename)

        if not os.path.exists(filepath):
            flash(f"Image file not found for {eye_side} eye", "error")
            return redirect(url_for("patient_detail", patient_id=patient_id))

        # Preprocess and analyze
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

        # Update analysis record
        eye_analysis.prediction = result
        eye_analysis.confidence = confidence
        eye_analysis.all_probabilities = json.dumps(class_probabilities)
        eye_analysis.analyzed_at = datetime.utcnow()

        db.session.commit()
        flash(f"{eye_side.capitalize()} eye analyzed successfully!", "success")

    except Exception as e:
        flash(f"Error analyzing image: {str(e)}", "error")

    return redirect(url_for("patient_detail", patient_id=patient_id))

@app.route("/media/patients/<int:patient_id>/<filename>")
def serve_patient_image(patient_id, filename):
    """Serve uploaded patient eye images"""
    if "username" not in session:
        return redirect(url_for("login"))

    # Get current doctor
    doctor = User.query.filter_by(username=session["username"]).first()

    # Verify patient belongs to this doctor (security check)
    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor.id).first()
    if not patient:
        flash("Unauthorized access", "error")
        return redirect(url_for("dashboard"))

    # Serve the image file
    patient_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'patients', str(patient_id))
    return send_from_directory(patient_dir, filename)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
