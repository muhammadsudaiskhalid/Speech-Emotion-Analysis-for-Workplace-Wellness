from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import librosa
import os
from werkzeug.utils import secure_filename
import uuid
import tempfile
import signal
import threading
import time
from contextlib import contextmanager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'voice_analyzer_model.pkl'
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Emotion mapping to ensure consistent output
EMOTION_LABELS = {
    'neutral': 'Neutral',
    'calm': 'Calm', 
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'fearful': 'Fearful',
    'disgust': 'Disgust',
    'surprised': 'Surprised'
}

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling"""
    def timeout_handler():
        time.sleep(seconds)
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()

def extract_features(file_path):
    """Extract audio features consistent with training"""
    try:
        print(f"🎵 Loading audio file: {file_path}")
        
        # Check if file exists and is not empty
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError("Audio file is empty")
        
        # Use timeout for audio loading to prevent hanging
        with timeout(30):  # 30 second timeout
            try:
                # Load audio file with error handling
                y, sr = librosa.load(file_path, sr=22050, duration=30)
            except Exception as e:
                print(f"❌ Librosa load error: {e}")
                # Try alternative loading method
                import soundfile as sf
                try:
                    y, sr = sf.read(file_path)
                    if sr != 22050:
                        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                        sr = 22050
                except:
                    raise ValueError(f"Could not load audio file: {e}")
        
        if len(y) == 0:
            raise ValueError("Audio file is empty or corrupted")
        
        print(f"🎵 Audio loaded - Length: {len(y)}, Sample rate: {sr}")
        
        # Extract features with error handling
        try:
            # Extract MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            # Extract Chroma features (12 coefficients)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)
            
            # Extract Spectral Contrast features (7 coefficients)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast.T, axis=0)
            
            # Concatenate all features (13 + 12 + 7 = 32 features)
            features = np.concatenate([mfccs_mean, chroma_mean, contrast_mean])
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError("Features contain NaN or infinite values")
            
            print(f"🎵 Features extracted - Shape: {features.shape}")
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            raise ValueError(f"Failed to extract features: {e}")
        
    except TimeoutError:
        print("❌ Audio processing timed out")
        raise TimeoutError("Audio processing timed out")
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("🔍 /analyze endpoint hit")
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload a valid audio file.'}), 400
    
    file_path = None
    try:
        # Create temporary file
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        # Save uploaded file
        file.save(file_path)
        print(f"📁 File saved: {file_path}")
        
        # Verify file exists and has content
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise ValueError("File is empty or was not saved properly")
        
        # Extract features with timeout protection
        try:
            features = extract_features(file_path)
        except TimeoutError:
            return jsonify({'error': 'Audio processing timed out. Please try a shorter audio file.'}), 408
        except Exception as e:
            return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 400
        
        # Make prediction
        try:
            prediction = model.predict(features)
            emotion_raw = str(prediction[0]).lower()
            
            # Map to proper English format
            emotion = EMOTION_LABELS.get(emotion_raw, emotion_raw.capitalize())
            
            print(f"✅ Raw prediction: {emotion_raw}")
            print(f"✅ Formatted emotion: {emotion}")
            
            response = {'emotion': emotion}
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🧹 Cleaned up: {file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
    
    return jsonify(response)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': '✅ Flask is working!', 
        'model_loaded': model is not None,
        'available_emotions': list(EMOTION_LABELS.values())
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📝 Make sure you have:")
    print("   1. voice_analyzer_model.pkl in the same directory")
    print("   2. templates/index.html file")
    print("   3. static/style.css file")
    
    # Run with debug=False to prevent file watcher issues
    # Use threaded=True for better performance
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)