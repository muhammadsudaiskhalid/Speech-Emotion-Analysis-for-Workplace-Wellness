import zipfile
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from collections import Counter

# Step 1: Extract the RAVDESS ZIP file
zip_file_path = 'ravdess.zip'
extract_path = 'ravdess_dataset'

print("🗂️ Extracting dataset...")
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted to:", extract_path)
except FileNotFoundError:
    print(f"❌ Error: {zip_file_path} not found. Please download the RAVDESS dataset.")
    exit(1)

# Step 2: Define emotion map (RAVDESS emotion codes)
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Step 3: Feature Extraction Function (32 features total)
def extract_features(file_path):
    """Extract audio features consistently"""
    try:
        # Load audio file with consistent parameters
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        if len(y) == 0:
            print(f"⚠️ Empty audio file: {file_path}")
            return None
        
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
        
        return features
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Step 4: Load and Label Dataset
print("🎵 Loading and processing audio files...")
X, y = [], []
file_count = 0

for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.wav'):
            file_count += 1
            try:
                # Parse filename: "03-01-06-01-02-01-12.wav"
                parts = file.split('-')
                if len(parts) < 3:
                    print(f"⚠️ Skipping file with unexpected format: {file}")
                    continue
                
                emotion_code = parts[2]  # Extract '06' etc.
                label = emotion_map.get(emotion_code, 'unknown')
                
                if label == 'unknown':
                    print(f"⚠️ Unknown emotion code '{emotion_code}' in file: {file}")
                    continue
                
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                
                if features is not None:
                    X.append(features)
                    y.append(label)
                    
                    if len(X) % 100 == 0:
                        print(f"📊 Processed {len(X)} files...")
                        
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")

print(f"✅ Total files found: {file_count}")
print(f"✅ Total samples loaded: {len(X)}")

if len(X) == 0:
    print("❌ No valid samples found. Please check your dataset.")
    exit(1)

# Show emotion distribution
emotion_counts = Counter(y)
print("\n📊 Emotion distribution:")
for emotion, count in sorted(emotion_counts.items()):
    print(f"   {emotion}: {count} samples")

# Step 5: Prepare data for training
X = np.array(X)
y = np.array(y)

print(f"\n🎯 Feature matrix shape: {X.shape}")
print(f"🎯 Labels shape: {y.shape}")

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")

# Step 7: Train the Random Forest Model
print("\n🌲 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)
print("✅ Model training completed!")

# Step 8: Evaluate the Model
print("\n🔍 Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")

print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Feature importance
feature_names = (
    [f'mfcc_{i}' for i in range(13)] + 
    [f'chroma_{i}' for i in range(12)] + 
    [f'contrast_{i}' for i in range(7)]
)

feature_importance = model.feature_importances_
top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

print("\n🔝 Top 10 Most Important Features:")
for i, (feature, importance) in enumerate(top_features[:10]):
    print(f"   {i+1}. {feature}: {importance:.4f}")

# Step 10: Save Model
model_path = 'voice_analyzer_model.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Model saved as '{model_path}'")

# Step 11: Test prediction on a sample
print("\n🧪 Testing model prediction...")
sample_features = X_test[0].reshape(1, -1)
sample_prediction = model.predict(sample_features)
actual_emotion = y_test[0]

print(f"   Sample prediction: {sample_prediction[0]}")
print(f"   Actual emotion: {actual_emotion}")
print(f"   Prediction correct: {sample_prediction[0] == actual_emotion}")

print("\n🎉 Model training and evaluation completed successfully!")
print("📝 You can now run your Flask app with the trained model.")