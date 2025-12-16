# train_model.py
import numpy as np
import pickle
from sklearn import svm
from skimage.feature import hog
import cv2
import os

print("🔄 Retraining dummy model for compatibility...")

# 1. Create dummy data (Just to initialize the correct model structure)
# We train it on random noise just to save the *architecture* # (Since we can't download your Colab dataset here easily)
# IF YOU WANT THE REAL MODEL: You must keep the Colab one, but ignore the warning.
# THIS SCRIPT IS A FALLBACK if the Colab model crashes.

X_dummy = np.random.rand(10, 3780) # HOG feature size
y_dummy = ['R', 'O'] * 5 

# 2. Train Model
model = svm.SVC(probability=True)
model.fit(X_dummy, y_dummy)

# 3. Save
with open('ecosort_svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ New 'ecosort_svm_model.pkl' created! (Note: This is a dummy model. For real accuracy, move your Colab dataset here and train on that, or ignore the warning if the old model still predicts correctly.)")