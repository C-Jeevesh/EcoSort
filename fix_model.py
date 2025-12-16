import pickle
from sklearn import svm
import numpy as np

# Create a temporary compatible model
# (This prevents crashes, but requires real training data for high accuracy)
print("Creating a compatible model structure...")
X_dummy = np.random.rand(10, 3780) # Matches HOG feature size
y_dummy = ['R', 'O'] * 5 
model = svm.SVC(probability=True)
model.fit(X_dummy, y_dummy)

with open('ecosort_svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Fixed! Your app will now run without crashing.")