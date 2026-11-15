# Student Placement Prediction - Streamlit App

## Files
- `app.py`: Streamlit application
- `model (2).pkl`: trained Gaussian Naive Bayes model
- `scaler (5).pkl`: fitted StandardScaler
- `requirements.txt`: Python dependencies

## Run locally
1. Install Python 3.10, 3.11, or 3.12.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

Keep both pickle files in the same folder as `app.py`.

## Important label check
The app currently displays class `0` as **Not Placed** and class `1` as **Placed**.
Verify this matches the target encoding used when training your model. If your dataset
used the reverse encoding, swap the labels in `label_map` inside `app.py`.

The pickle files were saved with scikit-learn 1.6.1, so the requirements pin that
version to reduce model-loading compatibility issues. Only load pickle files you trust.
