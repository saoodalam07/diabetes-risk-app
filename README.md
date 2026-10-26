# 🩺 Diabetes Risk Predictor

KNN model (Pima Indians Diabetes, 768 patients) → **FastAPI** `/predict` → **Streamlit** UI.
The UI talks to the API over HTTP only; it never loads `model.pkl`.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py            # creates model.pkl (already included)
uvicorn main:app --reload        # terminal 1 → http://127.0.0.1:8000/docs
streamlit run streamlit_app.py   # terminal 2 → http://localhost:8501
```

## What the app does
| Feature | What it shows |
|---|---|
| Prediction + gauge | Diabetic / Not Diabetic, confidence (`st.metric`), risk gauge, 5-neighbour vote |
| Clinical context | Glucose (OGTT), BMI, diastolic BP, age, family-history bands |
| What would change it? | Re-asks the API with one value set to the typical non-diabetic level → biggest lever |
| What-if explorer | Sweeps one feature through the API and plots where the prediction flips |
| Patients like you | 5 most similar real patients from the CSV + population percentile chart |
| Quick-fill | Low / borderline / high-risk profiles, or a random real patient (with true outcome) |
| History | Every prediction this session, downloadable as CSV |
| Graceful failure | Friendly "start FastAPI first" message, live API status in the sidebar |

## Deploy (free)
1. **API → Render** (New → Web Service, connect the repo)
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
2. **UI → Streamlit Community Cloud** (share.streamlit.io → New app → `streamlit_app.py`)
   - Settings → Secrets:
     ```toml
     API_URL = "https://your-api.onrender.com"
     ```

The app defaults to `http://127.0.0.1:8000` and switches to `API_URL` from secrets / env vars automatically.

> Educational demo — not a medical diagnosis.
