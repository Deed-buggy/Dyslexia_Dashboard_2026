import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dyslexia Detection Research Dashboard", layout="wide")

st.title("Dyslexia Detection Research Dashboard")
st.write("Literature Review of High-Accuracy Machine Learning Models for Dyslexia Detection")

data = {

"Article":[
"Deep Learning Driven Dyslexia Detection – PeerJ CS (2024)",
"INSIGHT Eye Tracking Model – Dyslexia Journal (2025)",
"Eye Movement + Demographic AI – PLOS ONE (2023)",
"ML Model for Students – Scientific Reports (2024)",
"CNN Handwriting Detection – JDR (2024)",
"DyslexiaNet CNN – JEMR (2025)",
"YOLO Handwriting Detection (2025)",
"Akshar Mitra Multimodal Framework (2025)"
],

"Dataset Description":[
"MRI, fMRI, EEG neuroimaging datasets",
"Eye tracking fixation dataset (35 children)",
"Eye movement reading dataset + demographic features",
"Questionnaire dataset (~1200 students)",
"Handwriting image dataset",
"EOG eye movement signals during reading",
"Synthetic handwriting dataset",
"Eye tracking + speech + handwriting dataset"
],

"Preprocessing":[
"Cleaning, normalization, feature extraction",
"Fixation visualization, normalization",
"Feature engineering, scaling",
"Data cleaning and encoding",
"Image normalization, augmentation",
"Signal preprocessing and scalogram conversion",
"Image segmentation and synthetic generation",
"Multimodal feature extraction"
],

"Algorithm / Model":[
"MobileNetV3 + EfficientNet + LightGBM",
"ResNet18 CNN",
"Random Forest / MLP / Gradient Boosting",
"SVM / ANN",
"CNN",
"DyslexiaNet CNN",
"YOLOv11",
"Multimodal ML Framework"
],

"Accuracy":[
98.9,
86.65,
90,
94,
96.4,
99.96,
99.9,
93
],

"Precision":[
0.97,
0.85,
0.89,
0.92,
0.95,
0.999,
0.9998,
0.92
],

"Recall":[
0.96,
0.84,
0.88,
0.91,
0.94,
0.998,
0.9999,
0.91
],

"F1 Score":[
0.96,
0.84,
0.88,
0.91,
0.94,
0.998,
0.9998,
0.91
],

"Merits":[
"High accuracy using multimodal deep learning",
"Interpretable eye movement visualization",
"Large dataset including demographic features",
"Large student dataset",
"Effective handwriting detection",
"Highest accuracy model",
"Explainable handwriting detection",
"Combines multiple modalities"
],

"Demerits":[
"Requires expensive neuroimaging datasets",
"Small dataset",
"Model performance varies",
"Questionnaire bias",
"Requires labeled handwriting data",
"Requires EOG hardware",
"Synthetic dataset limitations",
"Complex system design"
]

}

df = pd.DataFrame(data)

# SECTION 1
st.header("1️⃣ Literature Review Data Table")
st.dataframe(df, use_container_width=True)

# SECTION 2
st.header("2️⃣ Accuracy Comparison")

accuracy_chart = df.set_index("Article")["Accuracy"]
st.bar_chart(accuracy_chart)

# SECTION 3
st.header("3️⃣ Precision vs Recall Comparison")

metrics_chart = df.set_index("Article")[["Precision","Recall","F1 Score"]]
st.line_chart(metrics_chart)

# SECTION 4
st.header("4️⃣ Best Performing Model")

best_model = df.loc[df["Accuracy"].idxmax()]

st.success(
f"Best Model: {best_model['Algorithm / Model']} | Accuracy: {best_model['Accuracy']}%"
)

# OPTIONAL FILTER
st.header("5️⃣ Filter by Algorithm")

algorithm_choice = st.selectbox(
"Select Algorithm",
df["Algorithm / Model"]
)

filtered_data = df[df["Algorithm / Model"] == algorithm_choice]

st.dataframe(filtered_data)