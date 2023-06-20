import streamlit as st
from disease_prediction import predict_disease

# Set page title and favicon
st.set_page_config(page_title="Symp2Diagnose", page_icon="🩺")

# Streamlit app
def main():
    st.title("Disease Classification")
    input_text = st.text_area("Describe the disease symptom:", height=100)
    if st.button("Predict"):
        predicted_label, confidence_score = predict_disease(input_text)
        st.write("The Predicted Disease:", predicted_label)
        st.write("Predicted probability:", confidence_score)

if __name__ == "__main__":
    main()