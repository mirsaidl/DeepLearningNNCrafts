from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Title for the Streamlit app
st.title("Transportation Classification Model by Mirsaid")

# File uploader for image
file = st.file_uploader('Upload Picture', type=['png', 'jpeg', 'gif', 'svg'])

if file is not None:
    # Display the uploaded image using PIL
    st.image(file)

    # Convert the uploaded file to a PILImage
    img = PILImage.create(file)

    # Load the trained model
    model = load_learner('transport_model.pkl')

    # Get prediction and probabilities
    pred, pred_id, probs = model.predict(img)

    # Display prediction and probability
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    # Create a bar chart using Plotly Express
     
    fig = px.bar(y=probs*100, x=model.dls.vocab)
    fig.update_layout(
    yaxis_title="Probability(%)",  # Label for the y-axis
    xaxis_title="Categories"        # Label for the x-axis
    )
    st.plotly_chart(fig)

else:
    st.write("No image uploaded. Please upload an image!")
x