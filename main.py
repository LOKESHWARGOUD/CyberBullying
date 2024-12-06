import joblib
import streamlit as st

# Load the trained model and vectorizer
model = joblib.load('cyberbullying_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Title of the app
st.title('Cyberbullying Detection Model')

# Instructions
st.write("Enter a message to check if it's related to cyberbullying or not.")

# Text input for the user to enter a message
user_input = st.text_area("Type your message here:")


# Function to predict whether the message is cyberbullying or not
def predict(text):
    # Vectorize the input text
    vectorized_text = vectorizer.transform([text])

    # Predict using the model
    prediction = model.predict(vectorized_text)

    # Map prediction result to class label
    result = 'Cyberbullying' if prediction[0] == 0 else 'Non-Cyberbullying'
    return result


# When the user clicks the "Predict" button
if st.button('Predict'):
    if user_input:
        # Predict and display result
        result = predict(user_input)
        st.success(f'The message is: {result}')
    else:
        st.warning('Please enter some text to classify.')
