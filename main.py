import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import requests
from typing import List
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools  # type: ignore
from langchain.tools import BaseTool
from googletrans import Translator
from gtts import gTTS
import tempfile
import os
import pydeck as pdk
import pandas as pd

def main():
    
    st.title("Medi Scan")
    
    tab1 , tab2 , tab3 , tab4, tab5, tab6 = st.tabs(["**Home**", "**Identification**" , "**Experts Linking**", "**Store**", "**About Us**", "**Adulterations**"])
      
    with tab1 :
        
        st.subheader("Medi Scan is a web application that helps to identify medicinal herbal plants and their properties. It also helps to find experts and stores for medicinal herbal plants.")
        
        
    with tab2:
        st.write("**Identification**")
        st.info("Capture the image ....")
        st.subheader("Upload the image to discover medicinal herbal plants")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg" , key="herbal")
        lan = st.selectbox("Select the language",("Select","Hindi","Tamil","Telugu","Kannada","Malayalam","Marathi","Bengali","Gujarati","Punjabi","Urdu"))
        if st.button("Predict"):
            res = predict(uploaded_file)
            st.write("Herbal Medicinal Plant : "+res)
            properties = get_properties(res)
            st.write("Medicinal Properties : "+properties)
            if res :
                if lan != "Select":
                    lan_code = get_language_code(lan)
                    st.write("Translated Medicinal Properties : "+translate(properties,lan_code))
                    audio_input = "Herbal Medicinal plant is "+res+" and its medicinal properties are "+properties+" and its translated medicinal properties in "+lan_code+" are "+translate(properties,lan_code)
                    audio_result = get_audio(audio_input)
                    st.audio(audio_result, format="audio/mp3")
                    os.remove(audio_result)
        
    with tab3:
        st.write("**Experts Linking**")
        loc = st.text_input("Enter Location to find experts")
        if st.button("Search Expert"):
            res_experts = get_experts(loc)
            #get title , latitude and longitude from res_experts and store in datafra
            data = []
            for i in res_experts:
                if 'title' in i:
                    title = i['title']
                else:
                    title = "Not Available"
                if 'latitude' in i:
                    lat = i['latitude']
                else:
                    lat = "Not Available"
                if 'longitude' in i:
                    lng = i['longitude']
                else:
                    lng = "Not Available"
                data.append([title,lat,lng])
                
            df = pd.DataFrame(data, columns=["title", "latitude", "longitude"])
            
            avg_latitude = df['latitude'].mean()
            avg_longitude = df['longitude'].mean()

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_radius=100,
                get_color=[200, 30, 0],  # Red color
                pickable=True,
            )

            # Create a PyDeck map
            deck_map = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={"latitude": avg_latitude, "longitude": avg_longitude, "zoom": 11, "pitch": 50,},
                layers=[layer],
                tooltip={"text": "{title}\nLatitude: {latitude}\nLongitude: {longitude}"},
            )

            st.pydeck_chart(deck_map)
           
            
            st.markdown("---")
            for i in res_experts:
                if 'title' in i:
                    st.write("**Title** : " + i['title'])
                else:
                    st.write("**Title** : Not Available")
                if 'address' in i:
                    st.write("**Address** : " + i['address'])
                else:
                    st.write("**Address** : Not Available")
                if 'phoneNumber' in i:
                    st.write("**Phone Number** : " + i['phoneNumber'])
                else:
                    st.write("**Phone Number** : Not Available")
                if 'category' in i:
                    st.write("**Category** : " + i['category'])
                else:
                   st.write("**Category** : Not Available")
                if 'rating' in i:
                    st.write("**Rating** : " + str(int(i['rating'])))
                else:
                    st.write("**Rating** : Not Available")
                st.markdown("---")
            
            
        
    with tab4:
        st.write("**Store**")
        selected_input = st.selectbox("Select Input", ["Herbal plant", "Disease"])
        plant = st.text_input("Enter Plant / Disease Name")
        location = st.text_input("Enter Location")
        if selected_input == "Herbal plant" and st.button("Search"):
            res = get_herbal_plant(location, plant)
            #write the result of the dictionary
            st.markdown("---")
            for i in res:
                if 'title' in i:
                    st.write("**Title** : " + i['title'])
                else:
                    st.write("**Title** : Not Available")
                if 'address' in i:
                    st.write("**Address** : " + i['address'])
                else:
                    st.write("**Address** : Not Available")
                if 'phoneNumber' in i:
                    st.write("**Phone Number** : " + i['phoneNumber'])
                else:
                    st.write("**Phone Number** : Not Available")
                if 'category' in i:
                    st.write("**Category** : " + i['category'])
                else:
                   st.write("**Category** : Not Available")
                if 'rating' in i:
                    st.write("**Rating** : " + str(int(i['rating'])))
                else:
                    st.write("**Rating** : Not Available")
                st.markdown("---")
            
        elif selected_input == "Disease" and st.button("Search"):
            result = get_disease(location, selected_input)
            st.markdown("---")
            for i in result:
                if 'title' in i:
                    st.write("**Title** : " + i['title'])
                else:
                    st.write("**Title** : Not Available")
                if 'address' in i:
                    st.write("**Address** : " + i['address'])
                else:
                    st.write("**Address** : Not Available")
                if 'phoneNumber' in i:
                    st.write("**Phone Number** : " + i['phoneNumber'])
                else:
                    st.write("**Phone Number** : Not Available")
                if 'category' in i:
                    st.write("**Category** : " + i['category'])
                else:
                   st.write("**Category** : Not Available")
                if 'rating' in i:
                    st.write("**Rating** : " + str(int(i['rating'])))
                else:
                    st.write("**Rating** : Not Available")
                st.markdown("---")
        
    with tab5:
        st.subheader("About Us")
        st.write("Medi Scan is a web application that helps to identify medicinal herbal plants and their properties. It also helps to find experts and stores for medicinal herbal plants.")  
        
    with tab6:
        st.write("Adulterations")
        adulteration_input = st.file_uploader("Choose an image...", type="jpg" , key="adulteration")
        if st.button("Predict Adulteration"):
            result = adulteration_predict(adulteration_input)
            st.write("Herbal plant  : "+result)
            st.write("Medicinal properties : "+get_properties(result))
            
            
def get_audio(audio):
    # Create a gTTS object
    tts = gTTS(audio)

    # Create a temporary file to save the speech
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_filename = tmp_file.name
        tts.save(tmp_filename)

    return tmp_filename
       
def get_language_code(lan):
    if lan == "Hindi":
        return "hi"
    elif lan == "Tamil":
        return "ta"
    elif lan == "Telugu":
        return "te"
    elif lan == "Kannada":
        return "kn"
    elif lan == "Malayalam":
        return "ml"
    elif lan == "Marathi":
        return "mr"
    elif lan == "Bengali":
        return "bn"
    elif lan == "Gujarati":
        return "gu"
    elif lan == "Punjabi":
        return "pa"
    elif lan == "Urdu":
        return "ur"
    else:
        return "en"
    
def translate(input,lan_code):
    translator = Translator()
    result = translator.translate(input, dest=lan_code)
    return result.text

                
def adulteration_predict(uploaded_image):
    if uploaded_image is not None:
        # Load your trained model
        model = tf.keras.models.load_model(r'E:\HerboScan\models\herbo.h5')

        # Perform classification when the user uploads an image
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Perform the prediction
        predictions = model.predict(img_array)
        
        #json file
        #{'MixingBlackPepper': 0, 'Pappaya': 1, 'Pippernigrum': 2, 'Schinusmole': 3, 'Vitexagnuscastus': 4}

        # Define the class labels
        class_labels = ['MixingBlackPepper', 'Pappaya', 'Pippernigrum', 'Schinusmole', 'Vitexagnuscastus']

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class label
        return predicted_class

def predict(uploaded_image):
    
    if uploaded_image is not None:
        # Load your trained model
        model = tf.keras.models.load_model(r'E:\HerboScan\models\herbo-medicinal.h5')

        # Perform classification when the user uploads an image
        img = image.load_img(uploaded_image, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Perform the prediction
        predictions = model.predict(img_array)
        
        #json file
        #{'Asthma Plant': 0, 'Avaram': 1, 'Balloon vine': 2, 'Bellyache bush (Green)': 3, 'Benghal dayflower': 4, 'Big Caltrops': 5, 'Black-Honey Shrub': 6, 'Bristly Wild Grape': 7, 'Butterfly Pea': 8, 'Cape Gooseberry': 9, 'Common Wireweed': 10, 'Country Mallow': 11, 'Crown flower': 12, 'Green Chireta': 13, 'Holy Basil': 14, 'Indian CopperLeaf': 15, 'Indian Jujube': 16, 'Indian Sarsaparilla': 17, 'Indian Stinging Nettle': 18, 'Indian Thornapple': 19, 'Indian wormwood': 20, 'Ivy Gourd': 21, 'Kokilaksha': 22, 'Land Caltrops (Bindii)': 23, 'Madagascar Periwinkle': 24, 'Madras Pea Pumpkin': 25, 'Malabar Catmint': 26, 'Mexican Mint': 27, 'Mexican Prickly Poppy': 28, 'Mountain Knotgrass': 29, 'Nalta Jute': 30, 'Night blooming Cereus': 31, 'Panicled Foldwing': 32, 'Prickly Chaff Flower': 33, 'Punarnava': 34, 'Purple Fruited Pea Eggplant': 35, 'Purple Tephrosia': 36, 'Rosary Pea': 37, 'Shaggy button weed': 38, 'Small Water Clover': 39, 'Spiderwisp': 40, 'Square Stalked Vine': 41, 'Stinking Passionflower': 42, 'Sweet Basil': 43, 'Sweet flag': 44, 'Tinnevelly Senna': 45, 'Trellis Vine': 46, 'Velvet bean': 47, 'coatbuttons': 48, 'heart-leaved moonseed': 49}

        # Define the class labels
        class_labels = ['Asthma Plant', 'Avaram', 'Balloon vine', 'Bellyache bush (Green)', 'Benghal dayflower', 'Big Caltrops', 'Black-Honey Shrub', 'Bristly Wild Grape', 'Butterfly Pea', 'Cape Gooseberry', 'Common Wireweed', 'Country Mallow', 'Crown flower', 'Green Chireta', 'Holy Basil', 'Indian CopperLeaf', 'Indian Jujube', 'Indian Sarsaparilla', 'Indian Stinging Nettle', 'Indian Thornapple', 'Indian wormwood', 'Ivy Gourd', 'Kokilaksha', 'Land Caltrops (Bindii)', 'Madagascar Periwinkle', 'Madras Pea Pumpkin', 'Malabar Catmint', 'Mexican Mint', 'Mexican Prickly Poppy', 'Mountain Knotgrass', 'Nalta Jute', 'Night blooming Cereus', 'Panicled Foldwing', 'Prickly Chaff Flower', 'Punarnava', 'Purple Fruited Pea Eggplant', 'Purple Tephrosia', 'Rosary Pea', 'Shaggy button weed', 'Small Water Clover', 'Spiderwisp', 'Square Stalked Vine', 'Stinking Passionflower', 'Sweet Basil', 'Sweet flag', 'Tinnevelly Senna', 'Trellis Vine', 'Velvet bean', 'coatbuttons', 'heart-leaved moonseed']

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class label
        return predicted_class


def get_properties(plant):
    
    llm = OpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        max_tokens=200,
        temperature=0,
        client=None,
        model="text-davinci-003",
        frequency_penalty=1,
        presence_penalty=0,
        top_p=1,
    )

    # Load the tools
    tools: List[BaseTool] = load_tools(["google-serper"], llm=llm)

    # Create a new instance of the AgentExecutor class
    agent: AgentExecutor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Create the template
    template = """I found this medicinal plant {topic}. Give me the medicinal properties of this plant"""

    # Generate the response
    response: str = agent.run(template.format(topic=plant))

    return response

    
def get_experts(location):
    
    query = "herbalist in "+location
    
    url = "https://google.serper.dev/places"

    payload = json.dumps({
    "q": query,
    })
    headers = {
    'X-API-KEY': st.secrets["GOOGLE_API_KEY"],
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    res = json.loads(response.text)
    
    print(res['places'])
    
    return res['places']
    
    
def get_disease(location, input_type):
    
    query = "i am having "+input_type+" disease, give me healthcare centers in "+location+" to cure it"
    
    url = "https://google.serper.dev/places"

    payload = json.dumps({
    "q": query,
    })
    headers = {
    'X-API-KEY': st.secrets["GOOGLE_API_KEY"],
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    res = json.loads(response.text)
    
    print(res['places'])
    
    return res['places']
    
    
    
def get_herbal_plant(location, input_type):
    
    query = "i need to find store for this medicinal herbal plant "+input_type+" in "+location
    
    url = "https://google.serper.dev/places"

    payload = json.dumps({
    "q": query,
    })
    headers = {
    'X-API-KEY':   st.secrets["GOOGLE_API_KEY"],
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    res = json.loads(response.text)
    
    print(res['places'])
    
    return res['places']
    
if __name__ == '__main__':
    main()