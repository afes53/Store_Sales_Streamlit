import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time
import os
import model_func
import plotly.express as px
import plotly
from streamlit_option_menu import option_menu
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from css import create_person_card
from css import create_card
from css import main
from css import send_email
from css import show_animated_card 
from css import pagetitle 
from css import bar 
from css import custom_title 
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder

#---------------------------------------------------------------------------------------------------

MODELS_FOLDER = "./model/"

def load_model(store_nbr, family, models_folder=MODELS_FOLDER):
    model_path = os.path.join(models_folder, f'store_{store_nbr}{family}_model.pkl')
    return joblib.load(model_path)

OH = joblib.load('onehot.pkl')


#---------------------------------------------------------------------------------------------------

#font
def html_content(text):
    font_link = '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">'
    return f"""{font_link}<h1><strong style='color: red; font-weight: bold; font-family: "Lucida Handwriting", script ;'>{text}</strong></h1>"""

#---------------------------------------------------------------------------------------------------

#Session_State
def session_state_():
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True

    if st.session_state.first_visit:
        placeholder = st.empty()
        
        person = {      
        "name": "Welcome to our project page. Our project is based on a Time Series problem involving store sales forecasts.",
        "text": html_content("NOBODY TEAM")
    }
        show_animated_card(placeholder, **person)
        time.sleep(2)
        placeholder.empty()
        st.session_state.first_visit = False
if __name__ == "__main__":
    session_state_()

#---------------------------------------------------------------------------------------------------
    
@st.cache_data   
def get_data():
    df = pd.read_csv('mergedata.csv')
    return df
df = get_data()

#-------------------------------------------------------------------------------------------------------------------
# Create menu
with st.sidebar:
    selected = option_menu("Menu", ["About Data", "Home", "Visualization", "Contact Us", "Project Developers"],
                           icons=['filter-square-fill','house-heart-fill','bar-chart-line-fill','bell-fill','people-fill'],
                           menu_icon="cast",default_index=1) 
  
# Main content
if selected == "Home":
    custom_title("Forecasting Part")
    bar()
    

    def create_date_picker(df, column, label, col_features1):
        min_date = pd.Timestamp(year=2017, month=8, day=16)
        max_date = pd.to_datetime(df[column].max())

        target_max_date = pd.Timestamp(year=2023, month=12, day=31)
        if max_date < target_max_date:
            max_date = target_max_date
            
        selected_date = col_features1.date_input(label, min_value=min_date, max_value=max_date, value=min_date)
        return selected_date


    inputs = ["date", "store_nbr", "family","sales", "onpromotion", "nat_terremoto","is_holiday","oil_interpolated","transactions"]

    # model
    with st.expander("Make a prediction", expanded=True):
        onpromotion = st.number_input(label="Please enter the total number of expected items to be on promotion",min_value=0,max_value=252)
        transactions = st.number_input(label="Please enter the total number of expected transactions",min_value=1,max_value=8359) 
        oil_interpolated = st.number_input(label="Number of Expected Oil Price",min_value=26.19,max_value=110.62)
        
        col_features1, col_features2, col_features3 = st.columns(3)

        date=create_date_picker(df, 'date', 'Select a Date:', col_features1)
        store_nbr = col_features2.selectbox('Store Number', np.sort(df['store_nbr'].unique()))
        family = col_features3.selectbox('Family Type', np.sort(df['family'].unique()))
        is_holiday = col_features1.radio('İs it a holiday (1 holiday, 0 not a holiday', np.sort(df['is_holiday'].unique())) 
        nat_terremoto = col_features2.radio('Earthquake (earthquake in process or not)', np.sort(df['nat_terremoto'].unique()))


        predicted = st.button("Predict")
        show_prediction_message = False
        if predicted:
            show_prediction_message = True  
            input_dict = {
                "date": [date],
                "store_nbr": [store_nbr],
                "family": [family],
                "onpromotion": [onpromotion],
                "nat_terremoto":[nat_terremoto],
                "is_holiday":[is_holiday],
                "oil_interpolated":[oil_interpolated],
                "transactions":[transactions],
                "sales": 0
            }

            input_df = pd.DataFrame.from_dict(input_dict)
            input_df["date"] = pd.to_datetime(input_df["date"]) 
            input_df = model_func.create_date_features(input_df)
            input_df = model_func.new_features(input_df)
        
            input_df= model_func.oil_lag(input_df)
    
            input_df = model_func.lag_features(input_df, 15)
            input_df = model_func.roll_mean_features(input_df, 15)

            alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
            input_df = model_func.ewm_features(input_df, alphas, 15)
            
            
            cat_cols = ['Days_name', 'season', 'promotion_level', 'promotiondays']

            onehot_encoded_features = OH.transform(input_df[cat_cols])

            onehot_encoded_array = onehot_encoded_features.toarray()

            new_column_names = OH.get_feature_names_out(cat_cols)

            encoded_df = pd.DataFrame(onehot_encoded_array, columns=new_column_names, index=input_df.index)
            input_df = pd.concat([input_df.drop(cat_cols, axis=1), encoded_df], axis=1)
              

            input_df= model_func.target_log_transmission(input_df) 
            input_df= model_func.selected_df(input_df,store_nbr, family)
            
            
            model = load_model(store_nbr, family)
            input_df.drop(columns=["date", "store_nbr", "family", "year","sales"], inplace=True) 

            try:
                input_df= input_df[model.feature_name_]
            except: 
                try:
                    input_df = input_df[model.feature_name()]
                except:
                    try:
                        input_df= input_df[model.feature_names_in_]                      
                    except:
                        input_df = input_df[model.get_booster().feature_names]
    
            model_output = model.predict(input_df)
            Y_pred_original = np.expm1(model_output)
            
            input_dict["Total Sales($)"] = Y_pred_original

            if int(Y_pred_original[0]) < 0:
                Y_pred_original[0] = 0
            formatted_output = f"<b>{Y_pred_original[0]}</b>"
            st.write(f"Your total predicted sales will be :  {formatted_output}", unsafe_allow_html=True)

#-------------------------------------------------------------------------------------------------------------------

#About Data
elif selected == "About Data":  
    def main():
        create_card("<strong style='color: red; font-weight: bold;'>Purpose Text</strong>", "In this data, we will estimate the sales of thousands of product families in Favorita stores in Ecuador. The training data includes sales dates, store and product information, whether the products are promoted or not, and sales figures. <br><ul><b style='color:green;'> Additional Notes: </b></ul><li>In the public sector, salaries are paid biweekly on the 15th and last day of each month. Supermarket sales may be affected by these payments.</li><li>On April 16, 2016, a 7.8 magnitude earthquake occurred in Ecuador. People joined the relief effort by donating water and other essential goods, which greatly affected supermarket sales for several weeks after the earthquake.</li><br>Using this data, we will develop a machine learning model, taking into account how your model may affect sales forecasts. The purpose of this competition is to help better plan business strategies by making sales forecasts on a product family basis for Favorita stores in Ecuador.")
        
        detailed_content = """
        <div>
            <strong>Store Number:</strong> 
            <span style='font-weight: normal;'>The number of the store.</span>
            <br>
            <strong>Product Family:</strong> 
            <span style='font-weight: normal;'>Product Family such as 'AUTOMOTIVE', 'BEAUTY', etc. Details:</span>
            <br>
            <ul>
                <li><strong>AUTOMOTIVE:</strong> Products related to the automotive industry.</li>
                <li><strong>BEAUTY:</strong> Beauty and personal care products.</li>
                <li><strong>CELEBRATION:</strong> Products for celebrations and special occasions.</li>
                <li><strong>CLEANING:</strong> Cleaning and household maintenance products.</li>
                <li><strong>CLOTHING:</strong> Clothing and apparel items.</li>
                <li><strong>FOODS:</strong> Food items and groceries.</li>
                <li><strong>GROCERY:</strong> Grocery products.</li>
                <li><strong>HARDWARE:</strong> Hardware and tools.</li>
                <li><strong>HOME:</strong> Home improvement and decor products.</li>
                <li><strong>LADIESWEAR:</strong> Women's clothing.</li>
                <li><strong>LAWN AND GARDEN:</strong> Lawn and garden products.</li>
                <li><strong>LIQUOR, WINE, BEER:</strong> Alcoholic beverages.</li>
                <li><strong>PET SUPPLIES:</strong> Products for pets and animals.</li>
                <li><strong>STATIONERY:</strong> Stationery and office supplies.</li>
            </ul>
            <strong>Number of Items on Promotion:</strong> 
            <span style='font-weight: normal;'>Number of items on promotion within a particular shop.</span>
            <br>
            <strong>City:</strong> 
            <span style='font-weight: normal;'>City where the store is located.</span>
            <br>
            <strong>Cluster:</strong> 
            <span style='font-weight: normal;'>Cluster number which is a grouping of similar stores.</span>
            <br>
             <strong>Transactions:</strong> 
            <span style='font-weight: normal;'>: Number of transactions.</span>
            <br>
             <strong>Crude Oil Price:</strong> 
            <span style='font-weight: normal;'>Daily Crude Oil Price..</span>
            <br>
        </div>
        """
        create_card("Product Categories", detailed_content)

    if __name__ == "__main__":
        main()


#-------------------------------------------------------------------------------------------------------------------

#Result Graphs and Powee BI
elif selected == "Visualization":  
    tabs = st.tabs(["Power BI", "Data Exploration"])
    with tabs[0]:
        custom_title("Power BI")
        embed_url = "https://app.powerbi.com/reportEmbed?reportId=76304c62-a234-4552-a450-bb4a7be5f096&autoAuth=true&ctid=4ebbea79-0cb8-4817-9d44-b66492d06260"
        st.components.v1.iframe(embed_url, width=750, height=500)

    with tabs[1]:
        def load_data(path):
            dataset = pd.read_csv(path)
            return dataset

        data_path = "mergedata.csv"
        load_df = load_data(data_path)

        data = st.container()
        with data:
            custom_title("You can preview the dataset",color="black",font_family="Times New Roman",font_size="28px")
            if st.button("Preview the dataset"):
                data.write(load_df)
            custom_title("A Chart of the Daily Sales Across Favorita Stores",color="black",font_family="Times New Roman",font_size="28px")
            if st.button("View Chart"):
                load_df = load_df.set_index('date')
                st.line_chart(load_df["sales"])
#-------------------------------------------------------------------------------------------------------------------

#contact
elif selected == "Contact Us":
    custom_title("Contact Us")
    if __name__ == "__main__":
        main()
#-------------------------------------------------------------------------------------------------------------------

#Project Developer
elif selected == "Project Developers":
    st.balloons()
    def main():
        people = [
            ("https://media.licdn.com/dms/image/D4D03AQGl_Ru5Nr2BIw/profile-displayphoto-shrink_400_400/0/1713279989810?e=1718841600&v=beta&t=NR2sr2ZTJWLZhUmWuV3FkORf2qPS5tNaZ_FF6KYLWpw", "Ömer Aydoğdu", "https://www.linkedin.com/in/omeraydogdu/"),
            
            ("https://media.licdn.com/dms/image/D4D03AQF144lyIqQDPQ/profile-displayphoto-shrink_400_400/0/1673693487767?e=1718841600&v=beta&t=nh8KoiI0oVWikH2UbVbrzVhifj5GbP6No6e3hZo_CiE", "Metin Kuriş", "https://www.linkedin.com/in/metin-kuris/"),
            
            ("https://media.licdn.com/dms/image/D4D03AQEx3oxABIzuFw/profile-displayphoto-shrink_400_400/0/1699646324460?e=1718841600&v=beta&t=9u8Sf_Y3GW07Uelq2YO0N5EQT-owSFNtDqasTwUvIfc", "Sevda Nur Çopur", "https://www.linkedin.com/in/sevdanurcopur/"),
            
            ("https://media.licdn.com/dms/image/D4D35AQFFivOBr1aqiw/profile-framedphoto-shrink_400_400/0/1707406247483?e=1714482000&v=beta&t=efw_axveoFNA2unGF2X-pt-hagr6Li8sgZhk1x8lTNE", "Sefa Berk Acar", "https://www.linkedin.com/in/sefaberkacar/")

        ]

        col1, col2= st.columns(2)

        with col1:
            create_person_card(*people[0])
            create_person_card(*people[2])
        with col2:
            create_person_card(*people[1])
            create_person_card(*people[3])

    if __name__ == "__main__":
        main()
