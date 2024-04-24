import streamlit as st
import time

#pagetitle
def pagetitle(name):
    html_title = f"<h1 style='color: red; text-align:center;'>{name}</h1>"
    st.markdown(html_title, unsafe_allow_html=True)
    
#---------------------------------------------------------------------------------------------------
 #bar   
def bar():
    progress_text = "The process is in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

#---------------------------------------------------------------------------------------------------
#PROJECT DEVELOPERS
def create_person_card(image_url, name, profile_url):
    card_style = """
    <style>
        .card {
            margin: 10px;
            padding: 10px;
            background-color: transparent;
            border: 1px solid silver;
            border-radius: 40px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .card img {
            width: 80%;
            border-radius: 50%;
        }
        .card-title {
            font-size: 20px;
            margin-top: 10px;
            color: #000000;
        }
        .card-link {
            font-size: 16px;
            color: #007BFF;
            display: inline-block;
            margin-top: 5px;
        }
        .card-link:hover {
            color: #0056b3;
        }
        .fa-linkedin {
            color: #0077b5;
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    """

    card_html = f"""
    <div class="card">
        <img src="{image_url}" alt="Profile Image">
        <div class="card-title">{name}</div>
        <a href="{profile_url}" target="_blank" class="card-link"><i class="fab fa-linkedin fa-lg"></i></a>
    </div>
    """

    st.markdown(card_style + card_html, unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------
#CONTACT US
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(name, email, message):
    sender_email = "example@gmail.com" 
    app_password = "google app password" 

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = "New Message"

    body = f"Name: {name}\nMessage: {message}"
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()  

    try:
        server.login(sender_email, app_password)  
        server.sendmail(sender_email, email, msg.as_string())
        print("Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("Authentication failed. Check your App Password or security settings.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        server.quit()
    return True

def main():
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if send_email(name, email, message):
                st.success("Thank you, your message has been sent via email.")
            else:
                st.error("Failed to send email. Please check the details and try again.")

#---------------------------------------------------------------------------------------------------
#ABOUT DATA
def create_card(title, content):
    card_html = f"""
    <div style="
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        padding: 16px;
        margin: 10px;
        background-color: white;">
        <h4 style='margin-bottom: 10px; color: red; font-weight: bold;'>{title}</h4>
        <p>{content}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
            
#---------------------------------------------------------------------------------------------------
#SESSION STATE
def show_animated_card(placeholder, name, text):
    card_html = f"""
    <style>
        .animated-card {{
            margin: 10px;
            padding: 20px;
            background-color: white;
            border: 2px solid #ccc;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        .animated-card img {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
        }}
        .animated-card-title {{
            font-size: 22px;
            margin-top: 10px;
            color: #333;
        }}
        .animated-card-link {{
            font-size: 18px;
            color: #007BFF;
            display: inline-block;
            margin-top: 5px;
        }}
        .animated-card-link:hover {{
            color: #0056b3;
        }}
    </style>
    <div class="animated-card">
        <div class="animated-card-title">{name}</div>
        <div class="animated-card-title">{text}</div>
    </div>
    """
    placeholder.markdown(card_html, unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------
#font and color for the title
def custom_title(text, font_family="Lucida Handwriting", color="red", font_size="32px"):
    """
    Creates a custom styled title in a Streamlit app.
    
    Parameters:
    text (str): The text to display as a title.
    font_family (str): The font family to use for the title.
    color (str): The color of the text.
    font_size (str): The size of the font.
    """
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family={font_family.replace(" ", "+")}&display=swap');
    </style>
    <h1 style='font-family: "{font_family}", script; color: {color}; font-size: {font_size};'>
        {text}
    </h1>
    """
    st.markdown(html, unsafe_allow_html=True)
    