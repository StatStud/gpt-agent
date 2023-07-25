import requests
from flask import Flask, request, render_template, jsonify, Response
from chat import OpenAIAgent
from datetime import datetime
import pandas as pd
#from twilio.twiml.messaging_response import MessagingResponse



# temp fix, move this to some env variable or other config option
tables_dir = "file_tables"
file_tables = dict()
file_tables["follow_up"] = f"{tables_dir}/follow_up.csv"
file_tables["users"] = f"{tables_dir}/users.csv"
file_tables["visits"] = f"{tables_dir}/visits.csv"
file_tables["whatsapp"] = f"{tables_dir}/whatsapp.csv"
file_tables["messages"] = f"{tables_dir}/messages.jsonl"
file_tables["intents"] = f"{tables_dir}/intents.csv"

class UserDB:
    def __init__(self):
        # load tables
        self.users = pd.read_csv(file_tables["users"])
        self.whatsapp = pd.read_csv(file_tables["whatsapp"])

    def get_whatsapp_property(self, phone_number, key):
        user = self.whatsapp[self.whatsapp.phone_number == phone_number]
        property = user[key].values[0]
        return property

    # TODO add

user_db = UserDB()


base_url = 'http://localhost:5000/chat'
app = Flask(__name__)
agent = OpenAIAgent()

def lookup_user_id_from_whatsapp(profile_name, phone_number, account_sid):
    user_id = user_db.get_whatsapp_property(phone_number, 'user_id')
    return user_id

def get_whatsapp_user_info(request):
    account_sid = request.values.get("AccountSid")
    profile_name = request.values.get("ProfileName")
    phone_number = request.values.get("From")
    print(f"whatsapp info - profile_name: {profile_name}, phone_number: {phone_number}, account_sid: {account_sid}")
    return profile_name, phone_number, account_sid


def get_user_info_from_request(request):
    # Figure out what type of message we are dealing with
    # AccountSid is the uniq id we use for linking WhatsApp accounts
    if request.values.get("AccountSid"):
        msg_type = "whatsapp"
    else:
        msg_type = "curl"
    print(f"Message Type: {msg_type}")

    # handle postman or curl message
    if msg_type == "curl":
        msg_time = datetime.now()
        user_id = request.json['user_id']
        user_input = request.json['message']
    elif msg_type == "whatsapp":
        # TODO, see if message time is in whatsapp fields
        msg_time = datetime.now()
        profile_name, phone_number, account_sid = get_whatsapp_user_info(request)
        user_id = lookup_user_id_from_whatsapp(profile_name, phone_number, account_sid)
        user_input = request.values.get("Body", "").lower()

    return user_id, user_input, msg_time, msg_type

def format_response(text_response, msg_type):
    if msg_type == 'curl':
        response = jsonify({'response': text_response})
    elif msg_type == 'whatsapp':
        msg_response = MessagingResponse()
        msg_response.message(text_response)
        response = Response(str(msg_response), mimetype="application/xml")
    return response

@app.route('/')
def index():
    return 'Welcome to the chat app!'


@app.route('/chat', methods=['POST'])
def chat():

    # handle whatsapp (twilio) messaging channel message
    user_id, user_input, msg_time, msg_type = get_user_info_from_request(request)

    text_response = agent.chat(user_id, user_input, msg_time)
    response = format_response(text_response, msg_type)

    return response



if __name__ == '__main__':
    app.run(debug=False, port=4001)
#    app.run(debug=True, port=4001)
