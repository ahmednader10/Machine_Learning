import os
import requests
import sys
import json
import random
from textblob import TextBlob
from dateutil.parser import parse
import calendar
from flask import Flask, request

app = Flask(__name__)

ACCESS_TOKEN = "EAAMGUoyI9DkBAHnsCHleRlZAKZBajpvkisREXm12jwd54wuxbslSZAaT0WSs2W40iarqtisKIn792Wo0O8QR1nG6yHwiUROHXBizsZBS1FLaBrxmT6qXjHQnsgZCDkYzJcBu5Gj7qwUcePYchK4u2RhoBzeSip6bOgHG7AUIbzAZDZD"
memory = {}
greetings = ['hola', 'hello', 'hi','hey','sup']
output_greetings = ['Hola', 'Hello', 'Hi','Hey','Sup']
seasons_months = {1:'winter', 2:'winter',3:'winter', 4:'spring',5:'spring', 6:'spring', 7:'summer', 8:'summer', 9:'summer',10:'autumn',11:'autumn',12:'autumn'}

seasons = ['summer', 'spring', 'winter', 'autumn']
durations = ['days', 'nights', 'weeks', 'months']
recommendations = {'summer':'Alexandria', 'winter':'Aswan','autumn':'Barcelona','spring':'Berlin'}

def check_for_start(sentence, sender_id):
    fields = ['first_name','last_name','gender']
    if "get started" in sentence.lower():
        #memory={}
        url = 'https://graph.facebook.com/v2.6/'+sender_id+'?fields=first_name,last_name,gender&access_token='+ACCESS_TOKEN
        resp = requests.get(url, data={'fields': ','.join(fields)})
        resp_data = resp.json()
        if resp_data:
            memory["name"] = resp_data['first_name']
        return random.choice(output_greetings) + " "+memory["name"]+" my name is Alex, and I'm your personal travel assistant, I can recommend places for you. would you please tell me about yourself (age, nationality, ..etc)?"
    elif check_for_greeting(sentence):
        #memory={}
        url = 'https://graph.facebook.com/v2.6/'+sender_id+'?fields=first_name,last_name,gender&access_token='+ACCESS_TOKEN
        resp = requests.get(url, data={'fields': ','.join(fields)})
        resp_data = resp.json()
        if resp_data:
            memory["name"] = resp_data['first_name']
        return random.choice(output_greetings) + " "+memory["name"]+", my name is Alex, and I'm your personal travel assistant, I can recommend places for you. would you please tell me about yourself (age, nationality, ..etc)?"

def check_for_greeting(sentence):
    """If any of the words in the user's input was a greeting, return a greeting response"""
    for word in sentence.words:
        if word.lower() in greetings:
            return True

def check_for_intro(sentence):
    response = None
    #name = find_name(sentence)
    #if name:
    #    response = random.choice(greetings) + " "+name+", Just to make sure, "
    age = find_age(sentence)
    if age:
        response ="Ok "+memory["name"]+", Just to make sure, Your age is: "+ age
    nationality = find_nationality(sentence)
    if nationality:
        if response:
            response +=", You come from: " + nationality
        else:
            response ="Ok "+memory["name"]+", Just to make sure, You come from: " + nationality
    occupation = find_occupation(sentence)
    if occupation:
        if response:
            response +=", You work as: " + occupation
        else:
            response ="Ok "+memory["name"]+", Just to make sure, You work as: " + occupation
    if response:
        response += ", Is that correct?"
    return response

def find_occupation(sentence):
    occupation = None
    if "work as" in sentence:
        index = sentence.words.index("as") + 1
        occupation = sentence.words[index]
        memory["occupation"] = occupation
    return occupation

def find_nationality(sentence):
    nationality = None
    for w, p in sentence.pos_tags:
        if p == 'NNP':  # This is a nationality
            index = sentence.words.index(w)
            prep = sentence.words[index-1]
            if prep.lower() in 'from':
                nationality = w
                memory["nationality"] = nationality
    return nationality

def find_age(sentence):
    age = None
    for w, p in sentence.pos_tags:
        if p == 'CD':  # This is a number
            index = sentence.words.index(w)
            if index+1 < len(sentence.words):
                next_word = sentence.words[index+1]
                if next_word.lower() in "years":
                    age = w
                    memory["age"] = age
    return age

def find_name(sentence):
    name = None
    if not name:
        if "name" in sentence:
            index = sentence.words.index("name") + 2
            name = sentence.words[index]
            memory["name"] = name
    return name

def check_for_info_confirmation(sentence):
    response = None
    if "yes" in sentence.lower().split():
        response = "Great! So when do you want to travel "+memory["name"]+"(Please provide the Season or Dates)?"
    elif "no" in sentence.lower().split():
        response = "My bad, Could you please provide me your information again?"
        # if not 'age' in memory:
        #     response = "Could you please tell me your age"
        #     memory["missing_info"] = True
        # if not 'nationality' in memory:
        #     if not 'age' in memory:
        #         response += ", nationality"
        #         memory["missing_info"] = True
        #     else:
        #         response += "Could you please tell me your nationality"
        #         memory["missing_info"] = True
        # if not 'occupation' in memory:
        #     if not 'age' in memory or not 'nationality' in memory:
        #         response += ", occupation"
        #         memory["missing_info"] = True
        #     else:
        #         response += "Could you please tell me your occupation"
        #         memory["missing_info"] = True
    return response

def check_for_time(sentence):
    season = None
    month = None
    sentence = sentence.lower()
    for word in sentence.words:
        if word in seasons:
            index = seasons.index(word)
            season = seasons[index]
            memory['season'] = season
    if season:
        return "And how long do you want to stay there during "+season+"?"
    else:
        try:
            month = parse(str(sentence)).month
            month_name = calendar.month_name[month]
            memory['month'] = month
            return "And how long do you want to stay there during "+str(month_name)+"?"
        except ValueError:
            return None

def check_for_duration(sentence):
    response = None
    duration = None
    sentence = sentence.lower()
    for word in sentence.words:
        if word in durations:
            index = sentence.words.index(word)
            duration = sentence.words[index - 1]+" "+sentence.words[index]
            memory['duration'] = duration
    if duration:
        response = recommend_destination()
    return response

def recommend_destination():
    if 'season' in memory:
        key = memory['season']
        return "Thanks "+memory['name']+", so for a "+memory['duration']+" vacation during " + memory['season']+", I'd recommend for you going to "+ recommendations[key]+", How do you feel about that?"
    if 'month' in memory:
        month = memory['month']
        month_name = calendar.month_name[month]
        season = seasons_months[month]
        return "Thanks "+memory['name']+", so for a "+memory['duration']+" vacation during " + str(month_name)+", I'd recommend for you going to "+ recommendations[season]+", How do you feel about that?"

def check_for_sentiment(sentence):
    response = None
    calc_sentiment = sentence.sentiment.polarity
    if calc_sentiment > 0.0:
        response = "I'm so glad you're happy with that!"
    else:
        response = "I'm sorry, I wasn't helpful enough, You can kindly reach one of our sales agents"
    return response

@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == "secret":
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():

    # endpoint for processing incoming messaging events

    data = request.get_json()

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message
                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text

                    parsed_text = TextBlob(message_text)
                    response = check_for_start(parsed_text, sender_id)
                    if not response:
                        response = check_for_intro(parsed_text)
                    if not response:
                        response = check_for_info_confirmation(parsed_text)
                    if not response:
                        response = check_for_time(parsed_text)
                    if not response:
                        response = check_for_duration(parsed_text)
                    if not response:
                        response = check_for_sentiment(parsed_text)

                    print(response)
                    send_message(sender_id, response)

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID

                    parsed_text = TextBlob("Get Started")
                    response = check_for_start(parsed_text, sender_id)
                    send_message(sender_id, response)

    return "ok", 200


def send_message(recipient_id, message_text):

    #log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    #if r.status_code != 200:
    #    log(r.status_code)
    #    log(r.text)


if __name__ == '__main__':
    app.run(debug=True)
