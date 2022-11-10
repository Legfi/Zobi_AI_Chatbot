import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import warnings
warnings.filterwarnings("ignore")

#import the model class and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
playsound.playsound("Zobi.mp3")
while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        utterance = ""

        try:
            utterance = r.recognize_google(audio)
            print(utterance)
        except Exception as e:
            print("Exception: " + str(e)[3:])
    if "goodbye" in utterance:
        break
    else:
        # Tokenize the utterance 
        inputs = tokenizer(utterance, return_tensors="pt")

        # Passing throuth the utterances to Blenderbot model
        res = model.generate(**inputs)

        #Decoding the model output
        respond = tokenizer.decode(res[0])[3:-4]
        print(respond)
        #sending respond to gTTS
        tts = gTTS(respond, lang='en') 
        filename = 'voice1.mp3'
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)