#import the model class and the tokenizer
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

# Download and setup the model and tokenizer
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot-90M")
#tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
#model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

def get_response(text):
        # Tokenize the utterance 
        inputs = tokenizer(text, return_tensors="pt")
        # Passing throuth the utterances to Blenderbot model
        res = model.generate(**inputs)
        #Decoding the model output
        respond = str(tokenizer.batch_decode(res, skip_special_tokens=True)[0])
        return respond