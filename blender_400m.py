#import the model class and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
def get_response(msg):
    # Tokenize the utterance 
        inputs = tokenizer(msg, return_tensors="pt")

        # Passing throuth the utterances to Blenderbot model
        res = model.generate(**inputs)

        #Decoding the model output
        respond = str(tokenizer.decode(res[0])[3:-4])
        return(respond)