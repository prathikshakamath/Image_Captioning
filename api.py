import os
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, request, jsonify

from model import EncoderCNN, DecoderRNN
from utils import clean_sentence
from data_loader import get_loader

# Initialize the Flask app
app = Flask(__name__)

# Define the transform to preprocess the testing images
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the data loader
data_loader = get_loader(transform=transform_test, mode='test')

def load_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the most recent checkpoint
    checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'))

    # Specify values for embed_size and hidden_size
    embed_size = 256
    hidden_size = 512

    # Get the vocabulary and its size
    vocab = data_loader.dataset.vocab
    vocab_size = len(vocab)

    # Initialize the encoder and decoder, and set each to inference mode
    encoder = EncoderCNN(embed_size).to(device)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    decoder.eval()

    # Load the pre-trained weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # Move the models to the same device
    encoder.to(device)
    decoder.to(device)

    return encoder, decoder, vocab

def get_caption(image):
    # Load the image and apply the transformation
    image = Image.open(image).convert("RGB")
    image = transform_test(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    # Obtain the embedded image features
    features = encoder(image.to(device))

    # Reshape the features tensor to have the expected shape
    features = features.unsqueeze(1)

    # Pass the embedded image features through the model to get a predicted caption
    output = decoder.sample(features)

    # Clean up the predicted caption
    sentence = clean_sentence(output, vocab)

    # Return the caption
    return sentence

@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    image = Image.open(BytesIO(image_data))

    # Convert the image to base64 encoded string
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Get the caption for the image
    caption = get_caption(image)

    return jsonify({'caption': caption, 'image': image_base64})

if __name__ == '__main__':
    # Load the image captioning model
    encoder, decoder, vocab = load_model()

    # Run the Flask app
    app.run(host='192.168.1.8',port=5000)
