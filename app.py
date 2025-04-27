from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F

from model import SketchCNN #import the CNN model
from model import SketchRNN #import the RNN model

from flask_cors import CORS

#initialize Flask app
app = Flask(__name__)
CORS(app) #enable CORS to allow requests from different origins

#list of class names for predictions
class_names = [
    'cat', 'tree', 'fish', 'clock', 'castle', 'crown', 'lollipop', 'moon',
    'watermelon', 'tornado', 'apple', 'bowtie', 'bicycle', 'diamond',
    'flower', 'butterfly', 'eye', 'lightning', 'cloud', 'pizza'
]

#set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize the model variable (will be set dynamically based on the the user's choice)
model = None

#define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    #parse the incoming JSON request
    data = request.get_json()
    print("Received JSON:", data)

    #determine which model to use based on the "model" field in the request
    chosen_model = data.get("model", "model1")  #default to "model1" if not specified

    #if model1 was chosen, load the SketchCNN model
    if chosen_model == "model1":
        print("Using SketchCNN")
        #load the CNN model
        model = SketchCNN(num_classes=len(class_names)).to(device)
        #load pre-trained weights
        model.load_state_dict(torch.load("model1.pth", map_location=device))
    #otherwise, load the SketchRNN model
    else:
        print("Using SketchRNN")
        #load the RNN model
        model = SketchRNN(num_classes=len(class_names)).to(device)
        #load pre-trained weights
        model.load_state_dict(torch.load("model2.pth", map_location=device))

    #set the model to evaluation mode
    model.eval()

    #check if the request contains "pixels" (for CNN input)
    if "pixels" in data:
        pixels = data["pixels"]  #extract pixel data
        #convert to a PyTorch tensor and reshape to match the expected input shape
        input_tensor = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 28, 28).to(device)

        with torch.no_grad(): #disable gradient computation for inference
            logits = model(input_tensor)  #get raw model outputs
            probs = F.softmax(logits, dim=1).squeeze()  #apply softmax to get probabilities

        print("Raw probabilities:", probs.cpu().numpy()) #print raw probabilities

    #check if the request contains "sequence" (for RNN input)
    elif "sequence" in data:
        sequence = data["sequence"]  #extract sequence data
        #convert to a PyTorch tensor and add batch dimension
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        #create a tensor for the sequence length
        length_tensor = torch.tensor([len(sequence)], dtype=torch.int64).to(device)

        with torch.no_grad(): #disable gradient computation for inference
            logits = model(input_tensor, length_tensor)  #get raw model outputs
            probs = F.softmax(logits, dim=1).squeeze()  #apply softmax to get probabilities

        print("Raw probabilities:", probs.cpu().numpy()) #print raw probabilities

    else:
        #if neither "pixels" nor "sequence" is provided, return an error
        return jsonify({"error": "Missing 'pixels' or 'sequence' in request"}), 400

    #prepare the results as a list of dictionaries with labels and confidence scores
    results = [
        {"label": class_names[i], "confidence": round(float(prob), 4)}
        for i, prob in enumerate(probs)
    ]

    #sort results by confidence in descending order
    results.sort(key=lambda x: x["confidence"], reverse=True)

    print("Sorted results:", results[:5]) #print the top 5 results

    #return the sorted results as a JSON response
    return jsonify(results)