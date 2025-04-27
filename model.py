import torch.nn as nn

class SketchCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        #convolutional blocks, inspired by AlexNet
        #the architecture consists of several convolutional layers followed by ReLU activations and max pooling
        self.conv_blocks = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1), #Input: [1, 28, 28], Output: [8, 28, 28] -> (n+2p-f)/s +1 = (28+2*1-3)/2 +1 = 28 -> 28x28xnum_of_channels
            nn.ReLU(),

            # Second conv layer
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1), # Output: [16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: [16, 14, 14]

            # Third conv layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # Output: [32, 14, 14]
            nn.ReLU(),

            # Fourth conv layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # Output: [64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: [64, 7, 7]
        )

        #fully connected block
        #consists of two linear layers with ReLU activations and dropout for regularization
        self.fc_block = nn.Sequential(
            nn.Flatten(), #flatten the input [64, 7, 7] -> [3136]
            nn.Linear(64 * 7 * 7, 128), #fully connected layer
            nn.ReLU(), #ReLU activation function
            nn.Dropout(0.3), #dropout for regularization
            nn.Linear(128, out_features=num_classes) #fully connected layer to output class scores
        )

    #forward pass: takes an input tensor, applies the convolutional blocks, and then the fully connected block
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_block(x)
        return x


class SketchRNN(nn.Module):
    """
    RNN-based model for classifying hand-drawn sketches using LSTM.
    Input: Sequences of pen movements (dx, dy, pen_state)
    Output: Class probabilities over 20 sketch categories
    """

    def __init__(self, num_classes, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2):
        """
        num_classes (int): Number of output categories/classes
        input_dim (int): Input size at each timestep (dx, dy, pen_state -> 3)
        hidden_dim (int): LSTM hidden size (memory per step)
        num_layers (int): How many LSTM layers to stack
        dropout (float): Dropout between layers (for regularization)
        """
        super().__init__()

        #embed each input timestep (dx, dy, pen) into a higher-dimensional vector space
        #this helps the LSTM learn better representations of strokes
        #we use a linear layer that takes the input of size 3 (dx, dy, pen) and projects it to a higher-dimensional space (hidden_dim=128)
        self.embedding = nn.Linear(input_dim, hidden_dim)  # projects 3 -> 128

        #define a multi-layer LSTM to process the sequence step by step
        #the LSTM will take the embedded input and learn to capture the temporal dependencies in the sketch sequences
        #it will output a hidden state for each timestep, which can be used for classification
        #the LSTM will also have a dropout layer between the layers to prevent overfitting
        self.lstm = nn.LSTM(
            input_size=hidden_dim, #the raw input is [dx, dy, pen] -> 3 values, but note that the embedding step will expand it from 3 to 128
            hidden_size=hidden_dim, #for every stroke, the model outputs a vector of 128 numbers that represent its "understanding" so far
            num_layers=num_layers, #how many LSTM layers to stack
            batch_first=True, #tells pytorch that the input shape is [batch_size, time_steps (strokes per sketch), features (dimension after embedding)]
            dropout=dropout
        )

        #final classification block
        #this will take the final hidden state of the LSTM (a summary of the entire sequence) and project it to the number of classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )


    #forward pass: takes a batch of sequences and their lengths and processes the sequences through the embedding layer, LSTM, and classifier
    def forward(self, x, lengths):
        """
        x: Tensor of shape [B, T, 3] -> stroke sequences
        lengths: Tensor of shape [B] -> real lengths before padding
        returns output logits: [B, num_classes], each of the B sketches gets a classification output (each row tells you how confident the model is for each class, per sketch)
        """

        #project each input to hidden_dim by embedding
        x = self.embedding(x)  # [B, T, 3] -> [B, T, H=128]

        #pack the sequence to ignore padding in LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        #LSTM over the packed input
        #packed_out will be the output of the LSTM for each timestep, but we only need the last hidden state
        #h_n is the final hidden state from each layer
        #since we have 2 LSTM layers, h_n will have shape [num_layers, B, H]
        packed_out, (h_n, c_n) = self.lstm(packed) 
        
        #we only want the final layer's output
        final_hidden = h_n[-1] #[B, H] -> each sketch's last hidden state

        #pass the final hidden state through the classifier
        #this will output the class probabilities for each sketch
        #the classifier takes the final hidden state and projects it to the number of classes
        out = self.classifier(final_hidden)  #[B, num_classes]

        return out