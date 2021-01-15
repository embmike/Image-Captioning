import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # Initialize the layers of the model
        super(EncoderCNN, self).__init__()
        
        # Transfer Learning using Resnet152
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Using the pretrained convolutional layers 
        # without the fully connected classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # New fully connected layer for image features
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        return features

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # Initialize the layers of the model
        super(DecoderRNN, self).__init__()
        
        # Embedding layer that turns words into a vector
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #  LSTM to learn and generate word output
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Generate the word output based on the vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    
    def init_weights(self):
        
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)      
        
    
    def forward(self, features, captions):
        
        embedded_in = self.word_embeddings(captions[:,:-1]) 
        lstm_in = torch.cat((features.unsqueeze(dim=1), embedded_in), dim=1)
        lstm_out, _ =self.lstm(lstm_in)
        outputs = self.linear(lstm_out)
        
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        captions = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            _, output = outputs.max(dim=1)                   
            captions.append(output.item())
            inputs = self.word_embeddings(output).unsqueeze(1)               
        return captions