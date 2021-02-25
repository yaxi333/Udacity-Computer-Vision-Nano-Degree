import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embedded word vectors (of a specified size) as inputs  and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True) 
        #  If batch_first = True, then the input and output tensors are provided as (batch, seq, feature).
        
        self.linear = nn.Linear(hidden_size, vocab_size)
       
                
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # remove the <end>
        captions = captions[:, :-1] 
              
        # create embedded word vectors for each word in a sentence
        embeds = self.embed(captions)
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, _ = self.lstm(embeds)
        
        # fully connected layer outputs size : (batch_size, caption length, vocab_size)
        outputs = self.linear(lstm_out) 
        
        return outputs 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Initialize the list of predicted sentence each sentence with max of 20 words
        ls_predict = []
        
        # returns predicted sentence (list of tensor ids of length max_len)
        for i in range(max_len):
            
            # input is the feature, the output from Encoder
            # go through the Decoder and get the output for this word
            hiddens, states = self.lstm(inputs, states)           
            output = self.linear(hiddens.squeeze(1))
            
            # outputs[i,j,k] contains the model's predicted score, 
            # indicating how likely the j-th token in the i-th caption in the batch is the k-th token in the vocabulary
            # Output the largest value of the token in i-th caption as the final predicted word for i-th caption
            # This value is most probable one in this caption
            predicted = output.max(1)[1]

            # Add this word to predicted list
            ls_predict.append(predicted.tolist()[0])
            
            # Use this predicted word as the input of next state
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                     # (batch_size, 1, embed_size)

        return ls_predict 