import torch.nn as nn

class get_lstm(nn.Module):
    def __init__(self, inputDim, hiddenDim, num_layers, outputDim, sequence_length):
        super(get_lstm, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            num_layers = num_layers,
                            batch_first = False)
        self.output_layer1 = nn.Linear(hiddenDim, outputDim)
        self.output_layer2 = nn.Linear(sequence_length, 1)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        # output = self.output_layer(output[:, -1, :]) #全結合層
        # print("output.shape in forward1", output.shape)
        output = self.output_layer1(output)
        output = output.permute(1, 2, 0)
        output = self.output_layer2(output)
        output = output.squeeze()
        # print("output.shape in forward", output.shape)
        return output