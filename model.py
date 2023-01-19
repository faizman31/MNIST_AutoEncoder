import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,
                input_size,
                output_size,
                use_batch_norm=True,
                dropout_p=.4):
        self.input_size = input_size
        self.output_size= output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p=dropout_p
        
        super().__init__()

        def get_regularizer(use_batch_norm,size):
            if use_batch_norm:
                return nn.BatchNorm1d(size)
            else:
                return nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size,output_size),
            nn.ReLU(),
            get_regularizer(use_batch_norm,output_size)
        )

    def forward(self,x):

        y=self.block(x)

        return y

        

class AutoEncoder(nn.Module):

    def __init__(self,
                input_size,
                output_size,
                hidden_sizes,
                btl_size=2,
                use_batch_norm=True,
                dropout_p=.3,
                ):

        assert len(hidden_sizes) > 0 ,"You need to specify hidden layers."

        super().__init__()

        last_hidden_size = input_size
        encoder_blocks=[]

        for hidden_size in hidden_sizes[1:]:
            encoder_blocks+=[Block(
                            last_hidden_size,
                            hidden_size,
                            use_batch_norm,
                            )]
            last_hidden_size = hidden_size

        self.encoder = nn.Sequential(
            *encoder_blocks,
            nn.Linear(last_hidden_size,btl_size),
        )

        decoder_blocks=[]
        last_hidden_size=btl_size

        for hidden_size in hidden_sizes[1::-1]:
            decoder_blocks+=[Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm
                )]
            last_hidden_size=hidden_size
        
        self.decoder = nn.Sequential(
            *decoder_blocks,
            nn.Linear(last_hidden_size,input_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self,x):
        # |x| = (batch_size,input_size)
        # |z| = (batch_size,btl_size)
        z = self.encoder(x)

        # |y| = (batch_size,input_size)
        y = self.decoder(z)

        return y


        

    




        



