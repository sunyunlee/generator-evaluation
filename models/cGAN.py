from Model import Model
import torch
import torch.nn as nn


class Generator(nn.Module):
    
    def __init__(self, latent_dim, label_dim, hidden_dim, output_h, output_w):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.output_h = output_h
        self.output_w = output_w
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + label_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_h * output_w),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], output_h, output_w)
        return img


class Discriminator(nn.Module):
    
    def __init__(self, input_h, input_w, label_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(label_dim, label_dim)

        self.model = nn.Sequential(
            nn.Linear(input_h * input_w + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.shape[0], -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
