import torch
import torch.nn as nn
from torchview import draw_graph
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional  as Fn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from piq import ssim
import os

import wandb
# wandb.login()

wandb.init(
    project="Transformer-Decoder",  
    name="experiment-1",    
    id="uqcub7jq",  
    resume="allow",
    # config={                       
    #     "epochs": 1000,
    #     "batch_size": 64,
    # }
)

class VectorQuantizeImage(nn.Module):
    def __init__(self, codeBookDim = 64, embeddingDim = 32, decay = 0.99, eps = 1e-5):
        super().__init__()

        self.codeBookDim = codeBookDim
        self.embeddingDim = embeddingDim
        self.decay = decay
        self.eps = eps
        self.dead_codeBook_threshold = codeBookDim * 0.6

        self.codebook = nn.Embedding(codeBookDim, embeddingDim)
        nn.init.xavier_uniform_(self.codebook.weight.data)

        self.register_buffer('ema_Count', torch.zeros(codeBookDim))
        self.register_buffer('ema_Weight', self.codebook.weight.data.clone())

    def forward(self, x):
        x_reshaped = x.view(-1, self.embeddingDim)

        distance = (torch.sum(x_reshaped**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(x_reshaped, self.codebook.weight.t()))
        
        encoding_indices = torch.argmin(distance, dim=1) 
        encodings = Fn.one_hot(encoding_indices, self.codeBookDim).type(x_reshaped.dtype)
        quantized = torch.matmul(encodings, self.codebook.weight)

        if self.training:
            self.ema_Count = self.decay * self.ema_Count + (1 - self.decay) * torch.sum(encodings, 0)
            
            x_reshaped_sum = torch.matmul(encodings.t(), x_reshaped.detach())
            self.ema_Weight = self.decay * self.ema_Weight + (1 - self.decay) * x_reshaped_sum
            
            n = torch.clamp(self.ema_Count, min=self.eps)
            updated_embeddings = self.ema_Weight / n.unsqueeze(1)
            self.codebook.weight.data.copy_(updated_embeddings)

        
        avg_probs = torch.mean(encodings, dim=0)
        log_encoding_sum = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(log_encoding_sum)

        entropy = log_encoding_sum
        normalized_entropy = entropy / torch.log(torch.tensor(self.codeBookDim, device=x.device))
        diversity_loss = 1.0 - normalized_entropy

        return quantized, encoding_indices, perplexity, diversity_loss
        
        
# vq = VectorQuantizeImage(codeBookDim=64,embeddingDim=32)
# rand = torch.randn(1024,32)
# quantized, encoding_indices, perplexity, diversity_loss = vq(rand)
# quantized.shape, encoding_indices.shape, perplexity, diversity_loss

class VecQVAE(nn.Module):
    def __init__(self, inChannels = 1, hiddenDim = 32, codeBookdim = 128, embedDim = 128):
        super().__init__()
        self.inChannels = inChannels
        self.hiddenDim = hiddenDim
        self.codeBookdim = codeBookdim
        self.embedDim = embedDim

        self.encoder = nn.Sequential(
            nn.Conv2d(inChannels, hiddenDim, 4, 2, 1), 
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2 * hiddenDim, 2 * hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2 * hiddenDim, embedDim, 1),
        )

        self.vector_quantize = VectorQuantizeImage(codeBookDim=codeBookdim,embeddingDim=embedDim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2 * hiddenDim, 2 * hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2 * hiddenDim, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, inChannels, 1),
            nn.Sigmoid()
        )

    def encodeImage(self, x, noise_std = 0.15):
        if self.training:
            encodedOut = self.encoder(x)
            encodedOut = encodedOut + torch.randn_like(encodedOut) * noise_std
        else:
            encodedOut = self.encoder(x)

        return encodedOut

    def decodeImage(self, quantized_vector):
        decodedOut = self.decoder(quantized_vector)
        return decodedOut

    def forward(self, x):
        batch_size, inChannels, height, width = x.shape
        encodedOut = self.encodeImage(x)
        batch_size, encoded_channel, encoded_height, encoded_width = encodedOut.shape
        
        # print(f"Encoded Shape: {encodedOut.shape}")

        
        vectorize_input = rearrange(encodedOut, 'b c h w -> (b h w) c')
        quantized_vectors, encoding_indices, perplexity, diversity_loss  = self.vector_quantize(vectorize_input)
        codebook_loss = Fn.mse_loss(vectorize_input.detach(), quantized_vectors)
        commitment_loss = Fn.mse_loss(vectorize_input, quantized_vectors.detach())

        quantized_vectors = vectorize_input + (quantized_vectors - vectorize_input).detach()
        # print(f"CodeBook Loss: {codebook_loss} , Commitment Loss: {commitment_loss}")
        # print(f"Quantized SHape: {quantized_vectors.shape}")

        decoder_input = rearrange(quantized_vectors, '(b h w) d -> b d h w', d = encoded_channel, h = encoded_height, w = encoded_width)
        # print(f"Decoded Input SHape: {decoder_input.shape}")
        decodedOut = self.decodeImage(decoder_input)

        # print(f"Decoded SHape: {decodedOut.shape}")
        
        return decoder_input, decodedOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss

# VQ = VecQVAE(inChannels = 1, hiddenDim = 256, codeBookdim = 128, embedDim = 64)
# test = torch.randn(32, 1, 64, 64)
# quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = VQ(test)
# quantized_latents.shape, decoderOut.shape, codebook_loss, commitment_loss, encoding_indices.shape, perplexity, diversity_loss

class FrameDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.data = self.data/255.0
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.data[index]
    

# test = torch.randn(1000, 1, 64, 64)
# out = FrameDataset(test)
# print(out.__len__())
# x, y = out.__getitem__(0)
# x.shape, y.shape

codeBookdim = 128
embedDim = 64
hiddenDim = 256
inChannels = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torchDataset = FrameDataset(flattened_frame_dataset)
# dataloader = DataLoader(torchDataset, batch_size=64, shuffle = True)
modelA = VecQVAE(inChannels = inChannels, hiddenDim = hiddenDim, codeBookdim = codeBookdim, embedDim = embedDim).to(device)
lossFn = nn.MSELoss()
optimizerA = torch.optim.Adam([
                    {'params': modelA.encoder.parameters(), 'lr': 2e-4},
                    {'params': modelA.decoder.parameters(), 'lr': 2e-4},
                    {'params': modelA.vector_quantize.parameters(), 'lr': 1e-4}
                ], weight_decay=1e-5)
schedulerA = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizerA, T_0=10, T_mult=2, eta_min=1e-6
            )

epochs = 1000
modelValA = torch.load("./model/VQVAE/vqvae-5.pt", map_location=torch.device('cpu'))
modelA.load_state_dict(modelValA)


data = np.load("./data/mnist_test_seq.npy")
print(data.shape)
data = np.transpose(data, (1, 0, 2, 3))
data = torch.from_numpy(data).to(dtype=torch.float32)
data = data.unsqueeze(2)
print(data.shape)


class VideoSequenceData(Dataset):
    def __init__(self, data, modelA, input_length=10, cache_dir="./data/encodingIndices"):
        super().__init__()
        self.data = data
        self.modelA = modelA.eval().to(device)
        self.input_length = input_length
        self.sequence_per_video = 20 - self.input_length
        self.total_samples = len(data) * self.sequence_per_video
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.video_cache = {}
    
    def _get_cache_path(self, video_idx):
        return os.path.join(self.cache_dir, f"video_{video_idx}_indices.pt")
    
    def _compute_video_indices(self, video_idx):
        cache_path = self._get_cache_path(video_idx)
        
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location='cpu')
        
        video = self.data[video_idx]
        indices = []
        
        with torch.inference_mode():
            frames = video / 255.0
            frames = frames.to(device)
            
            for frame in frames:
                frame = frame.unsqueeze(0)
                _, _, _, _, encoding_indices, _, _ = self.modelA(frame)
                indices.append(encoding_indices.cpu()) 
        
        video_indices = torch.stack(indices)
        
        torch.save(video_indices, cache_path)
        
        return video_indices
    
    def __getitem__(self, idx):
        video_idx = idx // self.sequence_per_video
        start = idx % self.sequence_per_video
        end = start + self.input_length + 1
        
        if video_idx not in self.video_cache:
            self.video_cache[video_idx] = self._compute_video_indices(video_idx)
        
        codebook_indices = self.video_cache[video_idx][start:end]
        
        X = codebook_indices[:-1]
        Y = codebook_indices[1:]
        return X, Y
    
    def __len__(self):
        return self.total_samples
    
# val = VideoSequenceData(data, modelA)
# X2, Y2 = val.__getitem__(2)
# X0, Y0 = val.__getitem__(0)

# print(X0.shape, Y0.shape)
# val.__len__()


class TransformerDecoderModel(nn.Module):
    def __init__(self, sequence_len, num_tokens, codebook_size, embedding_dim, num_layers, numHeads, feedForwardDim=2048, drop=0.2):
        super().__init__()
        self.sequence_len = sequence_len
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.total_tokens = sequence_len * num_tokens
        
        self.token_embedding = nn.Embedding(codebook_size, embedding_dim)        
        self.position_embedding = nn.Embedding(self.total_tokens, embedding_dim)
        
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=numHeads, 
            dim_feedforward=feedForwardDim, 
            dropout=drop, 
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(drop)
        self.output_layer = nn.Linear(embedding_dim, codebook_size)

    def forward(self, x):

        batch_size, seq_len, tokens = x.shape
        x_flat = rearrange(x, 'b s t -> b (s t)')
        # print(f"Flattened Shape: {x_flat.shape}")
        token_embeds = self.token_embedding(x_flat)
        
        positions = torch.arange(0, self.total_tokens, device=x.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        decoder_input = self.dropout(token_embeds + pos_embeds)
        # print(f"Decoder Input: {decoder_input.shape}")
        
        seq_len = self.total_tokens
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), 
            diagonal=1
        ).to(x.device)
        
        decoder_out = self.decoder(decoder_input, mask=causal_mask)
        
        logits = self.output_layer(decoder_out)
        return logits

tModel = TransformerDecoderModel(
    sequence_len=10,
    num_tokens=256,
    codebook_size=128,
    embedding_dim=512,
    num_layers=6,
    numHeads=4
)

# test_indices = torch.randint(0, 128, (1, 10, 256))
# logits = tModel(test_indices)
# print(logits.shape)

batch_size = 4
num_tokens=256
sequence_len = 10
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numHeads = 4
dropout = 0.2
num_layers = 6
embed_dim = 1024

torchDataset = VideoSequenceData(data, modelA)
dataloader = DataLoader(torchDataset, batch_size=batch_size, shuffle = True)
modelB = TransformerDecoderModel( sequence_len=sequence_len,num_tokens = num_tokens, codebook_size=codeBookdim, embedding_dim=embed_dim, num_layers=num_layers, numHeads=numHeads).to(device)
modelB = torch.nn.DataParallel(modelB)
modelB.to(device)

lossFn =  nn.CrossEntropyLoss()
optimizerB = torch.optim.AdamW(params=modelB.parameters(), lr=learning_rate, weight_decay=1e-5)
schedulerB = torch.optim.lr_scheduler.OneCycleLR(
    optimizerB, max_lr=3e-4, steps_per_epoch=len(dataloader),
    epochs=epochs, anneal_strategy='cos', final_div_factor=1e4
)

epochs = 1000

modelValB = torch.load("./model/TDecoder/tdecoder.pt", map_location=torch.device('cpu'))
modelB.load_state_dict(modelValB)

for each_epoch in range(epochs):
    modelB.train()
    lossVal = 0.0
    
    loop = tqdm(dataloader, f"{each_epoch}/{epochs}")

    for X, Y in loop:
        X = X.to(device).long()
        Y = Y.to(device).long()
        y_pred = modelB(X)
        y_pred = y_pred

        loss = lossFn(y_pred.view(-1, codeBookdim), Y.view(-1))
        lossVal += loss.item()
        
        
        optimizerB.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelB.parameters(), max_norm=1.0)
        optimizerB.step()
        loop.set_postfix({"Total Loss: ": f"{lossVal}"})
    #     break
    # break

    lossVal /= len(dataloader)   
    
    torch.save(modelB.state_dict(), "./model/TDecoder/tdecoder.pt")
    wandb.log({
        "Epoch": each_epoch,
        "Transformer Decoder LR": optimizerB.param_groups[0]['lr'],
        "Transformer Loss": lossVal,
    })
    schedulerB.step()


