import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

###########################################################################
#
#                       STEP ONE: create the DataLoader
#
###########################################################################

class DrivingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.data = []
        self.filenames = []

        for file in os.listdir(dataset_folder):
            if file.endswith(".npz"):
                traj = np.load(os.path.join(dataset_folder, file), allow_pickle=True)
                self.data.append(np.hstack((traj['orange_car_states'], traj['white_car_states'])))
                self.filenames.append(file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = self.data[idx]
        states = torch.tensor(states, dtype=torch.float32).flatten()
        return states, self.filenames[idx]


###########################################################################
#
#                       STEP TWO: create the VAE model
#
###########################################################################

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar



###########################################################################
#
#                       STEP THREE: train that diva!!
#
###########################################################################

if __name__ == "__main__":

    BETA = 0.1
    NUM_EPOCHS = 300
    EMBEDDING_DIM = 16

    for EMBEDDING_DIM in [2, 4, 6, 8, 16]:

        dataset = DrivingDataset("./data/episodes")
        print(f'{len(dataset)} episodes found in ./data/episodes')

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)#, num_workers=8)

        model = VAE(input_dim=2008, latent_dim=EMBEDDING_DIM)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        reconstruction_loss_fn = nn.MSELoss(reduction='sum')
        model.train()
        

        for epoch in tqdm(range(NUM_EPOCHS)):
            total_loss = 0
            recon_loss = 0

            for batch_data, _ in dataloader:  # Unpack batch and ignore filenames during training
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(batch_data)

                recon_loss = reconstruction_loss_fn(recon_batch, batch_data)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + BETA * kld_loss
                loss.backward()

                total_loss += loss.item()
                recon_loss += recon_loss.item()
                optimizer.step()
            tqdm.write(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}, Recon Loss: {recon_loss / len(dataset)}")

        torch.save(model.state_dict(), f"./data/models/vae_driving_{EMBEDDING_DIM}d.pth")
        

        # Extract and save embeddings for all data
        model.eval()
        with torch.no_grad():
            all_embeddings = []
            all_filenames = []

            for batch_data, batch_filenames in dataloader:
                _, mu, _ = model.forward(batch_data)
                all_embeddings.append(mu)
                all_filenames.extend(batch_filenames)

            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            np.savez(f"./data/embeddings/embeddings_driving_{EMBEDDING_DIM}d.npz", embeddings=all_embeddings, filenames=all_filenames)


