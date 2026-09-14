import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # no display on Rangpur - save figures instead of showing them
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader

LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Project the latent noise vector up to 512*4*4 values with a
        # linear layer, then reshape into a [512, 4, 4] feature map - the
        # small spatial starting point that the ConvTranspose2d stack
        # below progressively upsamples up to 128x128.
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        # Each stage doubles the spatial size and roughly halves the
        # channel count (kernel_size=4, stride=2, padding=1 doubles
        # resolution: (4-1)*2 - 2*1 + 4 = 8, i.e. 4x4 -> 8x8, etc.).
        # BatchNorm2d + ReLU follow every ConvTranspose2d except the
        # final one.
        self.deconv = nn.Sequential(
            # 4x4, 512 channels -> 8x8, 256 channels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8, 256 channels -> 16x16, 128 channels
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16, 128 channels -> 32x32, 64 channels
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32, 64 channels -> 64x64, 32 channels
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64x64, 32 channels -> 128x128, 1 channel (final output).
            # No BatchNorm/ReLU here - Sigmoid squashes the output into
            # [0, 1], matching the real training images' pixel range.
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)                # [batch, 512*4*4]
        x = x.view(x.size(0), 512, 4, 4)  # [batch, 512, 4, 4]
        x = self.deconv(x)            # [batch, 1, 128, 128]
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Mirror image of the Generator: Conv2d(kernel_size=4, stride=2,
        # padding=1) halves spatial size each stage while roughly
        # doubling the channel count, taking 128x128 down to 4x4.
        # BatchNorm2d follows every Conv2d except the first (BatchNorm on
        # the input layer tends to hurt GAN training), and LeakyReLU(0.2)
        # is used throughout instead of ReLU - the standard choice for
        # discriminators, since it still passes a (small) gradient for
        # negative inputs and avoids dead units during the already
        # unstable adversarial training process.
        self.conv = nn.Sequential(
            # 128x128, 1 channel -> 64x64, 32 channels (no BatchNorm here)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64, 32 channels -> 32x32, 64 channels
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32, 64 channels -> 16x16, 128 channels
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16, 128 channels -> 8x8, 256 channels
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8, 256 channels -> 4x4, 512 channels
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Collapse the final 4x4, 512-channel feature map down to a
        # single value (kernel_size=4, stride=1, padding=0 maps a 4x4
        # input to exactly 1x1), then flatten and squash with Sigmoid
        # into a real/fake confidence score in [0, 1].
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)              # [batch, 512, 4, 4]
        x = self.final_conv(x)        # [batch, 1, 1, 1]
        x = x.view(x.size(0), -1)     # [batch, 1] - flatten to one value per image
        x = self.sigmoid(x)
        return x


# --- Data pipeline ---
# The training loop below needs real image batches to train the
# Discriminator against - same image-only loading as the VAE notebook
# (grayscale, resized to 128x128, no labels/masks needed for a GAN).
class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image


data_root = "/home/groups/comp3710/OASIS"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = BrainMRIDataset(
    os.path.join(data_root, "keras_png_slices_train"), transform=transform
)

BATCH_SIZE = 64
data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# --- Device / model setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# Separate optimizers - each only updates its own network's parameters.
# lr=0.0002, betas=(0.5, 0.999) is the standard, well-established
# setting for GAN training (from the DCGAN paper); the lower beta1 (vs.
# Adam's usual default of 0.9) reduces the amount of momentum applied to
# past gradients, which otherwise tends to make the already-unstable
# adversarial training oscillate more.
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

NUM_EPOCHS = 50

# Loss history, tracked once per epoch (using each epoch's last batch
# loss, same as what's printed) so training progress can be plotted
# afterward.
g_losses = []
d_losses = []

# --- Training loop ---
for epoch in range(NUM_EPOCHS):
    for real_images in data_loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # --- 5a. Update the Discriminator ---
        optimizer_d.zero_grad()

        # One-sided label smoothing: train the Discriminator to output
        # 0.9 (not a maximally confident 1.0) for real images. This is a
        # standard GAN stabilization technique - without it, the
        # Discriminator can become overconfident and saturate, which
        # starves the Generator of useful gradient signal. Only applied
        # to real labels here, not fake_labels (those stay at exactly 0)
        # and not the Generator's own loss below (it still targets 1.0).
        real_labels_smooth = torch.full((batch_size, 1), 0.9, device=device)

        real_output = discriminator(real_images)
        loss_d_real = criterion(real_output, real_labels_smooth)

        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)
        # .detach() creates a copy of fake_images that's cut off from the
        # Generator's computation graph. Without it, calling backward()
        # here (on a loss that only the Discriminator's optimizer will
        # step on) would still needlessly compute gradients all the way
        # back through the Generator's weights too, since fake_images
        # was produced by it - wasted computation at best, and would
        # leave stale/incorrect Generator gradients lying around at
        # worst if anything used them before the Generator's own step
        # below.
        fake_output = discriminator(fake_images.detach())
        loss_d_fake = criterion(fake_output, fake_labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # --- 5b. Update the Generator ---
        optimizer_g.zero_grad()

        # Generate a fresh fake batch - no detach() this time, since we
        # need gradients to flow all the way back into the Generator's
        # weights for its own update.
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)
        output = discriminator(fake_images)

        # The adversarial trick: even though these images are fake, the
        # Generator's loss is computed AS IF the target label were
        # "real" (1). Minimizing this loss pushes the Generator's
        # weights toward whatever makes the Discriminator output closer
        # to 1 for these images - i.e. the Generator's objective isn't
        # to agree with the Discriminator, it's to fool it.
        loss_g = criterion(output, real_labels)
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

    g_losses.append(loss_g.item())
    d_losses.append(loss_d.item())

    # Every few epochs, save a grid of generated samples from fixed-size
    # fresh noise, so training progress - and any mode collapse (the
    # Generator producing the same few images regardless of input noise)
    # - can be tracked visually over time.
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, LATENT_DIM, device=device)
            samples = generator(sample_noise)
        vutils.save_image(samples, f"gan_samples_epoch_{epoch + 1}.png", nrow=4)

# Plot both loss histories together so the adversarial balance (or
# imbalance - e.g. one loss collapsing toward 0 while the other diverges)
# over the course of training can be inspected after the run.
plt.figure(figsize=(8, 5))
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Losses")
plt.legend()
plt.savefig("gan_loss_plot.png")
plt.close()
