import torch
import numpy
import argparse
import time
import os
from torchvision.utils import save_image
import torch.nn.functional as F

# print(torch.cuda.is_available())
# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)
from network_1 import VaeGan
from torch.utils.tensorboard import SummaryWriter
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
import progressbar
from generator import MRIPairedDataset, visualize_and_save_sample

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAEGAN")
    parser.add_argument("--anatomical_folder", action="store", dest="anatomical_folder", help="Path to anatomical MRI images")
    parser.add_argument("--fat_fraction_folder", action="store", dest="fat_fraction_folder", help="Path to fat fraction MRI images")
    parser.add_argument("--n_epochs", default=12, action="store", type=int, dest="n_epochs")
    parser.add_argument("--z_size", default=128, action="store", type=int, dest="z_size")
    parser.add_argument("--recon_level", default=3, action="store", type=int, dest="recon_level")
    parser.add_argument("--lambda_mse", default=1e-6, action="store", type=float, dest="lambda_mse")
    parser.add_argument("--lr", default=3e-4, action="store", type=float, dest="lr")
    parser.add_argument("--decay_lr", default=0.75, action="store", type=float, dest="decay_lr")
    parser.add_argument("--decay_mse", default=1, action="store", type=float, dest="decay_mse")
    parser.add_argument("--decay_margin", default=1, action="store", type=float, dest="decay_margin")
    parser.add_argument("--decay_equilibrium", default=1, action="store", type=float, dest="decay_equilibrium")

    args = parser.parse_args()
    anatomical_folder = args.anatomical_folder
    fat_fraction_folder = args.fat_fraction_folder
    z_size = args.z_size
    recon_level = args.recon_level
    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr

    dataset = MRIPairedDataset(anatomical_folder, fat_fraction_folder)

    # Visualize and save sample images to check data loading
    visualize_and_save_sample(dataset, save_path="output_images/originaltest.png")

    writer = SummaryWriter(comment="_MRI_PAIRED")
    net = VaeGan(z_size=z_size, recon_level=recon_level).cuda()

    # Load paired MRI dataset
    dataloader = torch.utils.data.DataLoader(
        MRIPairedDataset(anatomical_folder, fat_fraction_folder), 
        batch_size=64, shuffle=True, num_workers=4
    )
    dataloader_test = torch.utils.data.DataLoader(
        MRIPairedDataset(anatomical_folder, fat_fraction_folder), 
        batch_size=100, shuffle=False, num_workers=1
    )

    # Ensure output directory exists
    os.makedirs("output_images", exist_ok=True)

    # Optimizer and Scheduler
    optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=lr, alpha=0.9, eps=1e-8)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=decay_lr)
    optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=decay_lr)
    optimizer_discriminator = RMSprop(params=net.discriminator.parameters(), lr=lr, alpha=0.9, eps=1e-8)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=decay_lr)

    # Training Loop
    for i in range(n_epochs):
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(dataloader)).start()
        epoch_loss_total = 0

        for j, (anatomical_img, fat_fraction_img) in enumerate(dataloader):
            net.train()
            anatomical_img = anatomical_img.float().cuda()
            fat_fraction_img = fat_fraction_img.float().cuda()

            # Forward pass with paired inputs
            out, out_labels, out_layer, mus, variances = net(anatomical_img, fat_fraction_img)
            data_target = fat_fraction_img

            # Loss Calculation
            nle_value, kl_value, mse_value_1, mse_value_2, bce_dis_original_value, bce_dis_sampled_value, \
            bce_dis_predicted_value, bce_gen_sampled_value, bce_gen_predicted_value = VaeGan.loss(
                data_target, out, out_layer[:len(data_target)], out_layer[:len(data_target)], out_layer[-len(data_target):],
                out_labels[:len(data_target)], out_labels[:len(data_target)], out_labels[-len(data_target):], mus, variances
            )
            
            # Combined Losses
            loss_encoder = torch.sum(kl_value) + torch.sum(mse_value_1) + torch.sum(mse_value_2)
            loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value) + torch.sum(bce_dis_predicted_value)
            loss_decoder = torch.sum(lambda_mse / 2 * mse_value_1) + torch.sum(lambda_mse / 2 * mse_value_2) + \
                           (1.0 - lambda_mse) * (torch.sum(bce_gen_predicted_value) + torch.sum(bce_gen_sampled_value))

            # Combine all losses
            loss_total = loss_encoder + loss_decoder + loss_discriminator
            epoch_loss_total += loss_total.item()

            # Backpropagation
            net.zero_grad()
            loss_total.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
            optimizer_discriminator.step()

            # Logging each loss component
            writer.add_scalar('Loss/Total', loss_total.item(), i * len(dataloader) + j)
            writer.add_scalar('Loss/Encoder', loss_encoder.item(), i * len(dataloader) + j)
            writer.add_scalar('Loss/Decoder', loss_decoder.item(), i * len(dataloader) + j)
            writer.add_scalar('Loss/Discriminator', loss_discriminator.item(), i * len(dataloader) + j)
            writer.add_scalar('Loss/NLE', nle_value.mean().item(), i * len(dataloader) + j)
            writer.add_scalar('Loss/KL', kl_value.mean().item(), i * len(dataloader) + j)

        writer.add_scalar('Epoch/Average_Total_Loss', epoch_loss_total / len(dataloader), i)

        # Scheduler step
        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        # End timing the epoch and print duration
        epoch_duration = time.time() - start_time
        print(f"Epoch {i+1}/{n_epochs} completed in {epoch_duration:.2f} seconds with average loss: {epoch_loss_total / len(dataloader):.4f}")

        # Save a single slice for all four images side by side
        # Save a single slice for all four images side by side
        import torch.nn.functional as F
        from torchvision.utils import save_image

        # Function to normalize, convert to grayscale, and resize images to the target size
        def preprocess_image(image, target_size=(64, 64)):
            # Normalize to [0, 1] range
            image = torch.clamp((image - image.min()) / (image.max() - image.min()), 0, 1)
            # Reduce to single channel by averaging (in case image has more than 1 channel)
            if image.shape[0] > 1:
                image = image.mean(dim=0, keepdim=True)
            # Resize to target size
            image = F.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
            return image.squeeze(0)  # Ensure shape [1, H, W] for grayscale

        # Use this inside the epoch loop where you save the output images
        with torch.no_grad():
            net.eval()

            # Load a batch from the test set
            anatomical_img, fat_fraction_img = next(iter(dataloader_test))
            anatomical_img = anatomical_img.float().cuda()
            fat_fraction_img = fat_fraction_img.float().cuda()
            
            # Forward pass for reconstruction
            reconstructed = net(anatomical_img, fat_fraction_img)
            
            # Generate a synthetic image
            synthetic = net(None, 1)
            
            # Take only the first slice of each type, normalize, convert to grayscale, and resize
            slice_anatomical = preprocess_image(anatomical_img[0])
            slice_fat_fraction = preprocess_image(fat_fraction_img[0])
            slice_reconstructed = preprocess_image(reconstructed[0])
            slice_synthetic = preprocess_image(synthetic[0])

            # Concatenate along the width
            combined_image = torch.cat((slice_anatomical, slice_fat_fraction, slice_reconstructed, slice_synthetic), dim=2)
            
            # Save the combined image
            save_image(combined_image, f"output_images/epoch_{i+1}_comparison.png")


        progress.finish()
        
    writer.close()
