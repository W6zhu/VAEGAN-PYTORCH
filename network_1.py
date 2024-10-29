import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy

torch.autograd.set_detect_anomaly(True)

# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, return_layer=False):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten)  # Ensure no inplace operation
        if return_layer:
            layer_ten = ten.clone()
            return ten, layer_ten
        return ten

# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten)  # Removed inplace=True
        return ten

class Encoder(nn.Module):
    def __init__(self, z_size=128):
        super(Encoder, self).__init__()
        self.anatomical_encoder = EncoderBlock(channel_in=1, channel_out=64)
        self.fat_fraction_encoder = EncoderBlock(channel_in=1, channel_out=64)
        
        # Adjusted channels in shared layers after concatenation
        self.shared_layers = nn.Sequential(
            EncoderBlock(channel_in=128, channel_out=128),  # Adjusted input channels after concatenation
            EncoderBlock(channel_in=128, channel_out=256),
            EncoderBlock(channel_in=256, channel_out=256)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 8 * 8, out_features=512, bias=False),  # Adjusted for 128x128 input
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(True)
        )

        self.l_mu = nn.Linear(in_features=512, out_features=z_size)
        self.l_var = nn.Linear(in_features=512, out_features=z_size)

    def forward(self, anatomical_img, fat_fraction_img):
        anatomical_encoded = self.anatomical_encoder(anatomical_img)
        fat_fraction_encoded = self.fat_fraction_encoder(fat_fraction_img)
        combined_encoding = torch.cat((anatomical_encoded, fat_fraction_encoded), dim=1)  # Concatenate along channels
        
        combined_encoding = self.shared_layers(combined_encoding)
        print("Shape of combined_encoding before flattening:", combined_encoding.shape)
        combined_encoding = combined_encoding.view(len(combined_encoding), -1)  # Flatten

        ten = self.fc(combined_encoding)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        print("Shapes after Encoder -> mu:", mu.shape, ", logvar:", logvar.shape)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_size):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=8 * 8 * 256, bias=False),
            nn.BatchNorm1d(num_features=8 * 8 * 256, momentum=0.9),
            nn.ReLU(True)
        )

        self.conv = nn.Sequential(
            DecoderBlock(channel_in=256, channel_out=128),
            DecoderBlock(channel_in=128, channel_out=64),
            DecoderBlock(channel_in=64, channel_out=32),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2),  # Changed to 128 channels
            nn.Tanh()
        )

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 8, 8)
        ten = self.conv(ten)
        print("Shape after Decoder:", ten.shape)  # This should now print a shape with 128 channels
        return ten
    

class Discriminator(nn.Module):
    def __init__(self, channel_in=1, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_level = recon_level
        self.conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU()),
            EncoderBlock(channel_in=32, channel_out=128),
            EncoderBlock(channel_in=128, channel_out=256),
            EncoderBlock(channel_in=256, channel_out=256)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 256, out_features=1024, bias=False),
            nn.BatchNorm1d(num_features=1024, momentum=0.9),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, ten, ten_original, ten_sampled):
        ten = torch.cat((ten, ten_original, ten_sampled), 0)
        
        for i, lay in enumerate(self.conv):
            if i == self.recon_level:
                ten, layer_ten = lay(ten, return_layer=True)
                layer_ten = layer_ten.view(len(layer_ten), -1)
            else:
                ten = lay(ten)
        
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        return layer_ten, torch.sigmoid(ten)

import torch.nn.functional as F  # Add this import for resizing

class VaeGan(nn.Module):
    def __init__(self, z_size=128, recon_level=3):
        super(VaeGan, self).__init__()
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size)
        self.discriminator = Discriminator(channel_in=128, recon_level=recon_level)
        self.init_parameters()

        # New layer to match channels of `ten_original` to `ten`
        self.channel_match = nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                scale /= numpy.sqrt(3)
                nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, anatomical_img, fat_fraction_img=None, gen_size=10):
        if self.training:
            ten_original = anatomical_img
            mus, log_variances = self.encoder(anatomical_img, fat_fraction_img)
            variances = torch.exp(log_variances * 0.5)
            ten_from_normal = torch.randn(len(anatomical_img), self.z_size, device=ten_original.device, requires_grad=True)
            ten = ten_from_normal * variances + mus
            ten = self.decoder(ten)
            
            ten_sampled = torch.randn(len(anatomical_img), self.z_size, device=ten_original.device, requires_grad=True)
            ten_sampled = self.decoder(ten_sampled)
            
            # Only match channels if `ten_original` has a single channel
            if ten_original.shape[1] == 1:
                ten_original = self.channel_match(ten_original)
            ten_original = F.interpolate(ten_original, size=(ten.shape[2], ten.shape[3]), mode='bilinear', align_corners=False)
            ten_sampled = F.interpolate(ten_sampled, size=(ten.shape[2], ten.shape[3]), mode='bilinear', align_corners=False)

            # Debug statements to confirm shapes
            print(f"Shape of `ten`: {ten.shape}")
            print(f"Shape of `ten_original` after resize: {ten_original.shape}")
            print(f"Shape of `ten_sampled` after resize: {ten_sampled.shape}")
            
            ten_layer, ten_class = self.discriminator(ten, ten_original, ten_sampled)
            return ten, ten_class, ten_layer, mus, log_variances
        else:
            if anatomical_img is None:
                ten = torch.randn(gen_size, self.z_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)
                ten = self.decoder(ten)
            else:
                mus, log_variances = self.encoder(anatomical_img, fat_fraction_img)
                variances = torch.exp(log_variances * 0.5)
                ten_from_normal = torch.randn(len(anatomical_img), self.z_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=True)
                ten = ten_from_normal * variances + mus
                ten = self.decoder(ten)
            return ten

    @staticmethod
    def loss(data_target, out, out_layer_original, out_layer_predicted, out_layer_sampled, 
            out_labels_original, out_labels_predicted, out_labels_sampled, mus, variances):
        # Expand `data_target` to match the number of channels in `out`
        data_target = data_target.expand(-1, out.shape[1], -1, -1)  # Match channels
        
        # Resize spatial dimensions to match `out`
        data_target = F.interpolate(data_target, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=False)
        
        # Calculate losses
        nle_value = F.binary_cross_entropy_with_logits(out, data_target)
        mse_value_1 = F.mse_loss(out_layer_original, out_layer_predicted)
        mse_value_2 = F.mse_loss(out_layer_original, out_layer_sampled)
        kl_value = -0.5 * torch.sum(1 + variances - mus.pow(2) - variances.exp())
        bce_dis_original_value = F.binary_cross_entropy(out_labels_original, torch.ones_like(out_labels_original))
        bce_dis_sampled_value = F.binary_cross_entropy(out_labels_sampled, torch.zeros_like(out_labels_sampled))
        bce_dis_predicted_value = F.binary_cross_entropy(out_labels_predicted, torch.zeros_like(out_labels_predicted))
        bce_gen_sampled_value = F.binary_cross_entropy(out_labels_sampled, torch.ones_like(out_labels_sampled))
        bce_gen_predicted_value = F.binary_cross_entropy(out_labels_predicted, torch.ones_like(out_labels_predicted))
        
        return (nle_value, kl_value, mse_value_1, mse_value_2, 
                bce_dis_original_value, bce_dis_sampled_value, 
                bce_dis_predicted_value, bce_gen_sampled_value, bce_gen_predicted_value)
