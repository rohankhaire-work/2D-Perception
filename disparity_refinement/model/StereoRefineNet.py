import torch.nn as nn
import torch.nn.init as init


class StereoRefineNet(nn.Module):
    def __init__(self):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_2_rgb = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_4_rgb = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_6_rgb = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_8_rgb = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_16_rgb = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.rgb_encoder.apply(weights_init)
        self.encoder_2_rgb.apply(weights_init)
        self.encoder_4_rgb.apply(weights_init)
        self.encoder_6_rgb.apply(weights_init)
        self.encoder_8_rgb.apply(weights_init)
        self.encoder_16_rgb.apply(weights_init)

        self.disp_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_2_disp = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_4_disp = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_6_disp = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_8_disp = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_16_disp = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.disp_encoder.apply(weights_init)
        self.encoder_2_disp.apply(weights_init)
        self.encoder_4_disp.apply(weights_init)
        self.encoder_6_disp.apply(weights_init)
        self.encoder_8_disp.apply(weights_init)
        self.encoder_16_disp.apply(weights_init)

        self.upsample_16 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_16.apply(weights_init)

        self.upsample_8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_8.apply(weights_init)

        self.upsample_6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_6.apply(weights_init)

        self.upsample_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_4.apply(weights_init)

        self.upsample_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_2.apply(weights_init)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1),
        )
        self.final_conv.apply(weights_init)
        self.final_disp = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, in_img):
        x_rgb = in_img[1]
        x_disp = in_img[0]

        stem_block_d = self.disp_encoder(x_disp)
        downsample_2_d = self.encoder_2_disp(stem_block_d)
        downsample_4_d = self.encoder_4_disp(downsample_2_d)
        downsample_6_d = self.encoder_6_disp(downsample_4_d)
        downsample_8_d = self.encoder_8_disp(downsample_6_d)
        downsample_16_d = self.encoder_16_disp(downsample_8_d)

        stem_block_rgb = self.rgb_encoder(x_rgb)
        downsample_2_rgb = self.encoder_2_rgb(stem_block_rgb)
        downsample_4_rgb = self.encoder_4_rgb(downsample_2_rgb)
        downsample_6_rgb = self.encoder_6_rgb(downsample_4_rgb)
        downsample_8_rgb = self.encoder_8_rgb(downsample_6_rgb)
        downsample_16_rgb = self.encoder_16_rgb(downsample_8_rgb)

        upsample_16 = self.upsample_16(downsample_16_d + downsample_16_rgb)
        upsample_8 = self.upsample_8(
            upsample_16 + downsample_8_d + downsample_8_rgb)
        upsample_6 = self.upsample_6(
            upsample_8 + downsample_6_d + downsample_6_rgb)
        upsample_4 = self.upsample_4(
            upsample_6 + downsample_4_d + downsample_4_rgb)
        upsample_2 = self.upsample_2(
            upsample_4 + downsample_2_rgb + downsample_2_d)

        upsample_1 = self.final_conv(upsample_2)

        pred_disp = self.final_disp(upsample_1)

        return pred_disp
