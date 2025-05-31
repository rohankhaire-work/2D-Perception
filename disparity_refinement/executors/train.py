import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.save_path)
            print(f"[INFO] Model checkpoint saved to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                return True
        return False


def train(model, optimizer, epoch, train_loader):

    total_loss_disp = 0

    img_counter = 0

    dt = 0.

    progress_bar = tqdm(enumerate(train_loader),
                        desc=f"Epoch: {epoch}",
                        total=len(train_loader))

    for batch_idx, (input_data, target_data) in progress_bar:
        concat_data = torch.cat((input_data[:]), 1)
        input_data = concat_data.cuda()
        target_data = [ta_data.cuda() for ta_data in target_data]

        # Disp Map Prediction
        st = time.perf_counter()
        output_disp = model(input_data)
        dt += time.perf_counter() - st

        # Target depth
        target = target_data[0]

        # We only compute error on non-zero pixels (i.e. non 1 here)
        error_disp = combined_berhu_smoothl1_loss(output_disp, target)

        # loss
        total_loss_disp += error_disp.item()

        # Optimize the weights!
        error_disp.backward()
        optimizer.step()

        img_counter += len(train_loader)

        progress_bar.set_postfix(loss="%.8f" % float(total_loss_disp / img_counter),
                                 dt="%.2f" % (dt * 1000. / img_counter))

    return total_loss_disp


def test(model, epoch, test_loader):
    model.eval()

    img_counter = 0
    dt = 0.
    d1_cnn = 0.

    min_disp = 0.0
    max_disp = 192.0

    progress_bar = tqdm(enumerate(test_loader),
                        desc=f"Epoch: {epoch}", total=len(test_loader))

    with torch.no_grad():
        for batch_idx, (input_data, target_data) in progress_bar:
            concat_data = torch.cat((input_data[:]), 1)
            input_data = concat_data.cuda()
            target_data = [ta_data.cuda() for ta_data in target_data]

            # Disp Map Prediction
            st = time.perf_counter()

            output_disp = model(input_data)

            dt += time.perf_counter() - st

            # Target depth in normal depth
            target = target_data[0]

            # Get the predicted and targeted depth
            out_disp = output_disp.cpu().detach().numpy()
            tar = target.cpu().numpy()

            nb_img = out_disp.shape[0]
            img_counter += nb_img

            # For each image in the batch
            for i in range(nb_img):

                # Get the GT disp
                #######################
                gt_disp = np.squeeze(tar[i], axis=0)

                gt_h, gt_w = gt_disp.shape

                # Get the Pred depth
                #########################
                pred_disp = np.squeeze(out_disp[i], axis=0)

                # Bound the predicted depth
                pred_disp[pred_disp < min_disp] = min_disp
                pred_disp[pred_disp > max_disp] = max_disp

                # D1-all
                #########################################
                mask_disp = gt_disp > 0
                disp_diff = np.abs(
                    gt_disp[mask_disp] - pred_disp[mask_disp])
                bad_pixels = np.logical_and(
                    disp_diff >= 3, (disp_diff / gt_disp[mask_disp]) >= 0.05)
                d1 = 100.0 * bad_pixels.sum() / mask_disp.sum()

                d1_cnn += d1

            # display TQDM
            progress_bar.set_postfix(D1_all="%.2f" % float(d1_cnn / img_counter),
                                     dt="%.1f" % (dt * 1000. / img_counter))

        d1_cnn /= img_counter

        return d1_cnn


def train_network(model, optimizer, scheduler, epochs, train_loader, test_loader, early_stopping=None):
    for epoch in range(epochs):
        train_loss_disp = train(model, optimizer, epoch, train_loader)

        d1_cnn = test(model, epoch, test_loader)

        scheduler.step()

        lr_val = 0.

        for params in optimizer.param_groups:
            lr_val = params['lr']

            # Early stopping
        if early_stopping and early_stopping(d1_cnn, model):
            print("[INFO] Early stopping triggered.")
            break


def combined_berhu_smoothl1_loss(pred, gt, alpha=0.5):
    # Valid disparity mask
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Smooth L1
    smooth_l1 = F.smooth_l1_loss(pred, gt, reduction='mean')

    # BerHu
    diff = torch.abs(pred - gt)
    c = 0.2 * diff.max().item()
    berhu = torch.where(diff <= c, diff, (diff ** 2 + c ** 2) / (2 * c))
    berhu_loss = berhu.mean()

    # Combine
    total_loss = alpha * smooth_l1 + (1 - alpha) * berhu_loss
    return total_loss
