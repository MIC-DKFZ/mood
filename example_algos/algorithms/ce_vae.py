import os
import time
from math import ceil

import click
import numpy as np
import torch
import torch.distributions as dist
from torch import optim
from tqdm import tqdm
from trixi.logger import PytorchExperimentLogger
from trixi.util.config import monkey_patch_fn_args_as_config
from trixi.util.pytorchexperimentstub import PytorchExperimentStub
from trixi.util.pytorchutils import get_smooth_image_gradient

from example_algos.data.numpy_dataset import get_numpy2d_dataset
from example_algos.models.aes import VAE
from example_algos.util.ce_noise import get_square_mask, normalize, smooth_tensor
from example_algos.util.nifti_io import ni_load, ni_save


class ceVAE:
    @monkey_patch_fn_args_as_config
    def __init__(
        self,
        input_shape,
        lr=1e-4,
        n_epochs=20,
        z_dim=512,
        model_feature_map_sizes=(16, 64, 256, 1024),
        use_geco=False,
        beta=0.01,
        ce_factor=0.5,
        score_mode="combi",
        load_path=None,
        log_dir=None,
        logger="visdom",
        print_every_iter=100,
        data_dir=None,
    ):

        self.score_mode = score_mode
        self.ce_factor = ce_factor
        self.beta = beta
        self.print_every_iter = print_every_iter
        self.n_epochs = n_epochs
        self.batch_size = input_shape[0]
        self.z_dim = z_dim
        self.use_geco = use_geco
        self.input_shape = input_shape
        self.logger = logger
        self.data_dir = data_dir

        log_dict = {}
        if logger is not None:
            log_dict = {
                0: (logger),
            }
        self.tx = PytorchExperimentStub(name="cevae", base_dir=log_dir, config=fn_args_as_config, loggers=log_dict,)

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.model = VAE(input_size=input_shape[1:], z_dim=z_dim, fmap_sizes=model_feature_map_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.vae_loss_ema = 1
        self.theta = 1

        if load_path is not None:
            PytorchExperimentLogger.load_model_static(self.model, os.path.join(load_path, "vae_final.pth"))
            time.sleep(5)

    def train(self):

        train_loader = get_numpy2d_dataset(
            base_dir=self.data_dir,
            num_processes=16,
            pin_memory=False,
            batch_size=self.batch_size,
            mode="train",
            target_size=self.input_shape[2],
        )
        val_loader = get_numpy2d_dataset(
            base_dir=self.data_dir,
            num_processes=8,
            pin_memory=False,
            batch_size=self.batch_size,
            mode="val",
            target_size=self.input_shape[2],
        )

        for epoch in range(self.n_epochs):

            self.model.train()
            train_loss = 0

            print("Start epoch")
            data_loader_ = tqdm(enumerate(train_loader))
            for batch_idx, data in data_loader_:
                data = data * 2 - 1
                self.optimizer.zero_grad()

                inpt = data.to(self.device)

                ### VAE Part
                loss_vae = 0
                if self.ce_factor < 1:
                    x_rec_vae, z_dist, = self.model(inpt)

                    kl_loss = 0
                    if self.beta > 0:
                        kl_loss = self.kl_loss_fn(z_dist) * self.beta
                    rec_loss_vae = self.rec_loss_fn(x_rec_vae, inpt)
                    loss_vae = kl_loss + rec_loss_vae * self.theta

                ### CE Part
                loss_ce = 0
                if self.ce_factor > 0:

                    ce_tensor = get_square_mask(
                        data.shape,
                        square_size=(0, np.max(self.input_shape[2:]) // 2),
                        noise_val=(torch.min(data).item(), torch.max(data).item()),
                        n_squares=(0, 3),
                    )
                    ce_tensor = torch.from_numpy(ce_tensor).float()
                    inpt_noisy = torch.where(ce_tensor != 0, ce_tensor, data)

                    inpt_noisy = inpt_noisy.to(self.device)
                    x_rec_ce, _ = self.model(inpt_noisy)
                    rec_loss_ce = self.rec_loss_fn(x_rec_ce, inpt)
                    loss_ce = rec_loss_ce

                loss = (1.0 - self.ce_factor) * loss_vae + self.ce_factor * loss_ce

                if self.use_geco and self.ce_factor < 1:
                    g_goal = 0.1
                    g_lr = 1e-4
                    self.vae_loss_ema = (1.0 - 0.9) * rec_loss_vae + 0.9 * self.vae_loss_ema
                    self.theta = self.geco_beta_update(self.theta, self.vae_loss_ema, g_goal, g_lr, speedup=2)

                if torch.isnan(loss):
                    print("A wild NaN occurred")
                    continue

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if batch_idx % self.print_every_iter == 0:
                    status_str = (
                        f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                        f" ({100.0 * batch_idx / len(train_loader):.0f}%)] Loss: "
                        f"{loss.item() / len(inpt):.6f}"
                    )
                    data_loader_.set_description_str(status_str)

                    cnt = epoch * len(train_loader) + batch_idx

                    if self.ce_factor < 1:
                        self.tx.l[0].show_image_grid(inpt, name="Input-VAE", image_args={"normalize": True})
                        self.tx.l[0].show_image_grid(x_rec_vae, name="Output-VAE", image_args={"normalize": True})

                        if self.beta > 0:
                            self.tx.add_result(torch.mean(kl_loss).item(), name="Kl-loss", tag="Losses", counter=cnt)
                        self.tx.add_result(torch.mean(rec_loss_vae).item(), name="Rec-loss", tag="Losses", counter=cnt)
                        self.tx.add_result(loss_vae.item(), name="Train-loss", tag="Losses", counter=cnt)

                    if self.ce_factor > 0:
                        self.tx.l[0].show_image_grid(inpt_noisy, name="Input-CE", image_args={"normalize": True})
                        self.tx.l[0].show_image_grid(x_rec_ce, name="Output-CE", image_args={"normalize": True})

            print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.4f}")

            self.model.eval()

            val_loss = 0
            with torch.no_grad():
                data_loader_ = tqdm(enumerate(val_loader))
                for i, data in data_loader_:
                    data = data * 2 - 1
                    inpt = data.to(self.device)

                    x_rec, z_dist = self.model(inpt, sample=False)

                    kl_loss = 0
                    if self.beta > 0:
                        kl_loss = self.kl_loss_fn(z_dist) * self.beta
                    rec_loss = self.rec_loss_fn(x_rec, inpt)
                    loss = kl_loss + rec_loss * self.theta

                    val_loss += loss.item()

                self.tx.add_result(
                    val_loss / len(val_loader), name="Val-Loss", tag="Losses", counter=(epoch + 1) * len(train_loader)
                )

            print(f"====> Epoch: {epoch} Validation loss: {val_loss / len(val_loader):.4f}")

        self.tx.save_model(self.model, "vae_final")

        time.sleep(10)

    def score_sample(self, np_array):

        orig_shape = np_array.shape
        to_transforms = torch.nn.Upsample((self.input_shape[2], self.input_shape[3]), mode="bilinear")

        data_tensor = torch.from_numpy(np_array).float()
        data_tensor = to_transforms(data_tensor[None])[0]
        slice_scores = []

        for i in range(ceil(orig_shape[0] / self.batch_size)):
            batch = data_tensor[i * self.batch_size : (i + 1) * self.batch_size].unsqueeze(1)
            batch = batch * 2 - 1

            with torch.no_grad():
                inpt = batch.to(self.device).float()
                x_rec, z_dist = self.model(inpt, sample=False)
                kl_loss = self.kl_loss_fn(z_dist, sum_samples=False)
                rec_loss = self.rec_loss_fn(x_rec, inpt, sum_samples=False)
                img_scores = kl_loss * self.beta + rec_loss * self.theta

            slice_scores += img_scores.cpu().tolist()

        return np.max(slice_scores)

    def score_pixels(self, np_array):

        orig_shape = np_array.shape
        to_transforms = torch.nn.Upsample((self.input_shape[2], self.input_shape[3]), mode="bilinear")
        from_transforms = torch.nn.Upsample((orig_shape[1], orig_shape[2]), mode="bilinear")

        data_tensor = torch.from_numpy(np_array).float()
        data_tensor = to_transforms(data_tensor[None])[0]
        target_tensor = torch.zeros_like(data_tensor)

        for i in range(ceil(orig_shape[0] / self.batch_size)):
            batch = data_tensor[i * self.batch_size : (i + 1) * self.batch_size].unsqueeze(1)
            batch = batch * 2 - 1

            inpt = batch.to(self.device).float()
            x_rec, z_dist = self.model(inpt, sample=False)

            if self.score_mode == "combi":

                rec = torch.pow((x_rec - inpt), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)

                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    loss = self.kl_loss_fn(z_d)
                    return loss

                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=inpt, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(loss_grad_kl, dim=1, keepdim=True)

                pixel_scores = smooth_tensor(normalize(loss_grad_kl), kernel_size=8) * rec

            elif self.score_mode == "rec":

                rec = torch.pow((x_rec - inpt), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)
                pixel_scores = rec

            elif self.score_mode == "grad":

                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    kl_loss_ = self.kl_loss_fn(z_d)
                    rec_loss_ = self.rec_loss_fn(x_r, x)
                    loss_ = kl_loss_ * self.beta + rec_loss_ * self.theta
                    return torch.mean(loss_)

                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=inpt, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(loss_grad_kl, dim=1, keepdim=True)

                pixel_scores = smooth_tensor(normalize(loss_grad_kl), kernel_size=8)

            self.tx.elog.show_image_grid(inpt, name="Input", image_args={"normalize": True}, n_iter=i)
            self.tx.elog.show_image_grid(x_rec, name="Output", image_args={"normalize": True}, n_iter=i)
            self.tx.elog.show_image_grid(pixel_scores, name="Scores", image_args={"normalize": True}, n_iter=i)

            target_tensor[i * self.batch_size : (i + 1) * self.batch_size] = pixel_scores.detach().cpu()[:, 0, :]

        target_tensor = from_transforms(target_tensor[None])[0]

        return target_tensor.detach().numpy()

    @staticmethod
    def load_trained_model(model, tx, path):
        tx.elog.load_model_static(model=model, model_file=path)

    @staticmethod
    def kl_loss_fn(z_post, sum_samples=True, correct=False):
        z_prior = dist.Normal(0, 1.0)
        kl_div = dist.kl_divergence(z_post, z_prior)
        if correct:
            kl_div = torch.sum(kl_div, dim=(1, 2, 3))
        else:
            kl_div = torch.mean(kl_div, dim=(1, 2, 3))
        if sum_samples:
            return torch.mean(kl_div)
        else:
            return kl_div

    @staticmethod
    def rec_loss_fn(recon_x, x, sum_samples=True, correct=False):
        if correct:
            x_dist = dist.Laplace(recon_x, 1.0)
            log_p_x_z = x_dist.log_prob(x)
            log_p_x_z = torch.sum(log_p_x_z, dim=(1, 2, 3))
        else:
            log_p_x_z = -torch.abs(recon_x - x)
            log_p_x_z = torch.mean(log_p_x_z, dim=(1, 2, 3))
        if sum_samples:
            return -torch.mean(log_p_x_z)
        else:
            return -log_p_x_z

    @staticmethod
    def get_inpt_grad(model, inpt, err_fn):
        model.zero_grad()
        inpt = inpt.detach()
        inpt.requires_grad = True

        err = err_fn(inpt)
        err.backward()

        grad = inpt.grad.detach()

        model.zero_grad()

        return torch.abs(grad.detach())

    @staticmethod
    def geco_beta_update(beta, error_ema, goal, step_size, min_clamp=1e-10, max_clamp=1e4, speedup=None):
        constraint = (error_ema - goal).detach()
        if speedup is not None and constraint > 0.0:
            beta = beta * torch.exp(speedup * step_size * constraint)
        else:
            beta = beta * torch.exp(step_size * constraint)
        if min_clamp is not None:
            beta = np.max((beta.item(), min_clamp))
        if max_clamp is not None:
            beta = np.min((beta.item(), max_clamp))
        return beta

    @staticmethod
    def get_ema(new, old, alpha):
        if old is None:
            return new
        return (1.0 - alpha) * new + alpha * old

    def print(self, *args):
        print(*args)
        self.tx.print(*args)

    def log_result(self, val, key=None):
        self.tx.print(key, val)
        self.tx.add_result_without_epoch(val, key)


@click.option("-m", "--mode", default="pixel", type=click.Choice(["pixel", "sample"], case_sensitive=False))
@click.option(
    "-r", "--run", default="train", type=click.Choice(["train", "predict", "test", "all"], case_sensitive=False)
)
@click.option("--target-size", type=click.IntRange(1, 512, clamp=True), default=128)
@click.option("--batch-size", type=click.IntRange(1, 512, clamp=True), default=16)
@click.option("--n-epochs", type=int, default=20)
@click.option("--lr", type=float, default=1e-4)
@click.option("--z-dim", type=int, default=128)
@click.option("-fm", "--fmap-sizes", type=int, multiple=True, default=[16, 64, 256, 1024])
@click.option("--print-every-iter", type=int, default=100)
@click.option("-l", "--load-path", type=click.Path(exists=True), required=False, default=None)
@click.option("-o", "--log-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option(
    "--logger", type=click.Choice(["visdom", "tensorboard"], case_sensitive=False), required=False, default="visdom"
)
@click.option("-t", "--test-dir", type=click.Path(exists=True), required=False, default=None)
@click.option("-p", "--pred-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option("-d", "--data-dir", type=click.Path(exists=True), required=True, default=None)
@click.option("--use-geco", type=bool, is_flag=True, default=False)
@click.option("--beta", type=float, default=0.01)
@click.option("--ce-factor", type=click.FloatRange(0.0, 1.0, clamp=True), default=0.5)
@click.option(
    "--score-mode", type=click.Choice(["rec", "grad", "combi"], case_sensitive=False), required=False, default="rec"
)
@click.command()
def main(
    mode="pixel",
    run="train",
    target_size=128,
    batch_size=16,
    n_epochs=20,
    lr=1e-4,
    z_dim=128,
    fmap_sizes=(16, 64, 256, 1024),
    use_geco=False,
    beta=0.01,
    ce_factor=0.5,
    score_mode="rec",
    print_every_iter=100,
    load_path=None,
    log_dir=None,
    logger="visdom",
    test_dir=None,
    pred_dir=None,
    data_dir=None,
):

    from scripts.evalresults import eval_dir

    input_shape = (batch_size, 1, target_size, target_size)

    cevae_algo = ceVAE(
        input_shape,
        log_dir=log_dir,
        n_epochs=n_epochs,
        lr=lr,
        z_dim=z_dim,
        model_feature_map_sizes=fmap_sizes,
        use_geco=use_geco,
        beta=beta,
        ce_factor=ce_factor,
        score_mode=score_mode,
        print_every_iter=print_every_iter,
        load_path=load_path,
        logger=logger,
        data_dir=data_dir,
    )

    if run == "train" or run == "all":
        cevae_algo.train()

    if run == "predict" or run == "all":

        if pred_dir is None and log_dir is not None:
            pred_dir = os.path.join(cevae_algo.tx.elog.work_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
        elif pred_dir is None and log_dir is None:
            print("Please either give a log/ output dir or a prediction dir")
            exit(0)

        for f_name in os.listdir(test_dir):
            ni_file = os.path.join(test_dir, f_name)
            ni_data, ni_aff = ni_load(ni_file)
            if mode == "pixel":
                pixel_scores = cevae_algo.score_pixels(ni_data)
                ni_save(os.path.join(pred_dir, f_name), pixel_scores, ni_aff)
            if mode == "sample":
                sample_score = cevae_algo.score_sample(ni_data)
                with open(os.path.join(pred_dir, f_name + ".txt"), "w") as target_file:
                    target_file.write(str(sample_score))

    if run == "test" or run == "all":

        if pred_dir is None:
            print("Please either give a prediction dir")
            exit(0)
        if test_dir is None:
            print(
                "Please either give a test dir which contains the test samples "
                "and for which a test_dir_label folder exists"
            )
            exit(0)

        test_dir = test_dir[:-1] if test_dir.endswith("/") else test_dir
        score = eval_dir(pred_dir=pred_dir, label_dir=test_dir + f"_label/{mode}", mode=mode)

        print(score)


if __name__ == "__main__":

    main()
