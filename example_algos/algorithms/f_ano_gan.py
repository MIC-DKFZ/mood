import matplotlib

matplotlib.use("Agg", warn=True)

import time
import os
import warnings

import click
import numpy as np
from tqdm import tqdm
import torch
from trixi.logger import PytorchExperimentLogger
from trixi.util.pytorchexperimentstub import PytorchExperimentStub
from trixi.util.config import monkey_patch_fn_args_as_config

from example_algos.data.numpy_dataset import get_numpy2d_dataset
from example_algos.models.iwgan_nets import IWGenerator, IWDiscriminator, IWEncoder, weights_init
from example_algos.util.nifti_io import ni_load, ni_save
from math import ceil


class fAnoGAN:
    @monkey_patch_fn_args_as_config
    def __init__(
        self,
        input_shape,
        lr=1e-4,
        critic_iters=1,
        gen_iters=5,
        n_epochs=10,
        gp_lambda=10,
        z_dim=512,
        print_every_iter=20,
        plot_every_epoch=1,
        log_dir=None,
        load_path=None,
        logger="visdom",
        data_dir=None,
        use_encoder=True,
        enocoder_feature_weight=1e-4,
        encoder_discr_weight=0.0,
    ):

        self.plot_every_epoch = plot_every_epoch
        self.print_every_iter = print_every_iter
        self.gp_lambda = gp_lambda
        self.n_epochs = n_epochs
        self.gen_iters = gen_iters
        self.critic_iters = critic_iters
        self.size = input_shape[2]
        self.batch_size = input_shape[0]
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.logger = logger
        self.data_dir = data_dir
        self.use_encoder = use_encoder
        self.enocoder_feature_weight = enocoder_feature_weight
        self.encoder_discr_weight = encoder_discr_weight

        log_dict = {}
        if logger is not None:
            log_dict = {
                0: (logger),
            }
        self.tx = PytorchExperimentStub(name="fanogan", base_dir=log_dir, config=fn_args_as_config, loggers=log_dict,)

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.n_image_channels = input_shape[1]

        self.gen = IWGenerator(self.size, z_dim=z_dim, n_image_channels=self.n_image_channels)
        self.dis = IWDiscriminator(self.size, n_image_channels=self.n_image_channels)

        self.gen.apply(weights_init)
        self.dis.apply(weights_init)

        self.optimizer_G = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999))

        self.gen = self.gen.to(self.device)
        self.dis = self.dis.to(self.device)

        if self.use_encoder:
            self.enc = IWEncoder(self.size, z_dim=z_dim, n_image_channels=self.n_image_channels)
            self.enc.apply(weights_init)
            self.enc = self.enc.to(self.device)
            self.optimizer_E = torch.optim.Adam(self.enc.parameters(), lr=lr, betas=(0.5, 0.999))

        self.z = torch.randn(self.batch_size, z_dim).to(self.device)

        if load_path is not None:
            PytorchExperimentLogger.load_model_static(self.dis, os.path.join(load_path, "dis_final.pth"))
            PytorchExperimentLogger.load_model_static(self.gen, os.path.join(load_path, "gen_final.pth"))
            if self.use_encoder:
                try:
                    pass
                    # PytorchExperimentLogger.load_model_static(self.enc, os.path.join(load_path, "enc_final.pth"))
                except Exception:
                    warnings.warn("Could not find an Encoder in the directory")
            time.sleep(5)

    def train(self):

        train_loader = get_numpy2d_dataset(
            base_dir=self.data_dir,
            num_processes=16,
            pin_memory=False,
            batch_size=self.batch_size,
            mode="train",
            target_size=self.size,
            slice_offset=10,
        )

        print("Training GAN...")
        for epoch in range(self.n_epochs):
            # for epoch in range(0):

            data_loader_ = tqdm(enumerate(train_loader))
            for i, batch in data_loader_:
                batch = batch * 2 - 1 + torch.randn_like(batch) * 0.01

                real_imgs = batch.to(self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # disc_cost = []
                # w_dist = []
                if i % self.critic_iters == 0:
                    self.optimizer_G.zero_grad()
                    self.optimizer_D.zero_grad()

                    batch_size_curr = real_imgs.shape[0]

                    self.z.normal_()

                    fake_imgs = self.gen(self.z[:batch_size_curr])

                    real_validity = self.dis(real_imgs)
                    fake_validity = self.dis(fake_imgs)

                    gradient_penalty = self.calc_gradient_penalty(
                        self.dis,
                        real_imgs,
                        fake_imgs,
                        batch_size_curr,
                        self.size,
                        self.device,
                        self.gp_lambda,
                        n_image_channels=self.n_image_channels,
                    )

                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.gp_lambda * gradient_penalty
                    d_loss.backward()
                    self.optimizer_D.step()

                    # disc_cost.append(d_loss.item())
                    w_dist = (-torch.mean(real_validity) + torch.mean(fake_validity)).item()

                # -----------------
                #  Train Generator
                # -----------------
                # gen_cost = []
                if i % self.gen_iters == 0:
                    self.optimizer_G.zero_grad()
                    self.optimizer_D.zero_grad()

                    batch_size_curr = self.batch_size

                    fake_imgs = self.gen(self.z)

                    fake_validity = self.dis(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                    # gen_cost.append(g_loss.item())

                if i % self.print_every_iter == 0:
                    status_str = (
                        f"Train Epoch: {epoch} [{i}/{len(train_loader)} "
                        f" ({100.0 * i / len(train_loader):.0f}%)] Dis: "
                        f"{d_loss.item() / batch_size_curr:.6f} vs Gen: "
                        f"{g_loss.item() / batch_size_curr:.6f} (W-Dist: {w_dist / batch_size_curr:.6f})"
                    )
                    data_loader_.set_description_str(status_str)
                    # print(f"[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(train_loader)}]")

                    # print(d_loss.item(), g_loss.item())
                    cnt = epoch * len(train_loader) + i

                    self.tx.add_result(d_loss.item(), name="trainDisCost", tag="DisVsGen", counter=cnt)
                    self.tx.add_result(g_loss.item(), name="trainGenCost", tag="DisVsGen", counter=cnt)
                    self.tx.add_result(w_dist, "wasserstein_distance", counter=cnt)

                    self.tx.l[0].show_image_grid(
                        fake_imgs.reshape(batch_size_curr, self.n_image_channels, self.size, self.size),
                        "GeneratedImages",
                        image_args={"normalize": True},
                    )

        self.tx.save_model(self.dis, "dis_final")
        self.tx.save_model(self.gen, "gen_final")

        self.gen.train(True)
        self.dis.train(True)

        if not self.use_encoder:
            time.sleep(10)
            return

        weight_features = self.enocoder_feature_weight
        weight_disc = self.encoder_discr_weight
        print("Training Encoder...")
        for epoch in range(self.n_epochs // 2):
            data_loader_ = tqdm(enumerate(train_loader))
            for i, batch in data_loader_:
                batch = batch * 2 - 1 + torch.randn_like(batch) * 0.01
                real_img = batch.to(self.device)
                batch_size_curr = real_img.shape[0]

                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_E.zero_grad()

                z = self.enc(real_img)
                recon_img = self.gen(z)

                _, img_feats = self.dis.forward_last_feature(real_img)
                disc_loss, recon_feats = self.dis.forward_last_feature(recon_img)

                recon_img = recon_img.reshape(batch_size_curr, self.n_image_channels, self.size, self.size)
                loss_img = self.mse(real_img, recon_img)
                loss_feat = self.mse(img_feats, recon_feats) * weight_features
                disc_loss = -torch.mean(disc_loss) * weight_disc

                loss = loss_img + loss_feat + disc_loss

                loss.backward()
                self.optimizer_E.step()

                if i % self.print_every_iter == 0:
                    status_str = (
                        f"[Epoch {epoch}/{self.n_epochs // 2}] [Batch {i}/{len(train_loader)}] Loss:{loss:.06f}"
                    )
                    data_loader_.set_description_str(status_str)

                    cnt = epoch * len(train_loader) + i
                    self.tx.add_result(loss.item(), name="EncoderLoss", counter=cnt)

                    self.tx.l[0].show_image_grid(
                        real_img.reshape(batch_size_curr, self.n_image_channels, self.size, self.size),
                        "RealImages",
                        image_args={"normalize": True},
                    )
                    self.tx.l[0].show_image_grid(
                        recon_img.reshape(batch_size_curr, self.n_image_channels, self.size, self.size),
                        "ReconImages",
                        image_args={"normalize": True},
                    )

        self.tx.save_model(self.enc, "enc_final")
        self.enc.train(False)

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
            real_imgs = batch.to(self.device)
            batch_size_curr = real_imgs.shape[0]

            if self.use_encoder:
                z = self.enc(real_imgs)
            else:
                z = self.backprop_to_nearest_z(real_imgs)

            pseudo_img_recon = self.gen(z)

            pseudo_img_recon = pseudo_img_recon.reshape(batch_size_curr, self.n_image_channels, self.size, self.size)
            img_diff = torch.mean(torch.abs(pseudo_img_recon - real_imgs), dim=1, keepdim=True)

            loss = torch.sum(img_diff, dim=(1, 2, 3)).detach()

            slice_scores += loss.cpu().tolist()

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
            real_imgs = batch.to(self.device)
            batch_size_curr = real_imgs.shape[0]

            if self.use_encoder:
                z = self.enc(real_imgs)
            else:
                z = self.backprop_to_nearest_z(real_imgs)

            pseudo_img_recon = self.gen(z)

            pseudo_img_recon = pseudo_img_recon.reshape(batch_size_curr, self.n_image_channels, self.size, self.size)
            img_diff = torch.mean(torch.abs(pseudo_img_recon - real_imgs), dim=1, keepdim=True)

            loss = img_diff[:, 0, :]
            target_tensor[i * self.batch_size : (i + 1) * self.batch_size] = loss.cpu()

        target_tensor = from_transforms(target_tensor[None])[0]

        return target_tensor.detach().numpy()

    def backprop_to_nearest_z(self, real_imgs):

        batch_size_curr = real_imgs.shape[0]

        z = torch.randn(batch_size_curr, self.z_dim).to(self.device).normal_()
        z.requires_grad = True
        # optimizer_z = torch.optim.LBFGS([z], lr=0.02)
        optimizer_z = torch.optim.Adam([z], lr=0.002)
        # optimizer_z = torch.optim.RMSprop([z], lr=0.05)

        for i in range(200):

            def closure():
                self.gen.zero_grad()
                optimizer_z.zero_grad()

                pseudo_img_recon = self.gen(z)

                _, img_feats = self.dis.forward_last_feature(real_imgs)
                disc_loss, recon_feats = self.dis.forward_last_feature(pseudo_img_recon)

                pseudo_img_recon = pseudo_img_recon.reshape(
                    batch_size_curr, self.n_image_channels, self.size, self.size
                )
                disc_loss = torch.mean(disc_loss)

                imgs_diff = torch.mean(torch.abs(pseudo_img_recon - real_imgs))
                feats_diff = torch.mean(torch.abs(img_feats - recon_feats))
                loss = imgs_diff - disc_loss * 0.001  # + feats_diff

                loss.backward()

                return loss

            optimizer_z.step(closure)

        return z.detach()

    def score(self, batch):
        real_imgs = batch.to(self.device).float()

        z = self.enc(real_imgs)

        batch_size_curr = real_imgs.shape[0]

        # z = torch.randn(batch_size_curr, self.z_dim).to(self.device).normal_()
        # z.requires_grad = True
        # # optimizer_z = torch.optim.LBFGS([z], lr=0.02)
        # optimizer_z = torch.optim.Adam([z], lr=0.002)
        # # optimizer_z = torch.optim.RMSprop([z], lr=0.05)
        #
        # cn = dict(tr=0)
        #
        # self.tx.vlog.show_image_grid(real_imgs, "RealImages",
        #                              image_args={"normalize": True})
        #
        # for i in range(200):
        #     def closure():
        #         self.gen.zero_grad()
        #         optimizer_z.zero_grad()
        #
        #         pseudo_img_recon = self.gen(z)
        #
        #         _, img_feats = self.dis.forward_last_feature(real_imgs)
        #         disc_loss, recon_feats = self.dis.forward_last_feature(pseudo_img_recon)
        #
        #         pseudo_img_recon = pseudo_img_recon.reshape(batch_size_curr, self.n_image_channels, self.size, self.size)
        #         disc_loss = torch.mean(disc_loss)
        #
        #         imgs_diff = torch.mean(torch.abs(pseudo_img_recon - real_imgs))
        #         feats_diff = torch.mean(torch.abs(img_feats - recon_feats))
        #         loss = imgs_diff - disc_loss * 0.001  # + feats_diff
        #
        #         loss.backward()
        #         # optimizer_z.step()
        #         #
        #         # if cn['tr'] % 20 == 0:
        #         # pseudo_img_recon = pseudo_img_recon.clamp(-1.5, 1.5)
        #         self.tx.vlog.show_image_grid(pseudo_img_recon, "PseudoImages",
        #                                      image_args={"normalize": True})
        #         self.tx.vlog.show_image_grid(torch.mean(torch.abs(pseudo_img_recon - real_imgs), dim=1, keepdim=True),
        #                                      "DiffImages", image_args={"normalize": True})
        #         #
        #         # tx.add_result(disc_loss.item() * 0.001, name="DiscLoss", tag="AnoIter")
        #         # tx.add_result(imgs_diff.item(), name="ImgsDiff", tag="AnoIter")
        #         # tx.add_result(torch.mean(torch.pow(z, 2)).item(), name="ZDevi", tag="AnoIter")
        #         #
        #         # cn['tr'] += 1
        #
        #         return loss
        #
        #     optimizer_z.step(closure)
        #
        #     # time.sleep(1)
        #
        #     print(i)
        #
        pseudo_img_recon = self.gen(z)

        pseudo_img_recon = pseudo_img_recon.reshape(batch_size_curr, self.n_image_channels, self.size, self.size)
        img_diff = torch.mean(torch.abs(pseudo_img_recon - real_imgs), dim=1, keepdim=True)

        img_scores = torch.sum(img_diff, dim=(1, 2, 3)).detach().tolist()
        pixel_scores = img_diff.flatten().detach().tolist()

        self.tx.vlog.show_image_grid(pseudo_img_recon, "PseudoImages", image_args={"normalize": True})
        self.tx.vlog.show_image_grid(
            torch.mean(torch.abs(pseudo_img_recon - real_imgs), dim=1, keepdim=True),
            "DiffImages",
            image_args={"normalize": True},
        )

        # print("One Down")

        return img_scores, pixel_scores

    @staticmethod
    def mse(x, y):
        return torch.mean(torch.pow(x - y, 2))

    @staticmethod
    def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, device, gp_lambda, n_image_channels=3):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, n_image_channels, dim, dim)
        alpha = alpha.to(device)

        fake_data = fake_data.view(batch_size, n_image_channels, dim, dim)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

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
@click.option("--batch-size", type=click.IntRange(1, 512, clamp=True), default=32)
@click.option("--n-epochs", type=int, default=20)
@click.option("--lr", type=float, default=1e-4)
@click.option("--critic-iters", type=int, default=1)
@click.option("--gen-iters", type=int, default=5)
@click.option("--gp-lambda", type=float, default=10.0)
@click.option("--z-dim", type=int, default=128)
@click.option("--plot-every-epoch", type=int, default=1)
@click.option("--print-every-iter", type=int, default=100)
@click.option("-l", "--load-path", type=click.Path(exists=True), required=False, default=None)
@click.option("-o", "--log-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option(
    "--logger", type=click.Choice(["visdom", "tensorboard"], case_sensitive=False), required=False, default="visdom"
)
@click.option("-t", "--test-dir", type=click.Path(exists=True), required=False, default=None)
@click.option("-p", "--pred-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option("-d", "--data-dir", type=click.Path(exists=True), required=True, default=None)
@click.option("--use-encoder/--no-encoder", type=bool, is_flag=True, default=True)
@click.option("--enocoder-feature-weight", type=float, default=1e-4)
@click.option("--encoder-discr-weight", type=float, default=0.0)
@click.command()
def main(
    mode="pixel",
    run="train",
    target_size=128,
    batch_size=16,
    n_epochs=20,
    lr=1e-4,
    critic_iters=1,
    gen_iters=5,
    gp_lambda=10,
    z_dim=128,
    plot_every_epoch=1,
    print_every_iter=100,
    load_path=None,
    log_dir=None,
    logger="visdom",
    test_dir=None,
    pred_dir=None,
    data_dir=None,
    use_encoder=True,
    enocoder_feature_weight=1e-4,
    encoder_discr_weight=0.0,
):

    from scripts.evalresults import eval_dir

    input_shape = (batch_size, 1, target_size, target_size)

    anogan_algo = fAnoGAN(
        input_shape=input_shape,
        lr=lr,
        critic_iters=critic_iters,
        gen_iters=gen_iters,
        n_epochs=n_epochs,
        gp_lambda=gp_lambda,
        z_dim=z_dim,
        print_every_iter=print_every_iter,
        plot_every_epoch=plot_every_epoch,
        log_dir=log_dir,
        load_path=load_path,
        logger=logger,
        data_dir=data_dir,
        use_encoder=use_encoder,
        enocoder_feature_weight=enocoder_feature_weight,
        encoder_discr_weight=encoder_discr_weight,
    )

    if run == "train" or run == "all":
        anogan_algo.train()

    if run == "predict" or run == "all":

        if pred_dir is None and log_dir is not None:
            pred_dir = os.path.join(anogan_algo.tx.elog.work_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
        elif pred_dir is None and log_dir is None:
            print("Please either give a log/ output dir or a prediction dir")
            exit(0)

        for f_name in os.listdir(test_dir):
            ni_file = os.path.join(test_dir, f_name)
            ni_data, ni_aff = ni_load(ni_file)
            if mode == "pixel":
                pixel_scores = anogan_algo.score_pixels(ni_data)
                ni_save(os.path.join(pred_dir, f_name), pixel_scores, ni_aff)
            if mode == "sample":
                sample_score = anogan_algo.score_sample(ni_data)
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
