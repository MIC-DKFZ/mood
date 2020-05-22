import os
import sys
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

from example_algos.data.numpy_dataset import get_numpy2d_dataset, get_numpy3d_dataset
from example_algos.models.aes import AE
from example_algos.util.nifti_io import ni_load, ni_save


class AE3D:
    @monkey_patch_fn_args_as_config
    def __init__(
        self,
        input_shape,
        lr=1e-4,
        n_epochs=20,
        z_dim=512,
        model_feature_map_sizes=(16, 64, 256, 1024),
        load_path=None,
        log_dir=None,
        logger="visdom",
        print_every_iter=100,
        data_dir=None,
    ):

        self.print_every_iter = print_every_iter
        self.n_epochs = n_epochs
        self.batch_size = input_shape[0]
        self.z_dim = z_dim
        self.input_shape = input_shape
        self.logger = logger
        self.data_dir = data_dir

        log_dict = {}
        if logger is not None:
            log_dict = {
                0: (logger),
            }
        self.tx = PytorchExperimentStub(name="ae3d", base_dir=log_dir, config=fn_args_as_config, loggers=log_dict,)

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.model = AE(
            input_size=input_shape[1:],
            z_dim=z_dim,
            fmap_sizes=model_feature_map_sizes,
            conv_op=torch.nn.Conv3d,
            tconv_op=torch.nn.ConvTranspose3d,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if load_path is not None:
            PytorchExperimentLogger.load_model_static(self.model, os.path.join(load_path, "ae_final.pth"))
            time.sleep(5)

    def train(self):

        train_loader = get_numpy3d_dataset(
            base_dir=self.data_dir,
            num_processes=16,
            pin_memory=False,
            batch_size=self.batch_size,
            mode="train",
            target_size=self.input_shape[2],
        )
        val_loader = get_numpy3d_dataset(
            base_dir=self.data_dir,
            num_processes=8,
            pin_memory=False,
            batch_size=self.batch_size,
            mode="val",
            target_size=self.input_shape[2],
        )

        for epoch in range(self.n_epochs):

            ### Train
            self.model.train()

            train_loss = 0
            print("\nStart epoch ", epoch)
            data_loader_ = tqdm(enumerate(train_loader))
            for batch_idx, data in data_loader_:
                inpt = data.to(self.device)

                self.optimizer.zero_grad()
                inpt_rec = self.model(inpt)

                loss = torch.mean(torch.pow(inpt - inpt_rec, 2))
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
                    self.tx.add_result(loss.item(), name="Train-Loss", tag="Losses", counter=cnt)

                    if self.logger is not None:
                        self.tx.l[0].show_image_grid(
                            inpt[:, :, self.input_shape[2] // 2], name="Input", image_args={"normalize": True}
                        )
                        self.tx.l[0].show_image_grid(
                            inpt_rec[:, :, self.input_shape[2] // 2],
                            name="Reconstruction",
                            image_args={"normalize": True},
                        )

            print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.4f}")

            ### Validate
            self.model.eval()

            val_loss = 0
            with torch.no_grad():
                data_loader_ = tqdm(enumerate(val_loader))
                data_loader_.set_description_str("Validating")
                for i, data in data_loader_:
                    inpt = data.to(self.device)
                    inpt_rec = self.model(inpt)

                    loss = torch.mean(torch.pow(inpt - inpt_rec, 2))
                    val_loss += loss.item()

                self.tx.add_result(
                    val_loss / len(val_loader), name="Val-Loss", tag="Losses", counter=(epoch + 1) * len(train_loader)
                )

            print(f"====> Epoch: {epoch} Validation loss: {val_loss / len(val_loader):.4f}")

        self.tx.save_model(self.model, "ae_final")

        time.sleep(10)

    def score_sample(self, np_array):

        orig_shape = np_array.shape
        to_transforms = torch.nn.Upsample((self.input_shape[2], self.input_shape[3], self.input_shape[4]))
        from_transforms = torch.nn.Upsample((orig_shape[0], orig_shape[1], orig_shape[2]))

        data_tensor = torch.from_numpy(np_array).float()
        data_tensor = to_transforms(data_tensor[None][None])

        with torch.no_grad():
            inpt = data_tensor.to(self.device)
            inpt_rec = self.model(inpt)

            loss = torch.mean(torch.pow(inpt - inpt_rec, 2))

        score = loss.cpu().item()

        return score

    def score_pixels(self, np_array):

        orig_shape = np_array.shape
        to_transforms = torch.nn.Upsample((self.input_shape[2], self.input_shape[3], self.input_shape[4]))
        from_transforms = torch.nn.Upsample((orig_shape[0], orig_shape[1], orig_shape[2]))

        data_tensor = torch.from_numpy(np_array).float()
        data_tensor = to_transforms(data_tensor[None][None])

        with torch.no_grad():
            inpt = data_tensor.to(self.device)
            inpt_rec = self.model(inpt)

            loss = torch.pow(inpt - inpt_rec, 2)

        target_tensor = loss.cpu().detach()
        target_tensor = from_transforms(target_tensor)[0][0]

        return target_tensor.detach().numpy()

    def print(self, *args):
        print(*args)
        self.tx.print(*args)


@click.option("-m", "--mode", default="pixel", type=click.Choice(["pixel", "sample"], case_sensitive=False))
@click.option(
    "-r", "--run", default="train", type=click.Choice(["train", "predict", "test", "all"], case_sensitive=False)
)
@click.option("--target-size", type=click.IntRange(1, 512, clamp=True), default=128)
@click.option("--batch-size", type=click.IntRange(1, 512, clamp=True), default=4)
@click.option("--n-epochs", type=int, default=20)
@click.option("--lr", type=float, default=1e-4)
@click.option("--z-dim", type=int, default=128)
@click.option("-fm", "--fmap-sizes", type=int, multiple=True, default=[16, 64, 256, 1024])
@click.option("--print-every-iter", type=int, default=10)
@click.option("-l", "--load-path", type=click.Path(exists=True), required=False, default=None)
@click.option("-o", "--log-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option(
    "--logger", type=click.Choice(["visdom", "tensorboard"], case_sensitive=False), required=False, default=None
)
@click.option("-t", "--test-dir", type=click.Path(exists=True), required=False, default=None)
@click.option("-p", "--pred-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option("-d", "--data-dir", type=click.Path(exists=True), required=True, default=None)
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
    print_every_iter=100,
    load_path=None,
    log_dir=None,
    logger="visdom",
    test_dir=None,
    pred_dir=None,
    data_dir=None,
):

    from scripts.evalresults import eval_dir

    input_shape = (batch_size, 1, target_size, target_size, target_size)

    ae_algo = AE3D(
        input_shape,
        log_dir=log_dir,
        n_epochs=n_epochs,
        lr=lr,
        z_dim=z_dim,
        model_feature_map_sizes=fmap_sizes,
        print_every_iter=print_every_iter,
        load_path=load_path,
        logger=logger,
        data_dir=data_dir,
    )

    if run == "train" or run == "all":
        ae_algo.train()

    if run == "predict" or run == "all":

        if pred_dir is None and log_dir is not None:
            pred_dir = os.path.join(ae_algo.tx.elog.work_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
        elif pred_dir is None and log_dir is None:
            print("Please either give a log/ output dir or a prediction dir")
            exit(0)

        for f_name in os.listdir(test_dir):
            ni_file = os.path.join(test_dir, f_name)
            ni_data, ni_aff = ni_load(ni_file)
            if mode == "pixel":
                pixel_scores = ae_algo.score_pixels(ni_data)
                ni_save(os.path.join(pred_dir, f_name), pixel_scores, ni_aff)
            if mode == "sample":
                sample_score = ae_algo.score_sample(ni_data)
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
