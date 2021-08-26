'''Taken from https://github.com/Adi-iitd/AI-Art/blob/master/src/CycleGAN/CycleGAN-PL.py .'''
from __future__ import annotations
import itertools
import logging
from typing import Iterator, NamedTuple

from bolts.structures import Stage
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.model.model_utils import index_by_s, to_discrete

logger = logging.getLogger(__name__)


class SharedStepOut(NamedTuple):
    x: Tensor
    s: Tensor
    recon: Tensor
    cf_pred: Tensor
    recons_0: Tensor
    recons_1: Tensor


class Initializer:
    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Initializes the weight of the network!
        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """

        self.init_type = init_type
        self.init_gain = init_gain

    def init_module(self, m: nn.Module) -> None:

        cls_name = m.__class__.__name__
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)  # type: ignore[arg-type]
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)  # type: ignore[arg-type]
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)  # type: ignore[arg-type]

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)  # type: ignore[arg-type]
            nn.init.constant_(m.bias.data, val=0)  # type: ignore[arg-type]

    def __call__(self, net: nn.Module) -> nn.Module:

        """
        Parameters:
            net: Network
        """

        net.apply(self.init_module)

        return net


class HistoryPool:

    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """

    def __init__(self, pool_sz: int = 50):

        """
        Parameters:
            pool_sz: Size of the image buffer
        """

        self.nb_samples = 0
        self.history_pool: list[Tensor] = []
        self.pool_sz = pool_sz

    def push_and_pop(self, samples: Tensor) -> Tensor:

        """
        Parameters:
            samples: latest images generated by the generator
        Returns a batch of images from pool!
        """

        samples_to_return = []
        for sample in samples:
            sample = torch.unsqueeze(sample, 0)

            if self.nb_samples < self.pool_sz:
                self.history_pool.append(sample)
                samples_to_return.append(sample)
                self.nb_samples += 1
            elif np.random.uniform(0, 1) > 0.5:

                rand_int = np.random.randint(0, self.pool_sz)
                temp_img = self.history_pool[rand_int].clone()
                self.history_pool[rand_int] = sample
                samples_to_return.append(temp_img)
            else:
                samples_to_return.append(sample)

        return torch.cat(samples_to_return, 0)


class Loss:

    """
    This class implements different losses required to train the generators and discriminators of CycleGAN
    """

    def __init__(
        self,
        loss_type: str = 'MSE',
        lambda_: int = 10,
        feature_groups: dict[str, list[slice]] | None = None,
    ):

        """
        Parameters:
            loss_type: Loss Function to train CycleGAN
            lambda_:   Weightage of Cycle-consistency loss
        """

        self._loss_fn = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
        self._recon_loss_fn = nn.L1Loss(reduction="mean")  # TODO: Make this MSE
        self.lambda_ = lambda_
        self.feature_groups = feature_groups if feature_groups is not None else {}

    def get_dis_loss(self, dis_pred_real_data: Tensor, dis_pred_fake_data: Tensor) -> Tensor:

        """
        Parameters:
            dis_pred_real_data: Discriminator's prediction on real data
            dis_pred_fake_data: Discriminator's prediction on fake data
        """

        dis_tar_real_data = torch.ones_like(dis_pred_real_data, requires_grad=False)
        dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad=False)

        loss_real_data = self._loss_fn(dis_pred_real_data, dis_tar_real_data)
        loss_fake_data = self._loss_fn(dis_pred_fake_data, dis_tar_fake_data)

        return (loss_real_data + loss_fake_data) * 0.5

    def get_gen_gan_loss(self, dis_pred_fake_data: Tensor) -> Tensor:

        """
        Parameters:
            dis_pred_fake_data: Discriminator's prediction on fake data
        """

        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad=False)
        return self._loss_fn(dis_pred_fake_data, gen_tar_fake_data)

    def get_gen_cyc_loss(self, real_data: Tensor, cyc_data: Tensor) -> Tensor:

        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            cyc_data:  Image reconstructed after passing the real image through both the generators
                       X_recons = F * G (X_real), where F and G are the two generators
        """

        if self.feature_groups["discrete"]:
            gen_cyc_loss = real_data.new_tensor(0.0)
            for i in range(
                real_data[
                    :, slice(self.feature_groups["discrete"][-1].stop, real_data.shape[1])
                ].shape[1]
            ):
                gen_cyc_loss += self._recon_loss_fn(
                    cyc_data[
                        :, slice(self.feature_groups["discrete"][-1].stop, real_data.shape[1])
                    ][:, i].sigmoid(),
                    real_data[
                        :, slice(self.feature_groups["discrete"][-1].stop, real_data.shape[1])
                    ][:, i],
                )
            for group_slice in self.feature_groups["discrete"]:
                gen_cyc_loss += nn.functional.cross_entropy(
                    cyc_data[:, group_slice],
                    torch.argmax(real_data[:, group_slice], dim=-1),
                )
        else:
            gen_cyc_loss = self._recon_loss_fn(cyc_data.sigmoid(), real_data)

        return gen_cyc_loss * self.lambda_

    def get_gen_idt_loss(self, real_data: Tensor, idt_data: Tensor) -> Tensor:

        """
        Implements the identity loss:
            nn.L1Loss(LG_B2A(real_A), real_A)
            nn.L1Loss(LG_A2B(real_B), real_B)
        """

        if self.feature_groups["discrete"]:
            gen_idt_loss = real_data.new_tensor(0.0)
            for i in range(
                real_data[
                    :, slice(self.feature_groups["discrete"][-1].stop, real_data.shape[1])
                ].shape[1]
            ):
                gen_idt_loss += self._recon_loss_fn(
                    idt_data[
                        :, slice(self.feature_groups["discrete"][-1].stop, real_data.shape[1])
                    ][:, i].sigmoid(),
                    real_data[
                        :, slice(self.feature_groups["discrete"][-1].stop, real_data.shape[1])
                    ][:, i],
                )
            for group_slice in self.feature_groups["discrete"]:
                gen_idt_loss += nn.functional.cross_entropy(
                    idt_data[:, group_slice],
                    torch.argmax(real_data[:, group_slice], dim=-1),
                )
        else:
            gen_idt_loss = self._recon_loss_fn(idt_data.sigmoid(), real_data)

        return gen_idt_loss * self.lambda_ * 0.5

    def get_gen_loss(
        self,
        real_a: Tensor,
        real_b: Tensor,
        gen_fwd: GenFwd,
        d_a_pred_fake_data: Tensor,
        d_b_pred_fake_data: Tensor,
    ) -> GenLoss:

        """
        Implements the total Generator loss
        Sum of Cycle loss, Identity loss, and GAN loss
        """

        # Cycle loss
        cyc_loss_a = self.get_gen_cyc_loss(real_a, gen_fwd.cyc_a)
        cyc_loss_b = self.get_gen_cyc_loss(real_b, gen_fwd.cyc_b)
        tot_cyc_loss = cyc_loss_a + cyc_loss_b

        # GAN loss
        g_a2b_gan_loss = self.get_gen_gan_loss(d_b_pred_fake_data)
        g_b2a_gan_loss = self.get_gen_gan_loss(d_a_pred_fake_data)

        # Identity loss
        g_b2a_idt_loss = self.get_gen_idt_loss(real_a, gen_fwd.idt_a)
        g_a2b_idt_loss = self.get_gen_idt_loss(real_b, gen_fwd.idt_b)

        # Total individual losses
        g_a2b_loss = g_a2b_gan_loss + g_a2b_idt_loss + tot_cyc_loss
        g_b2a_loss = g_b2a_gan_loss + g_b2a_idt_loss + tot_cyc_loss
        g_tot_loss = g_a2b_loss + g_b2a_loss - tot_cyc_loss

        return GenLoss(a2b=g_a2b_loss, b2a=g_b2a_loss, tot=g_tot_loss)


class GenLoss(NamedTuple):
    a2b: Tensor
    b2a: Tensor
    tot: Tensor


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, apply_dp: bool = True):

        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        """
        Parameters:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        conv = nn.Linear(in_channels, in_channels)
        layers = [conv, nn.ReLU(True)]

        if apply_dp:
            layers += [nn.Dropout(0.5)]

        conv = nn.Linear(in_channels, in_channels)
        layers += [conv]

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class Generator(nn.Module):
    net: nn.Module

    def __init__(self, in_dims: int, nb_resblks: int = 3):

        """
                                Generator Architecture (Image Size: 256)
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3,
        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """

        """
        Parameters:
            in_channels:  Number of input channels
            out_channels: Number of output channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()
        out_dims = in_dims * 3
        conv = nn.Linear(in_dims, out_dims)
        layers: list[nn.Module] = [conv]

        for _ in range(nb_resblks):
            res_blk = ResBlock(in_channels=out_dims, apply_dp=False)
            layers += [res_blk]

        conv = nn.Linear(out_dims, in_dims)
        layers += [conv]

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    net: nn.Module

    def __init__(self, in_dims: int, nb_layers: int = 3):

        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """

        """
        Parameters:
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            nb_layers:      Number of layers in the 70*70 Patch Discriminator
        """

        super().__init__()
        out_dims = in_dims * 3
        conv = nn.Linear(in_dims, out_dims)
        layers = [conv, nn.LeakyReLU(0.2, True)]

        for _ in range(1, nb_layers):
            conv = nn.Linear(out_dims, out_dims)
            layers += [conv, nn.LeakyReLU(0.2, True)]

        conv = nn.Linear(out_dims, out_dims)
        layers += [conv, nn.LeakyReLU(0.2, True)]

        conv = nn.Linear(out_dims, 1)
        layers += [conv]

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CycleGan(pl.LightningModule):

    loss: Loss
    g_A2B: nn.Module
    g_B2A: nn.Module
    d_A: nn.Module
    d_B: nn.Module
    d_A_params: Iterator[Parameter]
    d_B_params: Iterator[Parameter]
    g_params: Iterator[Parameter]
    data_cols: list[str]
    scaler: MinMaxScaler
    example_input_array: list[Tensor]
    built: bool

    def __init__(
        self,
        d_lr: float = 2e-4,
        g_lr: float = 2e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        epoch_decay: int = 200,
    ):

        super().__init__()

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epoch_decay = epoch_decay

        self.fake_pool_A = HistoryPool(pool_sz=50)
        self.fake_pool_B = HistoryPool(pool_sz=50)

        self.init_fn = Initializer(init_type='normal', init_gain=0.02)

    def build(
        self,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        scaler: MinMaxScaler,
    ) -> None:

        self.loss = Loss(loss_type='MSE', lambda_=1, feature_groups=feature_groups)
        self.g_A2B = self.init_fn(Generator(in_dims=data_dim))
        self.g_B2A = self.init_fn(Generator(in_dims=data_dim))
        self.d_A = self.init_fn(Discriminator(in_dims=data_dim))
        self.d_B = self.init_fn(Discriminator(in_dims=data_dim))
        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        self.g_params = itertools.chain([*self.g_A2B.parameters(), *self.g_B2A.parameters()])
        self.data_cols = outcome_cols
        self.scaler = scaler

        self.example_input_array = [
            torch.rand(33, data_dim, device=self.device),
            torch.rand(33, data_dim, device=self.device),
        ]
        self.built = True

    @staticmethod
    def set_requires_grad(nets: nn.Module | list[nn.Module], requires_grad: bool = False) -> None:

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def forward(self, real_a: Tensor, real_b: Tensor) -> CycleFwd:
        fake_b = self.g_A2B(real_a)
        fake_a = self.g_B2A(real_b)
        return CycleFwd(fake_b=fake_b, fake_a=fake_a)

    def forward_gen(self, real_a: Tensor, real_b: Tensor, fake_a: Tensor, fake_b: Tensor) -> GenFwd:
        cyc_a = self.g_B2A(fake_b)
        idt_a = self.g_B2A(real_a)

        cyc_b = self.g_A2B(fake_a)
        idt_b = self.g_A2B(real_b)
        return GenFwd(cyc_a=cyc_a, idt_a=idt_a, cyc_b=cyc_b, idt_b=idt_b)

    @staticmethod
    def forward_dis(dis: nn.Module, real_data: Tensor, fake_data: Tensor) -> DisFwd:

        """
        Gets the Discriminator output
        Parameters:
            dis       -- Discriminator
            real_data -- real image
            fake_data -- fake image
        """

        pred_real_data = dis(real_data)
        pred_fake_data = dis(fake_data)

        return DisFwd(real=pred_real_data, fake=pred_fake_data)

    def training_step(self, batch: Batch | CfBatch, batch_idx: int, optimizer_idx: int) -> Tensor:
        real_a, real_b = batch.x[batch.s == 0], batch.x[batch.s == 1]
        size = min(len(real_a), len(real_b))
        real_a = real_a[:size]
        real_b = real_b[:size]
        cyc_out = self.forward(real_a, real_b)

        if optimizer_idx == 0:
            gen_fwd = self.forward_gen(real_a, real_b, cyc_out.fake_a, cyc_out.fake_b)

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad=False)
            d_a_pred_fake_data = self.d_A(cyc_out.fake_a)
            d_b_pred_fake_data = self.d_B(cyc_out.fake_b)

            gen_loss = self.loss.get_gen_loss(
                real_a,
                real_b,
                gen_fwd,
                d_a_pred_fake_data,
                d_b_pred_fake_data,
            )

            dict_ = {
                'g_tot_train_loss': gen_loss.tot,
                'g_A2B_train_loss': gen_loss.a2b,
                'g_B2A_train_loss': gen_loss.b2a,
            }
            self.log_dict(dict_, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return gen_loss.tot

        if optimizer_idx == 1:
            self.set_requires_grad([self.d_A], requires_grad=True)
            fake_a = self.fake_pool_A.push_and_pop(cyc_out.fake_a)
            dis_out = self.forward_dis(self.d_A, real_a, fake_a.detach())

            # GAN loss
            d_a_loss = self.loss.get_dis_loss(dis_out.real, dis_out.fake)
            self.log(
                "d_A_train_loss", d_a_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

            return d_a_loss

        if optimizer_idx == 2:
            self.set_requires_grad([self.d_B], requires_grad=True)
            fake_b = self.fake_pool_B.push_and_pop(cyc_out.fake_b)
            dis_b_out = self.forward_dis(self.d_B, real_b, fake_b.detach())

            # GAN loss
            d_b_loss = self.loss.get_dis_loss(dis_b_out.real, dis_b_out.fake)
            self.log(
                "d_B_train_loss", d_b_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

            return d_b_loss
        else:
            raise NotImplementedError("There should only be 3 optimizers.")

    def shared_step(self, batch: Batch | CfBatch, stage: Stage) -> SharedStepOut:
        real_a = batch.x
        real_b = batch.x

        cyc_out = self.forward(real_a, real_b)
        gen_fwd = self.forward_gen(real_a, real_b, cyc_out.fake_a, cyc_out.fake_b)

        dis_out_a = self.forward_dis(self.d_A, real_a, cyc_out.fake_a)
        dis_out_b = self.forward_dis(self.d_B, real_b, cyc_out.fake_b)

        # G_A2B loss, G_B2A loss, G loss
        g_a2b_loss, g_b2a_loss, g_tot_loss = self.loss.get_gen_loss(
            real_a, real_b, gen_fwd, dis_out_a.fake, dis_out_b.fake
        )

        # D_A loss, D_B loss
        d_a_loss = self.loss.get_dis_loss(dis_out_a.real, dis_out_a.fake)
        d_b_loss = self.loss.get_dis_loss(dis_out_b.real, dis_out_b.fake)

        dict_ = {
            f'g_tot_{stage.value}_loss': g_tot_loss,
            f'g_A2B_{stage.value}_loss': g_a2b_loss,
            f'g_B2A_{stage.value}_loss': g_b2a_loss,
            f'd_A_{stage.value}_loss': d_a_loss,
            f'd_B_{stage.value}_loss': d_b_loss,
        }
        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return SharedStepOut(
            x=batch.x,
            s=batch.s,
            recon=self.invert(index_by_s([cyc_out.fake_b, cyc_out.fake_a], batch.s), batch.x),
            cf_pred=self.invert(index_by_s([cyc_out.fake_b, cyc_out.fake_a], 1 - batch.s), batch.x),
            recons_0=self.invert(cyc_out.fake_b, batch.x),
            recons_1=self.invert(cyc_out.fake_a, batch.x),
        )

    def test_epoch_end(self, outputs: list[SharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.test)

    def validation_epoch_end(self, outputs: list[SharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.validate)

    def shared_epoch_end(self, outputs: list[SharedStepOut], stage: Stage) -> None:
        self.all_x = torch.cat([_r.x for _r in outputs], 0)
        self.all_s = torch.cat([_r.s for _r in outputs], 0)
        self.all_recon = torch.cat([_r.recon for _r in outputs], 0)
        self.all_cf_pred = torch.cat([_r.recon for _r in outputs], 0)

    def validation_step(self, batch: Batch | CfBatch, batch_idx: int) -> SharedStepOut:
        return self.shared_step(batch, Stage.validate)

    def test_step(self, batch: Batch | CfBatch, batch_idx: int) -> SharedStepOut:
        return self.shared_step(batch, Stage.test)

    def lr_lambda(self, epoch: int) -> float:

        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[LambdaLR]]:

        # define the optimizers here
        g_opt = torch.optim.Adam(self.g_params, lr=self.g_lr, betas=(self.beta_1, self.beta_2))
        d_a_opt = torch.optim.Adam(self.d_A_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))
        d_b_opt = torch.optim.Adam(self.d_B_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))

        # define the lr_schedulers here
        g_sch = LambdaLR(g_opt, lr_lambda=self.lr_lambda)
        d_a_sch = LambdaLR(d_a_opt, lr_lambda=self.lr_lambda)
        d_b_sch = LambdaLR(d_b_opt, lr_lambda=self.lr_lambda)

        # first return value is a list of optimizers and second is a list of lr_schedulers
        # (you can return empty list also)
        return [g_opt, d_a_opt, d_b_opt], [g_sch, d_a_sch, d_b_sch]

    @torch.no_grad()
    def invert(self, z: Tensor, x: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        k = z.detach().clone()
        if self.loss.feature_groups["discrete"]:
            for i in range(
                k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])].shape[1]
            ):
                if i in []:  # [0]: Features to transplant to the reconstrcution
                    k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ] = x[:, slice(self.loss.feature_groups["discrete"][-1].stop, x.shape[1])][:, i]
                else:
                    k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ] = k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ].sigmoid()
            for i, group_slice in enumerate(self.loss.feature_groups["discrete"]):
                if i in []:  # [2, 4]: Features to transplant
                    k[:, group_slice] = x[:, group_slice]
                else:
                    one_hot = to_discrete(inputs=k[:, group_slice])
                    k[:, group_slice] = one_hot
        else:
            k = k.sigmoid()

        return k


class GenFwd(NamedTuple):
    cyc_a: Tensor
    idt_a: Tensor
    cyc_b: Tensor
    idt_b: Tensor


class DisFwd(NamedTuple):
    real: Tensor
    fake: Tensor


class CycleFwd(NamedTuple):
    fake_b: Tensor
    fake_a: Tensor
