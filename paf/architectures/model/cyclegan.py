"""Taken from https://github.com/Adi-iitd/AI-Art/blob/master/src/CycleGAN/CycleGAN-PL.py ."""
from __future__ import annotations
from enum import Enum, auto
import itertools
import logging
from typing import Any, Iterator, NamedTuple

from conduit.data import TernarySample
from conduit.fair.data import EthicMlDataModule
from conduit.types import Stage
import numpy as np
from ranzen import implements, parsable
import torch
from torch import Tensor, nn, optim
from torch.nn import Parameter
from torch.utils.data import DataLoader

from paf.base_templates.dataset_utils import Batch, CfBatch

from .model_components import Adversary, CommonModel, Decoder, Encoder, index_by_s, to_discrete

__all__ = [
    "SharedStepOut",
    "Initializer",
    "HistoryPool",
    "Loss",
    "GenLoss",
    "ResBlock",
    "Generator",
    "Discriminator",
    "CycleGan",
    "GenFwd",
    "DisFwd",
    "CycleFwd",
]

from ...base_templates import BaseDataModule
from ...plotting import make_plot
from ...utils import HistoryPool, Stratifier

logger = logging.getLogger(__name__)


class SharedStepOut(NamedTuple):
    x: Tensor
    s: Tensor
    recon: Tensor
    cf_pred: Tensor
    recons_0: Tensor
    recons_1: Tensor
    idt_recon: Tensor
    cyc_recon: Tensor
    real_s0: Tensor
    real_s1: Tensor


class InitType(Enum):
    KAIMING = auto()
    NORMAL = auto()
    XAVIER = auto()
    UNIFORM = auto()


class Initializer:
    def __init__(self, init_type: InitType = InitType.NORMAL, *, init_gain: float = 0.02):
        self.init_type = init_type
        self.init_gain = init_gain

    def init_module(self, module: nn.Module) -> None:
        cls_name = module.__class__.__name__
        if hasattr(module, "weight") and (
            cls_name.find("Conv") != -1 or cls_name.find("Linear") != -1
        ):
            if self.init_type is InitType.KAIMING:
                nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif self.init_type is InitType.XAVIER:
                nn.init.xavier_normal_(module.weight.data, gain=self.init_gain)  # type: ignore[arg-type]
            elif self.init_type is InitType.NORMAL:
                nn.init.normal_(module.weight.data, mean=0, std=self.init_gain)  # type: ignore[arg-type]
            elif self.init_type is InitType.UNIFORM:
                nn.init.xavier_uniform_(module.weight.data)  # type: ignore[arg-type]
            else:
                raise ValueError("Initialization not found!!")

            if module.bias is not None:
                nn.init.constant_(module.bias.data, val=0)  # type: ignore[arg-type]

        if hasattr(module, "weight") and cls_name.find("BatchNorm2d") != -1:
            nn.init.normal_(module.weight.data, mean=1.0, std=self.init_gain)  # type: ignore[arg-type]
            nn.init.constant_(module.bias.data, val=0)  # type: ignore[arg-type]

    def __call__(self, net: nn.Module) -> nn.Module:
        net.apply(self.init_module)
        return net


class LossType(Enum):
    MSE = auto()
    BCE = auto()


class Loss:
    def __init__(
        self,
        loss_type: LossType = LossType.MSE,
        lambda_: float = 10,
        feature_groups: dict[str, list[slice]] | None = None,
    ):
        """Init Loss."""
        self._loss_fn = nn.MSELoss() if loss_type is LossType.MSE else nn.BCEWithLogitsLoss()
        self._recon_loss_fn = nn.L1Loss(reduction="mean")
        self.lambda_ = lambda_
        self.feature_groups = feature_groups if feature_groups is not None else {}

    def get_dis_loss(self, dis_pred_real_data: Tensor, *, dis_pred_fake_data: Tensor) -> Tensor:
        dis_tar_real_data = torch.ones_like(dis_pred_real_data, requires_grad=False)
        dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad=False)

        loss_real_data = self._loss_fn(dis_pred_real_data, dis_tar_real_data)
        loss_fake_data = self._loss_fn(dis_pred_fake_data, dis_tar_fake_data)
        return (loss_real_data + loss_fake_data) * 0.5

    def get_gen_gan_loss(self, dis_pred_fake_data: Tensor) -> Tensor:
        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad=False)
        return self._loss_fn(dis_pred_fake_data, gen_tar_fake_data)

    def get_gen_cyc_loss(self, real_data: Tensor, *, cyc_data: Tensor) -> Tensor:
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

    def get_gen_idt_loss(self, real_data: Tensor, *, idt_data: Tensor) -> Tensor:
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
        *,
        real_s0: Tensor,
        real_s1: Tensor,
        gen_fwd: GenFwd,
        d_s0_pred_fake_data: Tensor,
        d_s1_pred_fake_data: Tensor,
    ) -> GenLoss:
        # Cycle loss
        cyc_loss_s0 = self.get_gen_cyc_loss(real_data=real_s0, cyc_data=gen_fwd.cyc_s0)
        cyc_loss_s1 = self.get_gen_cyc_loss(real_data=real_s1, cyc_data=gen_fwd.cyc_s1)
        tot_cyc_loss = cyc_loss_s0 + cyc_loss_s1

        # GAN loss
        g_s0_2_s1_gan_loss = self.get_gen_gan_loss(d_s1_pred_fake_data)
        g_s1_2_s0_gan_loss = self.get_gen_gan_loss(d_s0_pred_fake_data)

        # Identity loss
        g_s1_2_s0_idt_loss = self.get_gen_idt_loss(real_data=real_s0, idt_data=gen_fwd.idt_s0)
        g_s0_2_s1_idt_loss = self.get_gen_idt_loss(real_data=real_s1, idt_data=gen_fwd.idt_s1)

        # Total individual losses
        g_s0_s1_loss = g_s0_2_s1_gan_loss + g_s0_2_s1_idt_loss + tot_cyc_loss
        g_s1_2_s0_loss = g_s1_2_s0_gan_loss + g_s1_2_s0_idt_loss + tot_cyc_loss
        g_tot_loss = g_s0_s1_loss + g_s1_2_s0_loss - tot_cyc_loss
        return GenLoss(
            s0_2_s1=g_s0_s1_loss,
            s1_2_s0=g_s1_2_s0_loss,
            tot=g_tot_loss,
            cycle_loss=tot_cyc_loss,
            s0_idt=g_s1_2_s0_idt_loss,
            s1_idt=g_s0_2_s1_idt_loss,
            s0_cyc=cyc_loss_s0,
            s1_cyc=cyc_loss_s1,
        )


class GenLoss(NamedTuple):
    s0_2_s1: Tensor
    s1_2_s0: Tensor
    tot: Tensor
    cycle_loss: Tensor
    s0_idt: Tensor
    s1_idt: Tensor
    s0_cyc: Tensor
    s1_cyc: Tensor


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, *, apply_dp: bool = True):
        """ResBlock."""
        super().__init__()
        conv = nn.Linear(in_channels, in_channels)
        layers = [conv, nn.SELU(), nn.LayerNorm(in_channels)]
        if apply_dp:
            layers += [nn.Dropout(0.5)]
        conv = nn.Linear(in_channels, in_channels)
        layers += [conv]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class Generator(nn.Module):
    net: nn.Module

    def __init__(self, in_dims: int, *, latent_multiplier: int, nb_resblks: int):
        """Generator."""
        super().__init__()
        out_dims = in_dims * latent_multiplier
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

    def __init__(self, in_dims: int, *, latent_multiplier: int, nb_layers: int):
        """Discriminator."""
        super().__init__()
        out_dims = in_dims * latent_multiplier
        conv = nn.Linear(in_dims, out_dims)
        layers = [conv, nn.SELU(), nn.LayerNorm(out_dims)]

        for _ in range(1, nb_layers):
            conv = nn.Linear(out_dims, out_dims)
            layers += [conv, nn.SELU(), nn.LayerNorm(out_dims)]

        conv = nn.Linear(out_dims, out_dims)
        layers += [conv, nn.SELU(), nn.LayerNorm(out_dims)]

        conv = nn.Linear(out_dims, 1)
        layers += [conv]

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CycleGan(CommonModel):
    loss: Loss
    g_s0_2_s1: nn.Module
    g_s1_2_s0: nn.Module
    d_s0: nn.Module
    d_s1: nn.Module
    d_a_params: Iterator[Parameter]
    d_b_params: Iterator[Parameter]
    g_params: Iterator[Parameter]
    data_cols: list[str]
    example_input_array: dict[str, Tensor]
    built: bool
    all_x: Tensor
    all_s: Tensor
    all_recon: Tensor
    all_cf_pred: Tensor

    @parsable
    def __init__(
        self,
        encoder_blocks: int,
        decoder_blocks: int,
        adv_blocks: int,
        latent_multiplier: int,
        batch_size: int,
        g_weight_decay: float,
        d_weight_decay: float,
        latent_dims: int,
        scheduler_rate: float = 0.99,
        d_lr: float = 2e-4,
        g_lr: float = 2e-4,
        debug: bool = False,
        adv_weight: float = 1.0,
        lambda_: float = 10.0,
    ):
        super().__init__(name="CycleGan")
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.scheduler_rate = scheduler_rate

        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.adv_blocks = adv_blocks
        self.latent_multiplier = latent_multiplier

        self.fake_pool_s0 = HistoryPool(pool_size=batch_size // 4)
        self.fake_pool_s1 = HistoryPool(pool_size=batch_size // 4)

        self.pool_x0 = Stratifier(pool_size=batch_size // 2)
        self.pool_x1 = Stratifier(pool_size=batch_size // 2)

        self.g_weight_decay = g_weight_decay
        self.d_weight_decay = d_weight_decay

        self.lambda_ = lambda_

        self._adv_weight = adv_weight

        self.init_fn = Initializer(init_type=InitType.UNIFORM)

        self.s_as_input = False
        self.latent_dims = latent_dims

        self.debug = debug

    @implements(CommonModel)
    def build(
        self,
        *,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        data: BaseDataModule | EthicMlDataModule,
        indices: list[str] | None,
    ) -> None:
        _ = (num_s, s_dim, cf_available, indices, data)
        self.data_dim = data_dim
        self.loss = Loss(
            loss_type=LossType.BCE, lambda_=self.lambda_, feature_groups=feature_groups
        )
        self.g_s0_2_s1 = nn.Sequential(
            Encoder(
                in_size=self.data_dim + s_dim if self.s_as_input else self.data_dim,
                latent_dim=self.latent_dims,
                blocks=self.encoder_blocks,
                hid_multiplier=self.latent_multiplier,
            ),
            Decoder(
                latent_dim=self.latent_dims,
                in_size=self.data_dim,
                blocks=self.decoder_blocks,
                hid_multiplier=self.latent_multiplier,
            ),
        )
        self.g_s1_2_s0 = nn.Sequential(
            Encoder(
                in_size=self.data_dim + s_dim if self.s_as_input else self.data_dim,
                latent_dim=self.latent_dims,
                blocks=self.encoder_blocks,
                hid_multiplier=self.latent_multiplier,
            ),
            Decoder(
                latent_dim=self.latent_dims,
                in_size=self.data_dim,
                blocks=self.decoder_blocks,
                hid_multiplier=self.latent_multiplier,
            ),
        )
        self.d_s0 = self.init_fn(
            Decoder(
                latent_dim=self.data_dim,
                in_size=1,
                blocks=self.adv_blocks,
                hid_multiplier=self.latent_multiplier,
            )
        )
        self.d_s1 = self.init_fn(
            Decoder(
                latent_dim=self.data_dim,
                in_size=1,
                blocks=self.adv_blocks,
                hid_multiplier=self.latent_multiplier,
            )
        )
        self.d_a_params = self.d_s0.parameters()
        self.d_b_params = self.d_s1.parameters()
        self.g_params = itertools.chain(
            [*self.g_s0_2_s1.parameters(), *self.g_s1_2_s0.parameters()]
        )
        self.data_cols = outcome_cols
        self.example_input_array = {
            "real_s0": torch.rand(33, data_dim, device=self.device),
            "real_s1": torch.rand(33, data_dim, device=self.device),
        }
        self.built = True

    def soft_invert(self, z: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        if self.loss.feature_groups["discrete"]:
            for i in range(
                z[:, slice(self.loss.feature_groups["discrete"][-1].stop, z.shape[1])].shape[1]
            ):
                z[:, slice(self.loss.feature_groups["discrete"][-1].stop, z.shape[1])][:, i] = z[
                    :, slice(self.loss.feature_groups["discrete"][-1].stop, z.shape[1])
                ][:, i].sigmoid()
            for i, group_slice in enumerate(self.loss.feature_groups["discrete"]):
                z[:, group_slice] = torch.nn.functional.softmax(z[:, group_slice], dim=-1)
        else:
            z = z.sigmoid()
        return z

    @staticmethod
    def set_requires_grad(nets: nn.Module | list[nn.Module], requires_grad: bool = False) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def forward(self, *, real_s0: Tensor, real_s1: Tensor) -> CycleFwd:
        fake_s1 = self.g_s0_2_s1(real_s0)
        fake_s0 = self.g_s1_2_s0(real_s1)
        return CycleFwd(fake_s1=fake_s1, fake_s0=fake_s0)

    def forward_gen(
        self, *, real_s0: Tensor, real_s1: Tensor, fake_s0: Tensor, fake_s1: Tensor
    ) -> GenFwd:
        cyc_s0 = self.g_s1_2_s0(self.soft_invert(fake_s1))
        idt_s0 = self.g_s1_2_s0(real_s0)

        cyc_s1 = self.g_s0_2_s1(self.soft_invert(fake_s0))
        idt_s1 = self.g_s0_2_s1(real_s1)
        return GenFwd(cyc_s0=cyc_s0, idt_s0=idt_s0, cyc_s1=cyc_s1, idt_s1=idt_s1)

    @staticmethod
    def forward_dis(dis: nn.Module, *, real_data: Tensor, fake_data: Tensor) -> DisFwd:
        pred_real_data = dis(real_data)
        pred_fake_data = dis(fake_data)
        return DisFwd(real=pred_real_data, fake=pred_fake_data)

    def training_step(
        self, batch: Batch | CfBatch | TernarySample, batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        _ = (batch_idx,)
        x0 = self.pool_x0.push_and_pop(batch.x[batch.s == 0])
        s0 = batch.x.new_zeros(x0.shape[0])
        x1 = self.pool_x1.push_and_pop(batch.x[batch.s == 1])
        s1 = batch.x.new_ones(x1.shape[0])
        x = torch.cat([x0, x1], dim=0)
        s = torch.cat([s0, s1], dim=0)

        real_s0, real_s1 = x[s == 0], x[s == 1]
        size = min(len(real_s0), len(real_s1))
        real_s0 = real_s0[:size]
        real_s1 = real_s1[:size]
        cyc_out = self.forward(real_s0=real_s0, real_s1=real_s1)

        if optimizer_idx == 0:
            gen_fwd = self.forward_gen(
                real_s0=real_s0, real_s1=real_s1, fake_s0=cyc_out.fake_s0, fake_s1=cyc_out.fake_s1
            )

            # mmd_results = self.mmd_reporting(
            #     gen_fwd=gen_fwd, enc_fwd=cyc_out, batch=batch, train=True
            # )
            # self.log(f"{Stage.fit}/enc/recon_mmd", mmd_results.recon)
            # self.log(f"{Stage.fit}/enc/cf_recon_mmd", mmd_results.cf_recon)
            # self.log(f"{Stage.fit}/enc/s0_dist_mmd", mmd_results.s0_dist)
            # self.log(f"{Stage.fit}/enc/s1_dist_mmd", mmd_results.s1_dist)

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_s0, self.d_s1], requires_grad=False)
            d_s0_pred_fake_data = self.d_s0(self.invert(cyc_out.fake_s0))
            d_s1_pred_fake_data = self.d_s1(self.invert(cyc_out.fake_s1))

            gen_loss = self.loss.get_gen_loss(
                real_s0=real_s0,
                real_s1=real_s1,
                gen_fwd=gen_fwd,
                d_s0_pred_fake_data=d_s0_pred_fake_data,
                d_s1_pred_fake_data=d_s1_pred_fake_data,
            )

            self.log(f"{Stage.fit}/enc/g_tot_loss", gen_loss.tot)
            self.log(f"{Stage.fit}/enc/g_A2B_loss", gen_loss.s0_2_s1)
            self.log(f"{Stage.fit}/enc/g_B2A_loss", gen_loss.s1_2_s0)
            self.log(f"{Stage.fit}/enc/cycle_loss", gen_loss.cycle_loss)
            self.log(f"{Stage.fit}/enc/s0_idt_loss", gen_loss.s0_idt)
            self.log(f"{Stage.fit}/enc/s1_idt_loss", gen_loss.s1_idt)
            self.log(f"{Stage.fit}/enc/s0_cyc_loss", gen_loss.s0_cyc)
            self.log(f"{Stage.fit}/enc/s1_cyc_loss", gen_loss.s1_cyc)

            return gen_loss.tot

        if optimizer_idx in {1, 2}:
            self.set_requires_grad([self.d_s0], requires_grad=True)
            fake_s0 = self.fake_pool_s0.push_and_pop(self.invert(cyc_out.fake_s0, cyc_out.fake_s0))
            dis_out = self.forward_dis(dis=self.d_s0, real_data=real_s0, fake_data=fake_s0.detach())

            # GAN loss
            d_s0_loss = self.loss.get_dis_loss(
                dis_pred_real_data=dis_out.real, dis_pred_fake_data=dis_out.fake
            )
            self.log(f"{Stage.fit}/enc/d_A_loss", d_s0_loss)
            return d_s0_loss

        if optimizer_idx in {3, 4}:
            self.set_requires_grad([self.d_s1], requires_grad=True)
            fake_s1 = self.fake_pool_s1.push_and_pop(self.invert(cyc_out.fake_s1, cyc_out.fake_s1))
            dis_s1_out = self.forward_dis(
                dis=self.d_s1, real_data=real_s1, fake_data=fake_s1.detach()
            )

            # GAN loss
            d_s1_loss = self.loss.get_dis_loss(
                dis_pred_real_data=dis_s1_out.real, dis_pred_fake_data=dis_s1_out.fake
            )
            self.log(f"{Stage.fit}/enc/d_B_loss", d_s1_loss)
            return d_s1_loss
        raise NotImplementedError("There should only be 3 optimizers.")

    def shared_step(self, batch: Batch | CfBatch | TernarySample, *, stage: Stage) -> SharedStepOut:
        real_s0 = batch.x
        real_s1 = batch.x

        cyc_out = self.forward(real_s0=real_s0, real_s1=real_s1)
        gen_fwd = self.forward_gen(
            real_s0=real_s0, real_s1=real_s1, fake_s0=cyc_out.fake_s0, fake_s1=cyc_out.fake_s1
        )

        dis_out_a = self.forward_dis(dis=self.d_s0, real_data=real_s0, fake_data=cyc_out.fake_s0)
        dis_out_b = self.forward_dis(dis=self.d_s1, real_data=real_s1, fake_data=cyc_out.fake_s1)

        # G_A2B loss, G_B2A loss, G loss
        gen_losses = self.loss.get_gen_loss(
            real_s0=real_s0,
            real_s1=real_s1,
            gen_fwd=gen_fwd,
            d_s0_pred_fake_data=dis_out_a.fake,
            d_s1_pred_fake_data=dis_out_b.fake,
        )

        # D_A loss, D_B loss
        d_a_loss = self.loss.get_dis_loss(
            dis_pred_real_data=dis_out_a.real, dis_pred_fake_data=dis_out_a.fake
        )
        d_b_loss = self.loss.get_dis_loss(
            dis_pred_real_data=dis_out_b.real, dis_pred_fake_data=dis_out_b.fake
        )

        # mmd_results = self.mmd_reporting(gen_fwd=gen_fwd, enc_fwd=cyc_out, batch=batch)

        dict_ = {
            f"{stage}/enc/g_tot_loss": gen_losses.tot,
            f"{stage}/enc/g_A2B_loss": gen_losses.s0_2_s1,
            f"{stage}/enc/g_B2A_loss": gen_losses.s1_2_s0,
            f"{stage}/enc/d_A_loss": d_a_loss,
            f"{stage}/enc/d_B_loss": d_b_loss,
            f"{stage}/enc/loss": gen_losses.tot,
            f"{stage}/enc/cycle_loss": gen_losses.cycle_loss,
            # f"{stage}/enc/recon_mmd": mmd_results.recon,
            # f"{stage}/enc/cf_recon_mmd": mmd_results.cf_recon,
            # f"{stage}/enc/s0_dist_mmd": mmd_results.s0_dist,
            # f"{stage}/enc/s1_dist_mmd": mmd_results.s1_dist,
        }
        self.log_dict(dict_)

        return SharedStepOut(
            x=batch.x,
            s=batch.s,
            recon=self.invert(index_by_s([cyc_out.fake_s0, cyc_out.fake_s1], batch.s)),
            cf_pred=self.invert(index_by_s([cyc_out.fake_s1, cyc_out.fake_s0], batch.s)),
            real_s0=real_s0[batch.s == 0],
            real_s1=real_s1[batch.s == 1],
            recons_0=self.invert(cyc_out.fake_s0[batch.s == 1]),
            recons_1=self.invert(cyc_out.fake_s1[batch.s == 0]),
            idt_recon=self.invert(index_by_s([gen_fwd.idt_s0, gen_fwd.idt_s1], batch.s)),
            cyc_recon=self.invert(index_by_s([gen_fwd.cyc_s0, gen_fwd.cyc_s1], batch.s)),
        )

    def test_epoch_end(self, outputs: list[SharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.test)

    def validation_epoch_end(self, outputs: list[SharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.validate)

    def shared_epoch_end(self, outputs: list[SharedStepOut], *, stage: Stage) -> None:
        self.all_x = torch.cat([_r.x for _r in outputs], 0)
        self.all_s = torch.cat([_r.s for _r in outputs], 0)
        self.all_recon = torch.cat([_r.recon for _r in outputs], 0)
        self.all_cf_pred = torch.cat([_r.recon for _r in outputs], 0)
        all_idt = torch.cat([_r.idt_recon for _r in outputs], 0)
        all_cyc = torch.cat([_r.cyc_recon for _r in outputs], 0)
        real_s0 = torch.cat([_r.real_s0 for _r in outputs], 0)
        real_s1 = torch.cat([_r.real_s1 for _r in outputs], 0)
        fake_s0 = torch.cat([_r.recons_0 for _r in outputs], 0)
        fake_s1 = torch.cat([_r.recons_1 for _r in outputs], 0)

        if self.debug:
            make_plot(
                x=self.all_recon.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_recon",
                cols=self.data_cols,
            )
            make_plot(
                x=self.all_x.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_true_x",
                cols=self.data_cols,
            )
            make_plot(
                x=self.all_cf_pred.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_cf_recons",
                cols=self.data_cols,
            )
            make_plot(
                x=all_idt,
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_idt_recon",
                cols=self.data_cols,
            )
            make_plot(
                x=all_cyc,
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_cyc_recons",
                cols=self.data_cols,
            )
            make_plot(
                x=real_s0,
                s=torch.zeros(real_s0.shape[0]),
                logger=self.logger,
                name=f"{stage}_s0",
                cols=self.data_cols,
            )
            make_plot(
                x=real_s1,
                s=torch.ones(real_s1.shape[0]),
                logger=self.logger,
                name=f"{stage}_s1",
                cols=self.data_cols,
            )
            make_plot(
                x=fake_s0,
                s=torch.zeros(fake_s0.shape[0]),
                logger=self.logger,
                name=f"{stage}_fake_s0",
                cols=self.data_cols,
            )
            make_plot(
                x=fake_s1,
                s=torch.ones(fake_s1.shape[0]),
                logger=self.logger,
                name=f"{stage}_fake_s1",
                cols=self.data_cols,
            )

    def validation_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> SharedStepOut:
        return self.shared_step(batch=batch, stage=Stage.validate)

    def test_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> SharedStepOut:
        return self.shared_step(batch=batch, stage=Stage.test)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[optim.lr_scheduler.ExponentialLR]]:

        # define the optimizers here
        g_opt = torch.optim.AdamW(self.g_params, lr=self.g_lr, weight_decay=self.g_weight_decay)
        d_a_opt_1 = torch.optim.AdamW(
            self.d_a_params, lr=self.d_lr, weight_decay=self.d_weight_decay
        )
        d_a_opt_2 = torch.optim.AdamW(
            self.d_a_params, lr=self.d_lr, weight_decay=self.d_weight_decay
        )
        d_b_opt_1 = torch.optim.AdamW(
            self.d_b_params, lr=self.d_lr, weight_decay=self.d_weight_decay
        )
        d_b_opt_2 = torch.optim.AdamW(
            self.d_b_params, lr=self.d_lr, weight_decay=self.d_weight_decay
        )

        # define the lr_schedulers here
        g_sch = optim.lr_scheduler.ExponentialLR(g_opt, gamma=self.scheduler_rate)
        d_a_sch_1 = optim.lr_scheduler.ExponentialLR(d_a_opt_1, gamma=self.scheduler_rate)
        d_a_sch_2 = optim.lr_scheduler.ExponentialLR(d_a_opt_2, gamma=self.scheduler_rate)
        d_b_sch_1 = optim.lr_scheduler.ExponentialLR(d_b_opt_1, gamma=self.scheduler_rate)
        d_b_sch_2 = optim.lr_scheduler.ExponentialLR(d_b_opt_2, gamma=self.scheduler_rate)

        # first return value is a list of optimizers and second is a list of lr_schedulers
        # (you can return empty list also)
        return [g_opt, d_a_opt_1, d_a_opt_2, d_b_opt_1, d_b_opt_2], [
            g_sch,
            d_a_sch_1,
            d_a_sch_2,
            d_b_sch_1,
            d_b_sch_2,
        ]

    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("This shouldn't be called. Only implementing for the abc.")

    # def mmd_reporting(
    #     self,
    #     *,
    #     gen_fwd: GenFwd,
    #     enc_fwd: CycleFwd,
    #     batch: Batch | CfBatch | TernarySample,
    #     train: bool = False,
    # ) -> MmdReportingResults:
    #     with torch.no_grad():
    #         if train:
    #             x = torch.cat([gen_fwd.idt_a, gen_fwd.idt_b], dim=0)
    #             cf_x = torch.cat([enc_fwd.fake_a, enc_fwd.fake_b], dim=0)
    #             recon_mmd = mmd2(
    #                 batch.x,
    #                 self.invert(x, batch.x),
    #                 kernel=KernelType.LINEAR,
    #             )
    #             cf_mmd = mmd2(
    #                 batch.x,
    #                 self.invert(cf_x, batch.x),
    #                 kernel=KernelType.LINEAR,
    #             )
    #             s0_dist_mmd = mmd2(
    #                 batch.x[batch.s == 0],
    #                 self.invert(enc_fwd.fake_a, batch.x),
    #                 kernel=KernelType.LINEAR,
    #                 biased=True,
    #             )
    #             s1_dist_mmd = mmd2(
    #                 batch.x[batch.s == 1],
    #                 self.invert(enc_fwd.fake_b, batch.x),
    #                 kernel=KernelType.LINEAR,
    #                 biased=True,
    #             )
    #         else:
    #             x = [enc_fwd.fake_a, enc_fwd.fake_b]
    #             recon_mmd = mmd2(
    #                 batch.x,
    #                 self.invert(index_by_s(x, batch.s), batch.x),
    #                 kernel=KernelType.LINEAR,
    #             )
    #             cf_mmd = mmd2(
    #                 batch.x,
    #                 self.invert(index_by_s(x, 1 - batch.s), batch.x),
    #                 kernel=KernelType.LINEAR,
    #             )
    #             s0_dist_mmd = mmd2(
    #                 batch.x[batch.s == 0],
    #                 self.invert(index_by_s(x, torch.zeros_like(batch.s)), batch.x),
    #                 kernel=KernelType.LINEAR,
    #                 biased=True,
    #             )
    #             s1_dist_mmd = mmd2(
    #                 batch.x[batch.s == 1],
    #                 self.invert(index_by_s(x, torch.ones_like(batch.s)), batch.x),
    #                 kernel=KernelType.LINEAR,
    #                 biased=True,
    #             )
    #
    #     return MmdReportingResults(
    #         recon=recon_mmd, cf_recon=cf_mmd, s0_dist=s0_dist_mmd, s1_dist=s1_dist_mmd
    #     )


class GenFwd(NamedTuple):
    cyc_s0: Tensor
    idt_s0: Tensor
    cyc_s1: Tensor
    idt_s1: Tensor


class DisFwd(NamedTuple):
    real: Tensor
    fake: Tensor


class CycleFwd(NamedTuple):
    fake_s1: Tensor
    fake_s0: Tensor

    @property
    def x(self) -> list[Tensor]:
        return [self.fake_s0, self.fake_s1]
