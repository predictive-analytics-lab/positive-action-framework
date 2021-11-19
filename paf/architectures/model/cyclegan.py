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

from .model_components import CommonModel, index_by_s

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
        lambda_: int = 10,
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
        real_a: Tensor,
        real_b: Tensor,
        gen_fwd: GenFwd,
        d_a_pred_fake_data: Tensor,
        d_b_pred_fake_data: Tensor,
    ) -> GenLoss:
        # Cycle loss
        cyc_loss_a = self.get_gen_cyc_loss(real_data=real_a, cyc_data=gen_fwd.cyc_a)
        cyc_loss_b = self.get_gen_cyc_loss(real_data=real_b, cyc_data=gen_fwd.cyc_b)
        tot_cyc_loss = cyc_loss_a + cyc_loss_b

        # GAN loss
        g_a2b_gan_loss = self.get_gen_gan_loss(d_b_pred_fake_data)
        g_b2a_gan_loss = self.get_gen_gan_loss(d_a_pred_fake_data)

        # Identity loss
        g_b2a_idt_loss = self.get_gen_idt_loss(real_data=real_a, idt_data=gen_fwd.idt_a)
        g_a2b_idt_loss = self.get_gen_idt_loss(real_data=real_b, idt_data=gen_fwd.idt_b)

        # Total individual losses
        g_a2b_loss = g_a2b_gan_loss + g_a2b_idt_loss + tot_cyc_loss
        g_b2a_loss = g_b2a_gan_loss + g_b2a_idt_loss + tot_cyc_loss
        g_tot_loss = g_a2b_loss + g_b2a_loss - tot_cyc_loss

        return GenLoss(a2b=g_a2b_loss, b2a=g_b2a_loss, tot=g_tot_loss, cycle_loss=tot_cyc_loss)


class GenLoss(NamedTuple):
    a2b: Tensor
    b2a: Tensor
    tot: Tensor
    cycle_loss: Tensor


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
    g_a2b: nn.Module
    g_b2a: nn.Module
    d_a: nn.Module
    d_b: nn.Module
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
        blocks: int,
        adv_blocks: int,
        latent_multiplier: int,
        batch_size: int,
        g_weight_decay: float,
        d_weight_decay: float,
        scheduler_rate: float = 0.99,
        d_lr: float = 2e-4,
        g_lr: float = 2e-4,
        debug: bool = False,
    ):
        super().__init__(name="CycleGan")
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.scheduler_rate = scheduler_rate

        self.blocks = blocks
        self.adv_blocks = adv_blocks
        self.latent_multiplier = latent_multiplier

        self.fake_pool_a = HistoryPool(pool_size=batch_size // 4)
        self.fake_pool_b = HistoryPool(pool_size=batch_size // 4)

        self.pool_x0 = Stratifier(pool_size=batch_size // 2)
        self.pool_x1 = Stratifier(pool_size=batch_size // 2)

        self.g_weight_decay = g_weight_decay
        self.d_weight_decay = d_weight_decay

        self.init_fn = Initializer(init_type=InitType.UNIFORM)

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
        self.loss = Loss(loss_type=LossType.MSE, lambda_=1, feature_groups=feature_groups)
        self.g_a2b = self.init_fn(
            Generator(
                in_dims=data_dim, nb_resblks=self.blocks, latent_multiplier=self.latent_multiplier
            )
        )
        self.g_b2a = self.init_fn(
            Generator(
                in_dims=data_dim, nb_resblks=self.blocks, latent_multiplier=self.latent_multiplier
            )
        )
        self.d_a = self.init_fn(
            Discriminator(
                in_dims=data_dim,
                nb_layers=self.adv_blocks,
                latent_multiplier=self.latent_multiplier,
            )
        )
        self.d_b = self.init_fn(
            Discriminator(
                in_dims=data_dim,
                nb_layers=self.adv_blocks,
                latent_multiplier=self.latent_multiplier,
            )
        )
        self.d_a_params = self.d_a.parameters()
        self.d_b_params = self.d_b.parameters()
        self.g_params = itertools.chain([*self.g_a2b.parameters(), *self.g_b2a.parameters()])
        self.data_cols = outcome_cols
        self.example_input_array = {
            "real_a": torch.rand(33, data_dim, device=self.device),
            "real_b": torch.rand(33, data_dim, device=self.device),
        }
        self.built = True

    @staticmethod
    def set_requires_grad(nets: nn.Module | list[nn.Module], requires_grad: bool = False) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def forward(self, *, real_a: Tensor, real_b: Tensor) -> CycleFwd:
        fake_b = self.g_a2b(real_a)
        fake_a = self.g_b2a(real_b)
        return CycleFwd(fake_b=fake_b, fake_a=fake_a)

    def forward_gen(
        self, *, real_a: Tensor, real_b: Tensor, fake_a: Tensor, fake_b: Tensor
    ) -> GenFwd:
        cyc_a = self.g_b2a(fake_b)
        idt_a = self.g_b2a(real_a)

        cyc_b = self.g_a2b(fake_a)
        idt_b = self.g_a2b(real_b)
        return GenFwd(cyc_a=cyc_a, idt_a=idt_a, cyc_b=cyc_b, idt_b=idt_b)

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
        s0 = batch.x.new_zeros((x0.shape[0]))
        x1 = self.pool_x1.push_and_pop(batch.x[batch.s == 1])
        s1 = batch.x.new_ones((x1.shape[0]))
        x = torch.cat([x0, x1], dim=0)
        s = torch.cat([s0, s1], dim=0)

        real_a, real_b = x[s == 0], x[s == 1]
        size = min(len(real_a), len(real_b))
        real_a = real_a[:size]
        real_b = real_b[:size]
        cyc_out = self.forward(real_a=real_a, real_b=real_b)

        if optimizer_idx == 0:
            gen_fwd = self.forward_gen(
                real_a=real_a, real_b=real_b, fake_a=cyc_out.fake_a, fake_b=cyc_out.fake_b
            )

            # mmd_results = self.mmd_reporting(
            #     gen_fwd=gen_fwd, enc_fwd=cyc_out, batch=batch, train=True
            # )
            # self.log(f"{Stage.fit}/enc/recon_mmd", mmd_results.recon)
            # self.log(f"{Stage.fit}/enc/cf_recon_mmd", mmd_results.cf_recon)
            # self.log(f"{Stage.fit}/enc/s0_dist_mmd", mmd_results.s0_dist)
            # self.log(f"{Stage.fit}/enc/s1_dist_mmd", mmd_results.s1_dist)

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_a, self.d_b], requires_grad=False)
            d_a_pred_fake_data = self.d_a(cyc_out.fake_a)
            d_b_pred_fake_data = self.d_b(cyc_out.fake_b)

            gen_loss = self.loss.get_gen_loss(
                real_a=real_a,
                real_b=real_b,
                gen_fwd=gen_fwd,
                d_a_pred_fake_data=d_a_pred_fake_data,
                d_b_pred_fake_data=d_b_pred_fake_data,
            )

            self.log(f"{Stage.fit}/enc/g_tot_loss", gen_loss.tot)
            self.log(f"{Stage.fit}/enc/g_A2B_loss", gen_loss.a2b)
            self.log(f"{Stage.fit}/enc/g_B2A_loss", gen_loss.b2a)
            self.log(f"{Stage.fit}/enc/cycle_loss", gen_loss.cycle_loss)

            return gen_loss.tot

        if optimizer_idx == 1:
            self.set_requires_grad([self.d_a], requires_grad=True)
            fake_a = self.fake_pool_a.push_and_pop(cyc_out.fake_a)
            dis_out = self.forward_dis(dis=self.d_a, real_data=real_a, fake_data=fake_a.detach())

            # GAN loss
            d_a_loss = self.loss.get_dis_loss(
                dis_pred_real_data=dis_out.real, dis_pred_fake_data=dis_out.fake
            )
            self.log(f"{Stage.fit}/enc/d_A_loss", d_a_loss)
            return d_a_loss

        if optimizer_idx == 2:
            self.set_requires_grad([self.d_b], requires_grad=True)
            fake_b = self.fake_pool_b.push_and_pop(cyc_out.fake_b)
            dis_b_out = self.forward_dis(dis=self.d_b, real_data=real_b, fake_data=fake_b.detach())

            # GAN loss
            d_b_loss = self.loss.get_dis_loss(
                dis_pred_real_data=dis_b_out.real, dis_pred_fake_data=dis_b_out.fake
            )
            self.log(f"{Stage.fit}/enc/d_B_loss", d_b_loss)
            return d_b_loss
        raise NotImplementedError("There should only be 3 optimizers.")

    def shared_step(self, batch: Batch | CfBatch | TernarySample, *, stage: Stage) -> SharedStepOut:
        real_a = batch.x
        real_b = batch.x

        cyc_out = self.forward(real_a=real_a, real_b=real_b)
        gen_fwd = self.forward_gen(
            real_a=real_a, real_b=real_b, fake_a=cyc_out.fake_a, fake_b=cyc_out.fake_b
        )

        dis_out_a = self.forward_dis(dis=self.d_a, real_data=real_a, fake_data=cyc_out.fake_a)
        dis_out_b = self.forward_dis(dis=self.d_b, real_data=real_b, fake_data=cyc_out.fake_b)

        # G_A2B loss, G_B2A loss, G loss
        gen_losses = self.loss.get_gen_loss(
            real_a=real_a,
            real_b=real_b,
            gen_fwd=gen_fwd,
            d_a_pred_fake_data=dis_out_a.fake,
            d_b_pred_fake_data=dis_out_b.fake,
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
            f"{stage}/enc/g_A2B_loss": gen_losses.a2b,
            f"{stage}/enc/g_B2A_loss": gen_losses.b2a,
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
            recon=self.invert(index_by_s([cyc_out.fake_a, cyc_out.fake_b], batch.s), batch.x),
            cf_pred=self.invert(index_by_s([cyc_out.fake_a, cyc_out.fake_b], 1 - batch.s), batch.x),
            recons_0=self.invert(cyc_out.fake_a, batch.x),
            recons_1=self.invert(cyc_out.fake_b, batch.x),
        )

    def test_epoch_end(self, outputs: list[SharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.test)

    def validation_epoch_end(self, outputs: list[SharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.validation)

    def shared_epoch_end(self, outputs: list[SharedStepOut], *, stage: Stage) -> None:
        self.all_x = torch.cat([_r.x for _r in outputs], 0)
        self.all_s = torch.cat([_r.s for _r in outputs], 0)
        self.all_recon = torch.cat([_r.recon for _r in outputs], 0)
        self.all_cf_pred = torch.cat([_r.recon for _r in outputs], 0)

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

    def validation_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> SharedStepOut:
        return self.shared_step(batch=batch, stage=Stage.validate)

    def test_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> SharedStepOut:
        return self.shared_step(batch=batch, stage=Stage.test)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[optim.lr_scheduler.ExponentialLR]]:

        # define the optimizers here
        g_opt = torch.optim.AdamW(self.g_params, lr=self.g_lr, weight_decay=self.g_weight_decay)
        d_a_opt = torch.optim.AdamW(self.d_a_params, lr=self.d_lr, weight_decay=self.d_weight_decay)
        d_b_opt = torch.optim.AdamW(self.d_b_params, lr=self.d_lr, weight_decay=self.d_weight_decay)

        # define the lr_schedulers here
        g_sch = optim.lr_scheduler.ExponentialLR(g_opt, gamma=self.scheduler_rate)
        d_a_sch = optim.lr_scheduler.ExponentialLR(d_a_opt, gamma=self.scheduler_rate)
        d_b_sch = optim.lr_scheduler.ExponentialLR(d_b_opt, gamma=self.scheduler_rate)

        # first return value is a list of optimizers and second is a list of lr_schedulers
        # (you can return empty list also)
        return [g_opt, d_a_opt, d_b_opt], [g_sch, d_a_sch, d_b_sch]

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

    @property
    def x(self) -> list[Tensor]:
        return [self.fake_a, self.fake_b]
