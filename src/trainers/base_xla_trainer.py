import torch

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import os
import numpy as np

import wandb
import huggingface_hub as hf

import utils.constants as constants
from utils.data_utils import DotDict
from utils.logging_utils import LogSection, log_print, log_master_print


class BaseXLATrainer:

    def __init__(
        self,
        project,
        name,
        config,
        debug=False
    ):
        self.project = project
        self.name = name
        self.config = config
        self.debug = debug

        save_name = f"{project}_{name}"
        self.save_repo = f"{constants.HF_ID}/{save_name}"

        if constants.XLA_MAIN() and not self.debug:
            with LogSection("Save Locations Creation"):
                hf.create_repo(
                    save_name, private=True, exist_ok=True
                )
                os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)
                wandb.init(
                    project=project,
                    name=name,
                    config=config
                )

        # apply hyperparams
        for k in config:
            setattr(self, k, config[k])

        # init log
        self.log = DotDict()


    def log_step(self):
        if not constants.XLA_MAIN() or self.debug:
            return
        
        # save and clear log
        wandb.log(self.log.to_dict())
        self.log = DotDict()


    @torch.no_grad()
    def save_checkpoint(
        self,
        models,
        step
    ):
        if not constants.XLA_MAIN() or self.debug:
            return
        with LogSection("Saving Checkpoint"):

            api = hf.HfApi()
            base_path = os.path.join(constants.LOCAL_DATA_PATH, f"{step:012d}")

            for name, tup in models.items():
                model, on_device = tup

                path = os.path.join(base_path, name)

                if on_device:
                    os.makedirs(path, exist_ok=True)
                    xm.save(model.state_dict(), os.path.join(path, "state_dict.pt"))
                    try:
                        model.config.save_pretrained(path, push_to_hub=False)
                    except:
                        print(f"Warning: {name} config not saved")
                        pass

                else:
                    model.save_pretrained(path, push_to_hub=False)

                api.upload_folder(
                    repo_id=self.save_repo,
                    folder_path=path,
                    path_in_repo=os.path.join(f"{step:012d}", name),
                    repo_type="model"
                )
    

    def _get_scheduler(self, optimizer):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        if self.lr_steps is None:
            return warmup_scheduler

        cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1e-10,
            total_iters=self.lr_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cooldown_scheduler],
            milestones=[self.warmup_steps]
        )


    def train(
        self,
        model,
        loader
    ):

        # init model
        for p in model.parameters():
            p.requires_grad = True
        model.train()

        # init training objs
        optimizer = syncfree.Adam(
            model.parameters(), lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        lr_scheduler = self._get_scheduler(optimizer)

        # loop
        curr_step = 0
        token_tracker = xm.RateTracker()
        step_tracker = xm.RateTracker()
        for epoch in self.num_epochs:
            for x, y in loader:
                assert x.shape[0] == y.shape[0], "x and y must have same batch size"
                
                # prepare x for accum
                n_x = x.shape[0]
                if n_x % self.mini_bs != 0:
                    print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
                if n_x * constants.NUM_XLA_DEVICES() != self.bs:
                    print(f"Warning: sample size {n_x} with {constants.NUM_XLA_DEVICES()} devices does not match batch size {self.bs}")
                x_split = torch.split(x, self.mini_bs, dim=0)
                y_split = torch.split(y, self.mini_bs, dim=0)

                # accumulate gradients
                results_accum = DotDict()
                for i_split in range(len(x_split)):
                    mini_x = x_split[i_split]
                    mini_y = y_split[i_split]

                    # get results from train step
                    with autocast(constants.XLA_DEVICE()):
                        results = self.train_step(model, mini_x, mini_y)

                        # scale results for accumulation
                        for k, v in results.items():
                            results[k] = v / (len(x_split) * constants.NUM_XLA_DEVICES())

                        # save results
                        with torch.no_grad():
                            for k, v in results.items():
                                if k not in results_accum:
                                    results_accum[k] = 0.0
                                results_accum[k] = results_accum[k] + v.detach()
                    
                    # mark step to save gradients
                    results.loss.backward()
                    if len(x_split) > 1:
                        xm.mark_step()

                # perform a single optimizer step
                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                
                # update lr
                self.log.lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                # tracking
                token_tracker.add(self.bs * x.shape[1])
                step_tracker.add(1)
                curr_step += 1

                def _post_step():

                    # log
                    for k, v in results_accum.items():
                        r = xm.mesh_reduce(f"{k}_reduce", v.item(), np.sum)
                        self.log[k] = r

                    # print update
                    msg = [
                        f"Epoch {epoch+1}/{self.num_epochs}",
                        f"Step {curr_step}",
                        f"LR = {self.log.lr:.2e}",
                        f"Loss = {self.log.loss:.4f}",
                        f"{step_tracker.rate():.2f} steps/s",
                        f"{round(3600*token_tracker.rate()):_} tok/h"
                    ]
                    log_master_print("{: >15}{: >20}{: >20}{: >20}{: >23}".format(*msg))
                
                    # save
                    self.log_step()
                
                xm.add_step_closure(_post_step)
        
            self.save_checkpoint(
                {
                    'model': (model, True),
                    'optimizer': (optimizer, True),
                },
                epoch+1
            )
    

    def train_step(
        self,
        model,
        x,
        y
    ):
        """ Get results of a single training step.
         - Must return DotDict of results
         - Results must include 'loss' key

        Args:
            model: model to train
            x: input data [mini_bs, ...]
            y: target data [mini_bs]

        Returns:
            DotDict: result tensors containin 'loss' key
        """
        raise NotImplementedError("train_step must be implemented in child class!")