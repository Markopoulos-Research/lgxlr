import copy
import math
import torch


class LGXLR:
    def __init__(self, model, optimizer, initial_lr_bounds=(-5, -1), step_size=1, tolerance=1e-4, max_overfit_count=2, min_lr_exp=-5, verbose=False):
        self.model = model
        self.optimizer = optimizer

        # Constants
        self.Lo = initial_lr_bounds[0]
        self.Uo = initial_lr_bounds[1]
        self.So = step_size
        self.H = tolerance
        self.Cmax = max_overfit_count

        # State variables
        self.L = self.Lo
        self.U = self.Uo
        self.S = self.So
        self.C = 0  # overfitting counter
        self.verbose = verbose

        # Epoch state
        self.prev_train_loss = None
        self.prev_val_loss = None
        self.prev_model_state = copy.deepcopy(model.state_dict())

        self.set_lr(self.L)

    def set_lr(self, log_lr):
        lr = 10 ** log_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if self.verbose:
            print(f"[LG-XLR] Set LR to {lr:.2e} (10^{log_lr:.2f})")

    def step(self, train_loss, val_loss):
        if self.prev_train_loss is None:
            # First epoch, just store losses
            self.prev_train_loss = train_loss
            self.prev_val_loss = val_loss
            self.prev_model_state = copy.deepcopy(self.model.state_dict())
            return

        delta_tr = (train_loss / self.prev_train_loss) - 1
        delta_val = (val_loss / self.prev_val_loss) - 1

        if self.verbose:
            print(f"[LG-XLR] ΔTrain: {delta_tr:.2e}, ΔVal: {delta_val:.2e}, L: {self.L:.2f}, U: {self.U:.2f}, S: {self.S:.2f}, C: {self.C}")

        # Case 1: Progress phase
        if delta_tr <= -self.H and delta_val <= -self.H:
            self.L = min(self.L + self.S, self.U)
            self.C = 0

        # Case 2: Divergence phase
        elif delta_tr >= self.H and delta_val >= self.H:
            self.U = self.L
            self.L = max(self.L - self.S, self.Lo)
            self.S = self.S / 10
            self.C = 0
            self.model.load_state_dict(self.prev_model_state)  # revert model

        # Case 3: Overfitting phase
        elif delta_tr <= -self.H and delta_val >= self.H:
            self.C += 1
            if self.C > self.Cmax:
                self.U = self.L
                self.L = max(self.L - self.So, self.Lo)
                self.S = self.S / 10
                self.C = 0
                self.model.load_state_dict(self.prev_model_state)

        # Else: no update to L, U

        # Set new learning rate
        self.set_lr(self.L)

        # Update previous epoch state
        self.prev_train_loss = train_loss
        self.prev_val_loss = val_loss
        self.prev_model_state = copy.deepcopy(self.model.state_dict())

        # Early stop condition (optional for users to check externally)
        if self.L == self.U == self.Lo:
            if self.verbose:
                print("[LG-XLR] Termination condition met: L = U = Lo")

    def get_lr(self):
        return 10 ** self.L

