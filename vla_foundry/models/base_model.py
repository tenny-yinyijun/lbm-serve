import logging

import torch
import torch.nn as nn

from vla_foundry.params.model_params import ModelParams


class BaseModelMeta(type):
    """Metaclass that automatically calls _post_init after __init__."""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        instance.__init__(*args, **kwargs)

        # Call _post_init if it exists
        if hasattr(instance, "_post_init"):
            instance._post_init()

        return instance


class BaseModel(nn.Module, metaclass=BaseModelMeta):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.model_params = model_params
        # Use object.__setattr__ to prevent ema_model from being registered as a submodule
        # This prevents it from being included in state_dict() during checkpoint saving
        object.__setattr__(self, "ema_model", None)

    def _post_init(self):
        """Called automatically after subclass initialization is complete."""
        if self.model_params.freeze:
            self.freeze_parameters()

    def freeze_parameters(self):
        """Freeze all parameters in the model by setting requires_grad to False."""
        for param in self.parameters():
            param.requires_grad = False

    def apply_torchcompile(self, device=None):
        """Walk the model tree and apply torch.compile to appropriate subtrees.

        The torchcompile parameter on ModelParams controls compilation:
        - True: compile this model (validates no descendant has False)
        - False: explicitly refuse compilation (error if ancestor has True)
        - None (default): don't compile on own, but can be compiled as part of parent

        When a model has torchcompile=True, it becomes a "compile root" - torch.compile
        is applied to it and its children are included in the compiled graph.
        When a model has torchcompile=None/False, its children are checked recursively.

        Returns True if this model itself should be compiled (caller must handle,
        since a model cannot replace itself in its parent).
        """
        tc = getattr(self.model_params, "torchcompile", None)

        if tc is True:
            self._validate_no_false_descendants()
            return True

        # tc is None or False: don't compile self, but check children for compile roots
        for name, child in self.named_children():
            if isinstance(child, BaseModel):
                should_compile_child = child.apply_torchcompile(device)
                if should_compile_child:
                    if device is not None:
                        from vla_foundry.distributed import move_buffers_to_device

                        move_buffers_to_device(child, device)
                    logging.info(f"Compiling '{name}' ({type(child).__name__}) with torch.compile()...")
                    setattr(self, name, torch.compile(child))

        return False

    def _validate_no_false_descendants(self):
        """Validate that no descendant BaseModel has torchcompile=False."""
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, BaseModel):
                child_tc = getattr(module.model_params, "torchcompile", None)
                if child_tc is False:
                    raise ValueError(
                        f"'{type(self).__name__}' has torchcompile=True but descendant "
                        f"'{type(module).__name__}' has torchcompile=False. Cannot compile "
                        f"a model when a sub-model explicitly refuses compilation."
                    )

    def set_ema_model(self, ema_model):
        """Set the EMA model for inference/evaluation.

        Args:
            ema_model: EMA model instance (e.g., from create_ema_model())
        """
        # Use object.__setattr__ to bypass nn.Module's __setattr__
        # This prevents the ema_model from being registered as a child module
        # and therefore prevents it from being included in state_dict()
        object.__setattr__(self, "ema_model", ema_model)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
