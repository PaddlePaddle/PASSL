from ..utils.registry import Registry, build_from_config

HOOKS = Registry("HOOK")


def build_hook(cfg):
    return build_from_config(cfg, HOOKS)
