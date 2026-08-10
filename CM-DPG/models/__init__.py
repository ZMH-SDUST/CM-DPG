from groundingdino.models.GroundingDINO import build_groundingdino

from .registry import MODULE_BUILD_FUNCS


MODULE_BUILD_FUNCS.register(build_groundingdino, module_name="groundingdino", force=True)


def build_model(args):
    return build_groundingdino(args)
