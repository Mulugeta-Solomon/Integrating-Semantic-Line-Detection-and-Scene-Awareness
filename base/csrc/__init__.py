from torch.utils.cpp_extension import load 
import glob
import os.path as osp

__this__ = osp.dirname(__file__)

_C = load(name='_C',sources=[osp.join(__this__,'binding.cpp'), osp.join(__this__,'linesegment.cu')], verbose=True)

__all__ = ["_C"]