__version__ = '1.0.0'
__author__ = 'PengXiang Wang'
__all__ = ['pde', 'cfd', 'interface', 'common', 'solver']

# 其他必要的初始化代码
def version():
    """返回软件版本信息"""
    return __version__

def import_all():
    """导入所有子模块"""
    from . import pde   
    from . import cfd
    from . import interface
    from . import common
    from . import solver
    return locals()
