import os
from os.path import join as pjoin
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from icecream import ic


def find_in_path(name, path):
    """Найти файл в пути поиска"""

    # Адаптировано с http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """
    Найдите среду CUDA в системе
    Возвращает словарь с ключами «home», «nvcc», «include» и «lib64».
    и значения, дающие абсолютный путь к каждому каталогу.
    Начинается с поиска переменной env CUDAHOME. Если не найдено,
    все основано на поиске «nvcc» в PATH.
    """

    # Сначала проверьте, используется ли переменная env CUDAHOME.
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        # В противном случае найдите PATH для NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib', 'x64')}

    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()
ic(CUDA)

ext = Extension('cudaext',
                sources=['wrapper.pyx'],
                libraries=['lib/kernel', 'cudart'],
                language='c++',
                include_dirs=[CUDA['include']],
                library_dirs=[CUDA['lib64']],
                extra_compile_args=['/openmp']
                )

setup(
    name='gpuadder',
    include_dirs=[CUDA['include']],
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext},
)
