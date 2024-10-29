import numpy as np

def choose_backend(use_gpu=False):
    global np
    if use_gpu:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                np = cp
                np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

                print('\033[92m' + '-' * 60 + '\033[0m')
                print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
                print('\033[92m' + '-' * 60 + '\033[0m\n')
            else:
                print("No GPU detected. Falling back to NumPy.")
        except ImportError:
            print("CuPy not installed. Falling back to NumPy.")