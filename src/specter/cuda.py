import gc

import torch


def clear_memory(*names):
    torch.cuda.synchronize()

    before_alloc = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()

    # delete requested globals
    for name in names:
        if name in globals():
            del globals()[name]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after_alloc = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()

    freed_alloc = (before_alloc - after_alloc) / 1024**2
    freed_reserved = (before_reserved - after_reserved) / 1024**2

    print(f"Freed {freed_alloc:.2f} MB (allocated)")
    print(f"Freed {freed_reserved:.2f} MB (reserved cache)")
    print(f"Now allocated: {after_alloc / 1024**2:.2f} MB")
