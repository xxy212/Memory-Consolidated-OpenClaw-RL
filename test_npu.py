import os
import time
import signal
import sys
import multiprocessing as mp

import torch
import torch_npu


# 每张卡占用 30 GiB
BYTES_PER_DEVICE = 30 * 1024**3
NUM_DEVICES = 8


def hold_tensor_on_npu(device_id: int, size_bytes: int):
    """
    在指定 NPU 上分配一个 uint8 张量，占住 size_bytes 字节。
    uint8 每个元素 1 字节，方便精确控制占用大小。
    """
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    print(f"[device {device_id}] allocating {size_bytes / 1024**3:.2f} GiB on {device} ...", flush=True)

    try:
        # 1D uint8 tensor，元素数 = 字节数
        x = torch.empty(size_bytes, dtype=torch.uint8, device=device)

        # 写一遍，确保真正触发物理分配/提交
        x.fill_(1)

        # 可选：打印当前设备内存信息
        allocated = torch.npu.memory_allocated(device) / 1024**3
        reserved = torch.npu.memory_reserved(device) / 1024**3
        print(
            f"[device {device_id}] allocation done. "
            f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB",
            flush=True,
        )

        # 持有张量，直到进程被终止
        while True:
            time.sleep(60)

    except RuntimeError as e:
        print(f"[device {device_id}] RuntimeError: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[device {device_id}] Exception: {e}", flush=True)
        sys.exit(1)


def main():
    mp.set_start_method("spawn", force=True)

    procs = []
    for i in range(NUM_DEVICES):
        p = mp.Process(target=hold_tensor_on_npu, args=(i, BYTES_PER_DEVICE), daemon=False)
        p.start()
        procs.append(p)

    def cleanup(signum, frame):
        print("\nStopping all child processes...", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        print("All processes stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("All allocation processes started. Press Ctrl+C to release memory.", flush=True)

    # 主进程等待子进程
    while True:
        alive = [p.is_alive() for p in procs]
        if not all(alive):
            print("Some child process exited unexpectedly.", flush=True)
            cleanup(None, None)
        time.sleep(5)


if __name__ == "__main__":
    main()
