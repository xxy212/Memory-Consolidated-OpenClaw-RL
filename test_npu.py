import os
import time
import gc

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as e:
    raise RuntimeError(
        "未找到 torch_npu，请先确认当前环境是昇腾 NPU + PyTorch torch_npu 环境"
    ) from e


def bytes_to_human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{n} B"


def get_npu_device(device_id: int = 0) -> str:
    return f"npu:{device_id}"


def empty_cache():
    gc.collect()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    if hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def get_mem_info(device_id: int = 0):
    """
    返回 (free_bytes, total_bytes)
    部分版本 torch_npu 可能没有 mem_get_info，这里做兼容。
    """
    if hasattr(torch.npu, "mem_get_info"):
        free_bytes, total_bytes = torch.npu.mem_get_info(device_id)
        return int(free_bytes), int(total_bytes)
    return None, None


def allocate_chunk(device: str, size_mb: int, dtype: torch.dtype):
    """
    按指定 MB 分配 1D tensor。
    """
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    numel = (size_mb * 1024 * 1024) // bytes_per_elem
    if numel <= 0:
        raise ValueError("size_mb 太小，无法分配有效 tensor")
    x = torch.empty(numel, dtype=dtype, device=device)
    # 写入一次，确保真正占用
    x.fill_(1)
    return x


def test_npu_memory_limit(
    device_id: int = 0,
    dtype: torch.dtype = torch.float32,
    coarse_step_mb: int = 512,
    fine_step_mb: int = 64,
    max_chunks: int = 100000,
    hold_refs: bool = True,
):
    """
    粗粒度 + 细粒度两阶段逼近可分配上限。

    参数:
        device_id: NPU 编号
        dtype: 数据类型，float32 / float16 等
        coarse_step_mb: 粗测步长
        fine_step_mb: 细测步长
        max_chunks: 最大分配块数，防止死循环
        hold_refs: 是否持有已分配对象；测试上限时应为 True
    """
    device = get_npu_device(device_id)
    torch.npu.set_device(device)

    print(f"使用设备: {device}")
    print(f"dtype: {dtype}")
    print(f"粗测步长: {coarse_step_mb} MB, 细测步长: {fine_step_mb} MB")

    free_bytes, total_bytes = get_mem_info(device_id)
    if free_bytes is not None:
        print(f"初始 free:  {bytes_to_human(free_bytes)}")
        print(f"总显存 total: {bytes_to_human(total_bytes)}")
    else:
        print("当前 torch_npu 版本不支持 mem_get_info，跳过 free/total 查询")

    empty_cache()

    allocated_tensors = []
    total_allocated_bytes = 0

    def try_alloc(step_mb: int) -> bool:
        nonlocal total_allocated_bytes
        try:
            t = allocate_chunk(device, step_mb, dtype)
            if hold_refs:
                allocated_tensors.append(t)
            total_allocated_bytes += t.numel() * t.element_size()
            if hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
            print(
                f"[OK] +{step_mb} MB, 当前累计分配: {bytes_to_human(total_allocated_bytes)}"
            )
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg:
                print(f"[OOM] 再分配 {step_mb} MB 失败")
                return False
            raise

    # 第一阶段：粗测
    coarse_success = 0
    for _ in range(max_chunks):
        if try_alloc(coarse_step_mb):
            coarse_success += 1
        else:
            break

    # 第二阶段：细测
    fine_success = 0
    for _ in range(max_chunks):
        if try_alloc(fine_step_mb):
            fine_success += 1
        else:
            break

    print("\n===== 测试结果 =====")
    print(f"粗测成功次数: {coarse_success}")
    print(f"细测成功次数: {fine_success}")
    print(f"近似最大可分配容量: {bytes_to_human(total_allocated_bytes)}")

    free_bytes_after, total_bytes_after = get_mem_info(device_id)
    if free_bytes_after is not None:
        print(f"测试后 free:  {bytes_to_human(free_bytes_after)}")
        print(f"测试后 total: {bytes_to_human(total_bytes_after)}")

    return total_allocated_bytes, allocated_tensors


if __name__ == "__main__":
    # 可按需修改
    DEVICE_ID = 0
    DTYPE = torch.float32
    COARSE_STEP_MB = 512
    FINE_STEP_MB = 64

    try:
        total_bytes, tensors = test_npu_memory_limit(
            device_id=DEVICE_ID,
            dtype=DTYPE,
            coarse_step_mb=COARSE_STEP_MB,
            fine_step_mb=FINE_STEP_MB,
        )
    finally:
        # 释放
        tensors = None
        empty_cache()
        time.sleep(1)
        print("已释放测试分配的张量")
