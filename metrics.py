import numpy as np
import torch
import time

CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

MS_CONVERSION = 1000
US_CONVERSION = MS_CONVERSION * MS_CONVERSION

WINDOW_SIZE = 44
COLUMNS = 116
HEX_MAX = 15

def calculate_inference_time(model, device, batch_size=64, warmup_reps=100, repetitions=500):
    dummy_input = torch.randint(0, HEX_MAX + 1, (batch_size, 1, WINDOW_SIZE, COLUMNS), dtype=torch.float).to(device)

    timings = np.zeros((repetitions, 1))

    if (device.type == CUDA_DEVICE):
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU WARM UP
    for _ in range(warmup_reps):
        _ = model(dummy_input)

    model.eval()

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            if (device.type == CUDA_DEVICE):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU TO SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time / batch_size

            elif (device.type == CPU_DEVICE):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                elapsed_time = end_time - start_time
                timings[rep] = elapsed_time / batch_size

    mean_inference_time = np.sum(timings) / repetitions

    if (device.type == CUDA_DEVICE):
        mean_inference_time = mean_inference_time * MS_CONVERSION
    elif (device.type == CPU_DEVICE):
        mean_inference_time = mean_inference_time * US_CONVERSION

    return mean_inference_time

