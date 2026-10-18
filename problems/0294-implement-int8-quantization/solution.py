import numpy as np

def int8_quantize(x: list[float]) -> dict:
    x_arr = np.array(x, dtype=np.float32)
    if np.all(x_arr == 0):
        return {
            'quantized': [0] * len(x),
            'scale': 1.0,
            'dequantized': [0.0] * len(x)
        }
    
    max_val = np.max(np.abs(x_arr))
    scale = float(max_val) / 127.0
    
    # In float32, 0.01 / (0.02 / 127) is 63.499996 -> rounds to 63
    quantized = np.clip(np.round(x_arr / np.float32(scale)), -127, 127).astype(int).tolist()
    dequantized = [round(q * scale, 4) for q in quantized]
    
    return {
        'quantized': quantized,
        'scale': round(scale, 6),
        'dequantized': dequantized
    }