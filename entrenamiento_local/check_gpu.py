# ============================================================
# ExpoEscom - Verificación rápida de GPU/entorno
#   python check_gpu.py
# ============================================================
import torch

print("=" * 50)
print("VERIFICACIÓN DE ENTORNO")
print("=" * 50)
print(f"PyTorch         : {torch.__version__}")
print(f"CUDA disponible : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA (build)    : {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU             : {props.name}")
    print(f"VRAM            : {props.total_memory / 1e9:.1f} GB")
    print(f"Capability      : {props.major}.{props.minor}  "
          f"({'Ampere+ (TF32 ok)' if props.major >= 8 else 'Turing/anterior (sin TF32, AMP fp16 sí)'})")
    # prueba mínima de cómputo
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    for _ in range(10):
        x = x @ x
    torch.cuda.synchronize()
    print(f"Matmul test     : OK ({(time.time()-t0)*1000:.0f} ms / 10 iter)")
else:
    print("\n⚠️  No hay GPU CUDA. Revisa drivers o instala torch con CUDA:")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
