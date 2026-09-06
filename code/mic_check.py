import numpy as np
import sounddevice as sd
from config import cfg

SR = cfg.audio.sample_rate


def main():
    chunk = int(SR * 0.1)  # 100 мс
    print("Говори в микрофон и смотри на значения. Ctrl+C — стоп.\n")
    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        blocksize=chunk) as stream:
        while True:
            data, _ = stream.read(chunk)
            rms = float(np.sqrt(np.mean(data ** 2)))
            bar = '#' * int(min(rms * 200, 50))
            print(f"\rRMS: {rms:.4f} {bar:<50}", end='', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nОстановлено.')