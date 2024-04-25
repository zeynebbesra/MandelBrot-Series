import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import time

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def compute_mandelbrot(queue, re_min, re_max, im_min, im_max, width, height, max_iter):
    local_real_axis = np.linspace(re_min, re_max, width)
    local_imaginary_axis = np.linspace(im_min, im_max, height)
    local_grid = np.array([[complex(r, i) for r in local_real_axis] for i in local_imaginary_axis])

    local_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            c = local_grid[i, j]
            local_image[i, j] = mandelbrot(c, max_iter)

    queue.put((im_min, im_max, local_image))

def compute_chunk(index, num_processes, width, height):
    strip_height = height // num_processes
    im_min = -2 + index * strip_height * (4 / height)
    im_max = -2 + (index + 1) * strip_height * (4 / height)
    re_min, re_max = -2, 2
    return re_min, re_max, im_min, im_max, width, strip_height

def compute_mandelbrot_parallel(num_processes, width, height, max_iter):
    queue = Queue()
    processes = []
    for i in range(num_processes):
        re_min, re_max, im_min, im_max, width, strip_height = compute_chunk(i, num_processes, width, height)
        p = Process(target=compute_mandelbrot, args=(queue, re_min, re_max, im_min, im_max, width, strip_height, max_iter))
        processes.append(p)
        p.start()

    full_image = np.zeros((height, width))
    for _ in processes:
        im_min, im_max, local_image = queue.get()
        strip_index = int((im_min + 2) / (4 / height))
        full_image[strip_index:strip_index + local_image.shape[0], :] = local_image

    for p in processes:
        p.join()

    return full_image

def compute_mandelbrot_serial(width, height, max_iter):
    re_min, re_max = -2, 2
    im_min, im_max = -2, 2
    real_axis = np.linspace(re_min, re_max, width)
    imaginary_axis = np.linspace(im_min, im_max, height)
    image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            c = complex(real_axis[j], imaginary_axis[i])
            image[i, j] = mandelbrot(c, max_iter)
    return image

if __name__ == '__main__':
    width, height = 800, 800
    max_iter = 100
    speed_ups = []
    efficiencies = []

    # Seri süre ölçümü
    start_time_serial = time.time()
    image_serial = compute_mandelbrot_serial(width, height, max_iter)
    elapsed_time_serial = time.time() - start_time_serial

    # Farklı işlemci sayıları için paralel hesaplama ve metriklerin hesaplanması
    for num_processes in [1, 2, 3, 4]:
        start_time_parallel = time.time()
        image_parallel = compute_mandelbrot_parallel(num_processes, width, height, max_iter)
        elapsed_time_parallel = time.time() - start_time_parallel
        speed_up = elapsed_time_serial / elapsed_time_parallel
        efficiency = speed_up / num_processes
        speed_ups.append(speed_up)
        efficiencies.append(efficiency)
        print(f"{num_processes} islemci ile sure: {elapsed_time_parallel:.2f} saniye, "
              f"Speed-up: {speed_up:.2f}, Efficiency: %{efficiency * 100:.2f}")


    # Hızlanma ve verimlilik grafiği
    plt.figure(figsize=(10, 5))
    plt.plot([1, 2, 3, 4], speed_ups, 'o-', label='Hizlanma (Speed-up)')
    plt.plot([1, 2, 3, 4], efficiencies, 'o-', label='Verimlilik (Efficiency)')
    plt.xlabel('İşlemci Sayisi')
    plt.ylabel('Degerler')
    plt.title('Mandelbrot Hesaplama Performans Karsilastirmasi')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # MandelBrot grafiği
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_serial, extent=(-2, 2, -2, 2))
    plt.title("Seri Hesaplama")
    plt.gray()

    plt.subplot(1, 2, 2)
    plt.imshow(image_parallel, extent=(-2, 2, -2, 2))
    plt.title("Paralel Hesaplama")
    plt.gray()
    plt.show()



