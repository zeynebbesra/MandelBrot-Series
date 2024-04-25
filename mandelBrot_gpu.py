import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import time

# Seri Hesaplama
def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def mandelbrot_serial(width, height, max_iter):
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

# Seri hesaplama süresini ölçmek için zaman ölçümünü fonksiyon çağrısı etrafına yerleştirin
# start_time_seri = time.time()
# output_seri = mandelbrot_serial(1600, 1600, 100)  # Örnek olarak 1600x1600 ve 100 iterasyon kullanıldı
# end_time_seri = time.time()
# seri_sure = end_time_seri - start_time_seri

start_time_seri = time.perf_counter()
output_seri = mandelbrot_serial(1600,1600,100)
end_time_seri = time.perf_counter()
seri_sure = end_time_seri-start_time_seri

print(f"Seri sure: {seri_sure:.2f} saniye")

#  Paralel hesaplama
# start_time_paralel = time.time()
start_time_paralel = time.perf_counter()

kernel_code = """
__kernel void mandelbrot_set(__global float2 *input, __global int *output, const unsigned int max_iter) {
    int gid = get_global_id(0);
    float2 c = input[gid];
    float2 z;
    z.x = c.x;
    z.y = c.y;
    int n = 0;

    while ((z.x * z.x + z.y * z.y) < 4.0f && n < max_iter) {
        float x_new = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0f * z.x * z.y + c.y;
        z.x = x_new;
        n++;
    }
    output[gid] = n;
}
"""

# OpenCL için context (bağlam) ve queue (kuyruk) oluşturma
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Mandelbrot hesaplaması için OpenCL kernel programını derleme
program = cl.Program(context, kernel_code).build()

# Mandelbrot seti hesaplaması için girdi ve çıktı verilerini hazırlama
width, height = 1600, 1600
real = np.linspace(-2, 2, width, dtype=np.float32)
imag = np.linspace(-2, 2, height, dtype=np.float32)
real_grid, imag_grid = np.meshgrid(real, imag)
complex_grid = np.c_[real_grid.ravel(), imag_grid.ravel()]

mf = cl.mem_flags

# Çıktı için buffer oluşturma
output = np.empty(width * height, dtype=np.int32)
output_buf = cl.Buffer(context, mf.WRITE_ONLY, output.nbytes)

# Kernel'i çalıştırma ve çıktıyı okuma
max_iter = 100
global_size = (width * height,)
local_size = None
c_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=complex_grid)
program.mandelbrot_set(queue, global_size, local_size, c_buf, output_buf, np.uint32(max_iter))
cl.enqueue_copy(queue, output, output_buf).wait()

# Çıktı dizisini yeniden şekillendirme ve Mandelbrot setini görselleştirme
output_image = output.reshape((height, width))
plt.imshow(output_image, extent=(-2, 2, -2, 2))
plt.show(block=False)

# end_time_paralel = time.time()
end_time_paralel = time.perf_counter()
paralel_sure = end_time_paralel - start_time_paralel

print(f"Paralel sure: {paralel_sure:.2f} saniye")

# Hızlanma ve verimlilik hesaplamaları
speedup = seri_sure / paralel_sure
efficiency = speedup / 1*100 # GPU için genellikle 1 olarak alınır çünkü tek bir cihaz üzerinde çalışır.

print(f"Speed-up: {speedup:.2f}")
print(f"Verimlilik: %{efficiency:.2f}")

# Hızlanma ve verimlilik metriklerini görselleştirmek
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Seri', 'Paralel'], [seri_sure, paralel_sure], color=['red', 'green'])
plt.ylabel('Süre (saniye)')
plt.title('Hesaplama Süreleri')

plt.subplot(1, 2, 2)
plt.bar(['Hizlanma', 'Verimlilik'], [speedup, efficiency], color=['blue', 'orange'])
plt.ylabel('Değer')
plt.title('Performans Metrikleri')

plt.tight_layout()
plt.show()