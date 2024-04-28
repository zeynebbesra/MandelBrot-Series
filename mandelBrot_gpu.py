import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import time

# Seri Hesaplama Fonksiyonu
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

# Kernel kodu
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

def compute_parallel(width, height, max_iter):
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    program = cl.Program(context, kernel_code).build()

    real = np.linspace(-2, 2, width, dtype=np.float32)
    imag = np.linspace(-2, 2, height, dtype=np.float32)
    real_grid, imag_grid = np.meshgrid(real, imag)
    complex_grid = np.c_[real_grid.ravel(), imag_grid.ravel()]

    mf = cl.mem_flags
    output = np.empty(width * height, dtype=np.int32)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, output.nbytes)

    c_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=complex_grid)
    program.mandelbrot_set(queue, (width * height,), None, c_buf, output_buf, np.uint32(max_iter))
    cl.enqueue_copy(queue, output, output_buf).wait()

    return output.reshape((height, width))

dimensions = [800, 1000, 1200, 1400, 1600]  # Different dimensions to test
max_iter = 100
speed_ups = []
efficiencies = []
compute_units = 16  # Number of compute units in your GPU

for dim in dimensions:
    width = height = dim

    # Measure serial computation time
    start_time_serial = time.perf_counter()
    mandelbrot_serial(width, height, max_iter)
    end_time_serial = time.perf_counter()
    serial_time = end_time_serial - start_time_serial

    # Measure parallel computation time
    start_time_parallel = time.perf_counter()
    compute_parallel(width, height, max_iter)
    end_time_parallel = time.perf_counter()
    parallel_time = end_time_parallel - start_time_parallel

    # Calculate speed-up and efficiency
    speed_up = serial_time / parallel_time
    efficiency = (speed_up / compute_units) * 100

    speed_ups.append(speed_up)
    efficiencies.append(efficiency)

    print(f"Dimensions: {dim}x{dim}, Serial Time: {serial_time:.2f}, Parallel Time: {parallel_time:.2f}, Speed-up: {speed_up:.2f}, Efficiency: {efficiency:.2f}%")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(dimensions, speed_ups, label='Speed-up', marker='o')
plt.plot(dimensions, efficiencies, label='Efficiency', marker='o')
plt.xlabel('Dimension (pixels)')
plt.ylabel('Performance Metric')
plt.title('Performance Metrics across Different Dimensions')
plt.legend()
plt.grid(True)
plt.show()
