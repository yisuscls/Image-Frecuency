
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image
import os
#************************************************************************************************
# 2. Low pass filtering in frequency domain
#************************************************************************************************

def create_gaussian_filter(size, sigma, center=None):
    if center is None:
        center = (size[0] // 2, size[1] // 2)
    x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    x -= center[1]
    y -= center[0]
    gaussian = np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    return gaussian / gaussian.max()
def filter_image(image_path, sigma_values):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    fft_image = fftshift(fft2(image_array))
    
    cols = 3  # Número de columnas en el subplot
    rows = 2  # Fijamos las filas en 2 según la solicitud

    plt.figure(figsize=(15, 10))  # Ajusta el tamaño de la figura para adaptarse a 3 columnas y 2 filas
    
    # Mostrar la imagen original
    plt.subplot(rows, cols, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plot_position = 2  # Posición inicial para la primera imagen filtrada
    
    for sigma in sigma_values:
        gaussian_filter = create_gaussian_filter(image_array.shape, sigma)
        filtered_fft = fft_image * gaussian_filter
        filtered_image = np.abs(ifft2(ifftshift(filtered_fft)))
        
        if plot_position <= rows * cols:  # Asegurar que no excedemos el número total de subplots
            plt.subplot(rows, cols, plot_position)
            plt.imshow(filtered_image, cmap='gray')
            plt.title(f'Filtered σ={sigma}')
            plt.axis('off')
            plot_position += 1
    
    # Si hay espacio, mostrar el último filtro gaussiano
    if plot_position <= rows * cols:
        plt.subplot(rows, cols, plot_position)
        plt.imshow(gaussian_filter, cmap='gray')
        plt.title('Gaussian Filter')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plot_3d_filter(filter, title='3D View of Filter'):
    x, y = np.meshgrid(np.arange(filter.shape[1]), np.arange(filter.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, filter, cmap='viridis')
    ax.set_title(title)
    plt.savefig("./output_images/ex2_3d_filter.png")
    plt.show()

def save_filter_image(image_path, sigma_values):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    fft_image = fftshift(fft2(image_array))
    
    # Guardar la imagen original
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig(f'./output_images/ex2_original_image.png')
    plt.close()
    
    plot_position = 1  # Contador para el nombre del archivo
    
    for sigma in sigma_values:
        gaussian_filter = create_gaussian_filter(image_array.shape, sigma)
        filtered_fft = fft_image * gaussian_filter
        filtered_image = np.abs(ifft2(ifftshift(filtered_fft)))
        
        # Guardar cada imagen filtrada
        plt.figure(figsize=(5, 5))
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtered σ={sigma}')
        plt.axis('off')
        plt.savefig(f'./output_images/ex2_filtered_image_{plot_position}_sigma_{sigma}.png')
        plt.close()
        
        plot_position += 1
    
    # Guardar el último filtro gaussiano utilizado
    plt.figure(figsize=(5, 5))
    plt.imshow(gaussian_filter, cmap='gray')
    plt.title('Gaussian Filter')
    plt.axis('off')
    plt.savefig(f'./output_images/ex2_gaussian_filter.png')
    plt.close()

# Uso de las funciones definidas
path = './DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0417(a)(barbara).tif' # Actualiza esto con la ruta correcta a tu imagen
sigma_values = [10, 30, 50]
filter_image(path, sigma_values)
save_filter_image(path, sigma_values)

# Genera y muestra un filtro gaussiano en 3D
gaussian_filter_3d = create_gaussian_filter((50, 50), 10)  # Ajusta el tamaño y sigma según necesites
plot_3d_filter(gaussian_filter_3d, '3D View of Gaussian Filter')
