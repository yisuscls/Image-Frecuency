from scipy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#************************************************************************************************
# 3. High pass filtering in frequency domain
#************************************************************************************************

def generate_highpass_butterworth_filter(cutoff_frequency, dimensions, filter_order=1):
    """Generate a Butterworth highpass filter."""
    horizontal_freq = np.arange(-dimensions[1]//2, dimensions[1]//2)
    vertical_freq = np.arange(-dimensions[0]//2, dimensions[0]//2)
    horizontal_freq, vertical_freq = np.meshgrid(horizontal_freq, vertical_freq)
    frequency_distance = np.sqrt(horizontal_freq**2 + vertical_freq**2)
    filter_response = 1 / (1 + (frequency_distance / cutoff_frequency)**(2 * filter_order))
    return 1 - filter_response  # Convert to highpass by subtraction


path = './DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0417(a)(barbara).tif'  # path to your image file
cutoff_freq = 30  # Define the cutoff frequency
filter_order = 2  # Define the order of the filter

"""Apply a Butterworth highpass filter to an image."""
original_image = Image.open(path).convert('L')
image_data = np.asarray(original_image)

fft_transformed_image = fftshift(fft2(image_data))
highpass_filter = generate_highpass_butterworth_filter(cutoff_freq, image_data.shape, filter_order)


plt.figure(figsize=(6, 6))
plt.imshow(highpass_filter, cmap='gray')
plt.title(f'Butterworth Highpass Filter (Cutoff={cutoff_freq}, Order={filter_order})')
plt.colorbar()
plt.savefig(f'./output_images/ex3_butterworth.png')
plt.show()

filtered_fft_image = fft_transformed_image * highpass_filter
filtered_image = np.abs(ifft2(ifftshift(filtered_fft_image)))

# Enhance the original image with the highpass filtered image
original_img_data = np.asarray(Image.open(path).convert('L'))

"""Enhance the original image by adding the highpass filtered image to it."""
filtered_normalized = (filtered_image / filtered_image.max()) * 255
enhanced_image_data = original_img_data + filtered_normalized
enhanced_image_data = np.clip(enhanced_image_data, 0, 255).astype(np.uint8)

plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(original_img_data, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(filtered_image, cmap='gray'), plt.title('Highpass Filtered Image')
plt.subplot(133), plt.imshow(enhanced_image_data, cmap='gray'), plt.title('Enhanced Image')
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(original_img_data, cmap='gray'), plt.title('Original Image')
plt.axis('off')
plt.savefig('./output_images/ex3_original.png') 
plt.close()

plt.figure(figsize=(5, 5))
plt.imshow(filtered_image, cmap='gray'), plt.title('Highpass Filtered Image')
plt.axis('off')
plt.savefig('./output_images/ex3_highpass_filter.png') 
plt.close()

plt.figure(figsize=(5, 5))
plt.imshow(enhanced_image_data, cmap='gray'), plt.title('Enhanced Image')
plt.axis('off')
plt.savefig('./output_images/ex3_enhanced_image.png') 
plt.close()