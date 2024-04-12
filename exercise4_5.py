import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image


#************************************************************************************************
# 4. Periodic noise reduction using notch filter
#************************************************************************************************

path = './DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0417(a)(barbara).tif'
grayscale_image = Image.open(path).convert('L')
grayscale_image = grayscale_image.resize((512, 512))

image_array = np.array(grayscale_image, dtype=np.float32)

# Add sinusoidal noise to the image
height, width = image_array.shape
vertical_indices, horizontal_indices = np.ogrid[:height, :width]
freq_multiplier_x, freq_multiplier_y = height / 2, 1
freq_x, freq_y = freq_multiplier_x * 2 * np.pi, freq_multiplier_y * 2 * np.pi
noise_amplitude = 50

sinusoidal_noise = noise_amplitude * np.sin(freq_x * horizontal_indices / width + freq_y * vertical_indices / height)
image_with_noise = image_array + sinusoidal_noise
image_with_noise_uint8 = np.clip(image_with_noise, 0, 255).astype(np.uint8)

# Perform Fourier transformation and calculate the spectrum
fourier_transformed = fft2(image_with_noise_uint8)
centered_fourier = fftshift(fourier_transformed)
log_spectrum = np.log(np.abs(centered_fourier) + 1)

# Create and apply a notch filter
notch_radius = 10
x_grid, y_grid = np.meshgrid(np.linspace(-width//2, width//2-1, width), np.linspace(-height//2, height//2-1, height))
notch_pass_filter = np.ones((height, width), dtype=np.float32)
notch_pass_filter[np.sqrt((x_grid + freq_multiplier_x)**2 + (y_grid + freq_multiplier_y)**2) < notch_radius] = 0
notch_pass_filter[np.sqrt((x_grid - freq_multiplier_x)**2 + (y_grid - freq_multiplier_y)**2) < notch_radius] = 0
filtered_fft = centered_fourier * notch_pass_filter
image_after_filtering = np.real(ifft2(ifftshift(filtered_fft)))
image_after_filtering_uint8 = np.clip(image_after_filtering, 0, 255).astype(np.uint8)


#************************************************************************************************
# 5. Periodic noise filtering using band reject filter 
#************************************************************************************************
def band_reject_filter(shape, d0, w, order=1):
    m, n = np.meshgrid(np.linspace(-shape[0]//2, shape[0]//2, shape[0]),
    np.linspace(-shape[1]//2, shape[1]//2, shape[1]), indexing='ij')
    d = np.sqrt(m**2 + n**2)
    filter_mask = 1 / (1 + ((d*w)/(d**2-d0**2))**(2*order))
    return filter_mask

def compute_SNR(original, noisy):
    signal_power = np.mean(original**2)
    noise_power = np.mean((original- noisy)**2)
    return 10 * np.log10(signal_power / noise_power)

d0 = 30 # Cut-off frequency
w = 10 # Width of the band
# Apply band reject filter
br_filter = band_reject_filter(image_with_noise.shape, d0, w)
br_filtered_f = fftshift(fft2(image_with_noise)) * br_filter
br_filtered_img = np.real(ifft2(ifftshift(br_filtered_f)))
br_filtered_img_uint8 = np.clip(br_filtered_img, 0, 255).astype(np.uint8)
fig = plt.figure(figsize=(20, 10))
# Display 
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(image_array, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(image_with_noise_uint8, cmap='gray')
ax2.set_title('Noisy Image')
ax2.axis('off')



ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(log_spectrum, cmap='gray') 
ax3.set_title('Fourier Spectrum')
ax3.axis('off')

ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(image_after_filtering_uint8, cmap='gray')
ax4.set_title('Notch Filtered Image')
ax4.axis('off')

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(br_filtered_img_uint8, cmap='gray')
ax5.set_title('Band Reject Filtered Image')
ax5.axis('off')

snr_notch = compute_SNR(image_array, image_after_filtering_uint8)
snr_band_reject = compute_SNR(image_array, br_filtered_img_uint8)

print(f"SNR after Notch Filtering: {snr_notch:.2f} dB")
print(f"SNR after Band Reject Filtering: {snr_band_reject:.2f} dB")
plt.tight_layout()
plt.show()



#save Images
plt.figure(figsize=(5, 5))
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig('./output_images/original_image.png')  
plt.close()


plt.figure(figsize=(5, 5))
plt.imshow(image_with_noise_uint8, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')
plt.savefig('./output_images/noisy_image.png') 
plt.close()


plt.figure(figsize=(5, 5))
plt.imshow(log_spectrum, cmap='gray')  
plt.title('Fourier Spectrum')
plt.axis('off')
plt.savefig('./output_images/fourier_spectrum.png') 


plt.figure(figsize=(5, 5))
plt.imshow(image_after_filtering_uint8, cmap='gray')
plt.title('Notch Filtered Image')
plt.axis('off')
plt.savefig('./output_images/notch_filtered_image.png') 
plt.close()


plt.figure(figsize=(5, 5))
plt.imshow(br_filtered_img_uint8, cmap='gray')
plt.title('Band Reject Filtered Image')
plt.axis('off')
plt.savefig('./output_images/band_reject_filtered_image.png') 
plt.close()


