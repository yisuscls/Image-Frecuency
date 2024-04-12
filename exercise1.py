import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image
import os
#************************************************************************************************
# 1. 2-D Fast Fourier Transform
#************************************************************************************************

# read the image
path = './DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0417(a)(barbara).tif'
image = Image.open(path).convert('L')

#convert image to array
image = np.asarray(image)

# 2D FFT
fft_2D = fft2(image)

# Center
fft_center = fftshift(fft_2D)

# magnitude
magnitude = np.log(np.abs(fft_center))

# Compute average value of the image
average = np.mean(image)

# Reconstruct the image
reconstructed_image = np.abs(ifft2(ifftshift(fft_center)))

# Display  the results
plt.figure(figsize=(15, 5))
plt.suptitle(f'Average value of the image: {average}')
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(magnitude, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(133),plt.imshow(reconstructed_image, cmap='gray'), plt.title('Reconstructed Image')
plt.show()

#save images

plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig('./output_images/ex1_original.png') 
plt.close()

plt.figure(figsize=(5, 5))
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')
plt.savefig('./output_images/ex1_spectrum.png') 
plt.close()

plt.figure(figsize=(5, 5))
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.savefig('./output_images/ex1_final.png') 
plt.close()