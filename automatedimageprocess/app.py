import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import streamlit as st
from PIL import Image

# Streamlit Web Application
def process_image(image, is_medical):
    # Convert uploaded image to grayscale
    image = np.array(image.convert('L'))

    # Analyze the noise type first
    mean_intensity = np.mean(image)
    std_dev = np.std(image)
    skewness = skew(image.flatten())
    kurt = kurtosis(image.flatten())

    # Filter based on noise type
    if np.count_nonzero(image == 0) > 0 and np.count_nonzero(image == 255) > 0:
        st.write("Applying Median Filter for salt-and-pepper noise.")
        processed_image = cv2.medianBlur(image, 3)  # Kernel size of 3
    elif std_dev > 30:
        st.write("Applying Gaussian Blur for Gaussian noise.")
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)  # Kernel size of 5x5
    elif abs(skewness) > 1:
        st.write("Applying Wiener Filter for Poisson noise.")
        processed_image = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)
    else:
        st.write("No significant noise detected. Applying general smoothing.")
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)

    # If the image is medical, perform additional processing
    if is_medical:
        st.write("Performing essential medical image processing...")
        # Example: Negation
        negated_image = cv2.bitwise_not(processed_image)
        processed_image = negated_image

    # Sharpen the image
    kernel_sharpen = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
    sharpened_image = cv2.filter2D(processed_image, -1, kernel_sharpen)

    # Optional: Further smoothing if needed
    smoothed_image = cv2.GaussianBlur(sharpened_image, (3, 3), 0)

    return image, smoothed_image, sharpened_image


def plot_images(original, processed, sharpened):
    # Plot original, processed, and sharpened images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(processed, cmap='gray')
    axs[1].set_title('Processed Image')
    axs[1].axis('off')

    axs[2].imshow(sharpened, cmap='gray')
    axs[2].set_title('Sharpened Image')
    axs[2].axis('off')

    st.pyplot(fig)

    # Plot histogram for processed image
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    histogram_processed, bins = np.histogram(processed.flatten(), bins=256, range=[0, 256])
    ax_hist.plot(histogram_processed, color='black')
    ax_hist.set_title('Histogram of Processed Image')
    ax_hist.set_xlabel('Pixel Intensity')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_xlim([0, 256])
    ax_hist.grid()

    st.pyplot(fig_hist)


def main():
    st.title("Real-Time Image Processing Web App")

    # Sidebar for user input
    st.sidebar.title("Upload and Settings")

    # Upload an image file
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image_width = 300
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Ask user if it's a medical image
        is_medical = st.sidebar.checkbox("Is this a medical image?", value=False)

        if st.sidebar.button("Process Image"):
            # Process the image
            original, processed, sharpened = process_image(image, is_medical)

            # Display results
            plot_images(original, processed, sharpened)
    else:
        st.write("Please upload an image to get started.")


if __name__ == "__main__":
    main()
