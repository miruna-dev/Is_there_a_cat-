<h1 align="center">Cat Detector - From Image to Prediction</h1>

<p align="center">
  <i>An end-to-end MATLAB/Octave project for detecting cats in images using RGB processing and machine learning.</i>
</p>

---

## Project Overview

This project aims to build a complete image classification system that detects whether a **cat** is present in an image.

The pipeline includes:
- Reading and resizing images
- Splitting RGB channels
- Converting images to vectors
- Building a data matrix `X`
- *(Planned)* Training a lightweight machine learning model
- Making predictions on new images

> **Note for Octave users:**  
> You must run `pkg load image` before using image-related functions.

The goal is to keep the implementation simple and efficient, without using deep learning (CNN) — at least in the initial version.

---

## Functions

### `image_to_rgb_matrix`

**Purpose:**  
This function reads an image from a given path, resizes it to **64x64 pixels**, and separates it into its three individual color channels: **Red**, **Green**, and **Blue**.

---

#### Step-by-step behavior:

- The function starts by reading the image using `imread`.
- It resizes the image to `64x64` pixels using `imresize`.

**Why 64x64?**
- It simplifies computations by significantly reducing the number of pixels.
- It produces a smaller and easier-to-handle data matrix.
- It maintains sufficient visual information for the kind of lightweight ML model intended for this project.
- *(Note: The planned model is not a CNN, but a simpler custom algorithm that will be implemented later.)*

---

#### RGB Channel Extraction:

After resizing, the function separates the image into its 3 color channels:
- `R` — contains the red intensity values
- `G` — contains the green intensity values
- `B` — contains the blue intensity values

Each of these is a **64x64 matrix** that represents the saturation of that color across the image.

---

#### Visualizing the Output in Octave:

To test this function and visualize its result in Octave:

```matlab
pkg load image
[R, G, B] = image_to_rgb_matrix('path/to/your/image.jpg');
imshow(R)  % Visualizes the Red channel
