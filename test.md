# Fast Fourier Transform (FFT) - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [The DFT vs FFT](#the-dft-vs-fft)
4. [Cooley-Tukey Algorithm](#cooley-tukey-algorithm)
5. [Implementation from Scratch](#implementation-from-scratch)
6. [Practical Examples](#practical-examples)
7. [Applications](#applications)
8. [Advanced Topics](#advanced-topics)

## Introduction

The Fast Fourier Transform (FFT) is one of the most important algorithms in computational science. It efficiently computes the Discrete Fourier Transform (DFT) and its inverse, reducing the computational complexity from O(n²) to O(n log n).

**Key Applications:**
- Signal processing and filtering
- Image and audio compression
- Solving partial differential equations
- Fast polynomial multiplication
- Digital communications

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import time

# Set up plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
```

## Mathematical Foundation

### The Discrete Fourier Transform (DFT)

For a sequence of N complex numbers x₀, x₁, ..., x_{N-1}, the DFT is defined as:

**X_k = Σ(n=0 to N-1) x_n * e^(-2πikn/N)**

where:
- X_k is the k-th frequency component
- x_n is the n-th time-domain sample
- e^(-2πikn/N) is the complex exponential (twiddle factor)

```python
def naive_dft(x):
    """
    Compute DFT using the naive O(n²) algorithm
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X

# Example with a simple signal
t = np.linspace(0, 1, 8, endpoint=False)
x = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 4 * t)

print("Input signal:", x)
print("DFT result:", naive_dft(x))
```

### Complex Exponentials and Twiddle Factors

The key insight of the FFT is that the twiddle factors W_N^{kn} = e^(-2πikn/N) have special symmetry properties:

```python
def plot_twiddle_factors():
    """Visualize twiddle factors on the unit circle"""
    N = 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Twiddle factors for N=8
    for k in range(N):
        w = np.exp(-2j * np.pi * k / N)
        ax1.plot(w.real, w.imag, 'ro', markersize=10)
        ax1.annotate(f'W₈^{k}', (w.real, w.imag), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12)
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Twiddle Factors on Unit Circle (N=8)')
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    
    # Magnitude and phase
    k_values = np.arange(N)
    w_values = np.exp(-2j * np.pi * k_values / N)
    
    ax2.stem(k_values, np.abs(w_values), basefmt=' ')
    ax2.set_title('Magnitude of Twiddle Factors')
    ax2.set_xlabel('k')
    ax2.set_ylabel('|W₈^k|')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_twiddle_factors()
```

## The DFT vs FFT

### Computational Complexity Comparison

```python
def complexity_comparison():
    """Compare DFT and FFT computational complexity"""
    sizes = 2 ** np.arange(4, 12)  # Powers of 2 from 16 to 2048
    dft_times = []
    fft_times = []
    
    for N in sizes:
        # Generate random signal
        x = np.random.random(N) + 1j * np.random.random(N)
        
        # Time naive DFT (only for smaller sizes)
        if N <= 256:
            start = time.time()
            _ = naive_dft(x)
            dft_times.append(time.time() - start)
        else:
            dft_times.append(np.nan)
        
        # Time FFT
        start = time.time()
        _ = fft(x)
        fft_times.append(time.time() - start)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    valid_mask = ~np.isnan(dft_times)
    plt.loglog(sizes[valid_mask], np.array(dft_times)[valid_mask], 'ro-', label='Naive DFT O(n²)')
    plt.loglog(sizes, fft_times, 'bo-', label='FFT O(n log n)')
    plt.xlabel('Signal Length (N)')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    theoretical_dft = sizes**2 / (sizes[0]**2) * dft_times[0] if not np.isnan(dft_times[0]) else None
    theoretical_fft = sizes * np.log2(sizes) / (sizes[0] * np.log2(sizes[0])) * fft_times[0]
    
    if theoretical_dft is not None:
        plt.loglog(sizes, theoretical_dft, 'r--', alpha=0.7, label='Theoretical O(n²)')
    plt.loglog(sizes, theoretical_fft, 'b--', alpha=0.7, label='Theoretical O(n log n)')
    plt.loglog(sizes, fft_times, 'bo-', label='Actual FFT')
    plt.xlabel('Signal Length (N)')
    plt.ylabel('Relative Time')
    plt.title('Theoretical vs Actual Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

complexity_comparison()
```

## Cooley-Tukey Algorithm

The most common FFT algorithm uses a divide-and-conquer approach, recursively breaking down the DFT into smaller DFTs.

### Decimation-in-Time (DIT) Approach

```python
def fft_recursive(x):
    """
    Recursive implementation of Cooley-Tukey FFT
    Input length must be a power of 2
    """
    N = len(x)
    
    # Base case
    if N == 1:
        return x
    
    # Divide
    even = fft_recursive(x[0::2])  # Even-indexed elements
    odd = fft_recursive(x[1::2])   # Odd-indexed elements
    
    # Combine
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    
    return np.concatenate([even + T, even - T])

def bit_reverse_permutation(x):
    """
    Rearrange array elements according to bit-reversed indices
    """
    N = len(x)
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if j > i:
            x[i], x[j] = x[j], x[i]
    return x

def fft_iterative(x):
    """
    Iterative implementation of Cooley-Tukey FFT (more efficient)
    """
    x = x.copy()  # Don't modify original
    N = len(x)
    
    # Bit-reverse permutation
    x = bit_reverse_permutation(x)
    
    # Iterative FFT
    length = 2
    while length <= N:
        # Twiddle factor for this stage
        w = np.exp(-2j * np.pi / length)
        
        for i in range(0, N, length):
            wn = 1
            for j in range(length // 2):
                u = x[i + j]
                v = x[i + j + length // 2] * wn
                x[i + j] = u + v
                x[i + j + length // 2] = u - v
                wn *= w
        
        length *= 2
    
    return x

# Test our implementations
N = 16
t = np.linspace(0, 1, N, endpoint=False)
test_signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.cos(2 * np.pi * 4 * t)

print("Testing FFT implementations:")
print("NumPy FFT:      ", np.abs(fft(test_signal)))
print("Recursive FFT:  ", np.abs(fft_recursive(test_signal)))
print("Iterative FFT:  ", np.abs(fft_iterative(test_signal)))
print("Max difference: ", np.max(np.abs(fft(test_signal) - fft_iterative(test_signal))))
```

### Visualization of FFT Stages

```python
def visualize_fft_stages():
    """Visualize how FFT breaks down the problem"""
    N = 8
    x = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=complex)  # Simple rectangular pulse
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Original signal
    axes[0].stem(range(N), x.real, basefmt=' ')
    axes[0].set_title('Original Signal (Time Domain)')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # FFT result
    X = fft(x)
    axes[1].stem(range(N), np.abs(X), basefmt=' ')
    axes[1].set_title('FFT Result (Frequency Domain)')
    axes[1].set_xlabel('Frequency Bin')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True, alpha=0.3)
    
    # Phase
    axes[2].stem(range(N), np.angle(X), basefmt=' ')
    axes[2].set_title('Phase Spectrum')
    axes[2].set_xlabel('Frequency Bin')
    axes[2].set_ylabel('Phase (radians)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_fft_stages()
```

## Implementation from Scratch

### Complete FFT Implementation with Error Checking

```python
class FFT:
    """
    Complete FFT implementation with various algorithms and utilities
    """
    
    @staticmethod
    def is_power_of_2(n):
        """Check if n is a power of 2"""
        return n > 0 and (n & (n - 1)) == 0
    
    @staticmethod
    def next_power_of_2(n):
        """Find the next power of 2 greater than or equal to n"""
        return 1 << (n - 1).bit_length()
    
    @staticmethod
    def zero_pad_to_power_of_2(x):
        """Zero-pad signal to next power of 2 length"""
        N = len(x)
        next_pow2 = FFT.next_power_of_2(N)
        if next_pow2 > N:
            x_padded = np.zeros(next_pow2, dtype=complex)
            x_padded[:N] = x
            return x_padded
        return x
    
    @staticmethod
    def cooley_tukey_fft(x):
        """
        Cooley-Tukey FFT with automatic zero-padding
        """
        x = np.array(x, dtype=complex)
        
        # Zero-pad to power of 2 if necessary
        if not FFT.is_power_of_2(len(x)):
            x = FFT.zero_pad_to_power_of_2(x)
        
        return fft_iterative(x)
    
    @staticmethod
    def inverse_fft(X):
        """
        Compute inverse FFT
        """
        N = len(X)
        # Conjugate, apply FFT, conjugate again, and scale
        return np.conj(FFT.cooley_tukey_fft(np.conj(X))) / N
    
    @staticmethod
    def fft_2d(image):
        """
        2D FFT for image processing
        """
        # Apply 1D FFT to each row
        rows_fft = np.array([FFT.cooley_tukey_fft(row) for row in image])
        # Apply 1D FFT to each column
        return np.array([FFT.cooley_tukey_fft(col) for col in rows_fft.T]).T

# Test the complete implementation
print("Testing complete FFT implementation:")
test_data = np.random.random(15) + 1j * np.random.random(15)  # Non-power of 2
our_fft = FFT.cooley_tukey_fft(test_data)
scipy_fft = fft(test_data, n=FFT.next_power_of_2(len(test_data)))
print(f"Max difference: {np.max(np.abs(our_fft - scipy_fft)):.2e}")
```

## Practical Examples

### 1. Signal Filtering

```python
def signal_filtering_example():
    """Demonstrate signal filtering using FFT"""
    # Create a noisy signal
    t = np.linspace(0, 1, 1000, endpoint=False)
    clean_signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    noise = 0.5 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    # Compute FFT
    X = fft(noisy_signal)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    
    # Design a simple low-pass filter
    cutoff = 100  # Hz
    X_filtered = X.copy()
    X_filtered[np.abs(freqs) > cutoff] = 0  # Zero out high frequencies
    
    # Inverse FFT to get filtered signal
    filtered_signal = np.real(ifft(X_filtered))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain signals
    axes[0, 0].plot(t[:200], clean_signal[:200], 'g-', label='Clean', linewidth=2)
    axes[0, 0].plot(t[:200], noisy_signal[:200], 'r-', alpha=0.7, label='Noisy')
    axes[0, 0].plot(t[:200], filtered_signal[:200], 'b--', label='Filtered', linewidth=2)
    axes[0, 0].set_title('Time Domain Signals')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency domain - original
    axes[0, 1].semilogy(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]))
    axes[0, 1].axvline(cutoff, color='r', linestyle='--', label=f'Cutoff: {cutoff} Hz')
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency domain - filtered
    axes[1, 0].semilogy(freqs[:len(freqs)//2], np.abs(X_filtered[:len(X_filtered)//2]))
    axes[1, 0].axvline(cutoff, color='r', linestyle='--', label=f'Cutoff: {cutoff} Hz')
    axes[1, 0].set_title('Filtered Spectrum')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error analysis
    error = np.abs(clean_signal - filtered_signal)
    axes[1, 1].plot(t[:200], error[:200])
    axes[1, 1].set_title(f'Reconstruction Error (RMS: {np.sqrt(np.mean(error**2)):.3f})')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

signal_filtering_example()
```

### 2. Image Processing with 2D FFT

```python
def image_processing_example():
    """Demonstrate 2D FFT for image processing"""
    # Create a simple test image
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, y)
    
    # Create a pattern with different frequency components
    image = (np.sin(X) * np.cos(Y) + 0.5 * np.sin(4*X) * np.sin(4*Y) + 
             0.25 * np.sin(8*X) * np.cos(2*Y))
    
    # Add noise
    noisy_image = image + 0.3 * np.random.randn(*image.shape)
    
    # Compute 2D FFT
    F = np.fft.fft2(noisy_image)
    F_shifted = np.fft.fftshift(F)  # Shift zero frequency to center
    
    # Create a circular low-pass filter
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    r = 30  # Filter radius
    
    # Create mask
    y_mask, x_mask = np.ogrid[:rows, :cols]
    mask = (x_mask - ccol)**2 + (y_mask - crow)**2 <= r**2
    
    # Apply filter
    F_filtered = F_shifted.copy()
    F_filtered[~mask] = 0
    
    # Inverse FFT
    filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original and noisy images
    im1 = axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(filtered_image, cmap='gray')
    axes[0, 2].set_title('Filtered Image')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Frequency domain representations
    im4 = axes[1, 0].imshow(np.log(1 + np.abs(F_shifted)), cmap='hot')
    axes[1, 0].set_title('Original Spectrum (log scale)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Low-pass Filter Mask')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(np.log(1 + np.abs(F_filtered)), cmap='hot')
    axes[1, 2].set_title('Filtered Spectrum (log scale)')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()

image_processing_example()
```

### 3. Fast Convolution

```python
def convolution_example():
    """Demonstrate fast convolution using FFT"""
    # Create signals
    N = 1000
    t = np.linspace(0, 10, N)
    
    # Input signal: sum of sinusoids
    signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
    
    # Filter kernel: Gaussian
    kernel_size = 101
    kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size//2) / 10)**2)
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Direct convolution (slow)
    start_time = time.time()
    conv_direct = np.convolve(signal, kernel, mode='same')
    direct_time = time.time() - start_time
    
    # FFT-based convolution (fast)
    start_time = time.time()
    # Zero-pad both signals to avoid circular convolution artifacts
    n_conv = len(signal) + len(kernel) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n_conv)))
    
    signal_padded = np.zeros(n_fft)
    kernel_padded = np.zeros(n_fft)
    signal_padded[:len(signal)] = signal
    kernel_padded[:len(kernel)] = kernel
    
    # Convolution in frequency domain
    conv_fft_full = np.real(ifft(fft(signal_padded) * fft(kernel_padded)))
    # Extract the 'same' portion
    start_idx = len(kernel) // 2
    conv_fft = conv_fft_full[start_idx:start_idx + len(signal)]
    fft_time = time.time() - start_time
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original signal and kernel
    axes[0, 0].plot(t, signal, 'b-', label='Original Signal')
    axes[0, 0].set_title('Input Signal')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(kernel, 'r-', linewidth=2)
    axes[0, 1].set_title('Convolution Kernel (Gaussian)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Convolution results
    axes[1, 0].plot(t, conv_direct, 'g-', label='Direct Convolution', linewidth=2)
    axes[1, 0].plot(t, conv_fft, 'r--', label='FFT Convolution', alpha=0.8)
    axes[1, 0].set_title(f'Convolution Results\nDirect: {direct_time:.4f}s, FFT: {fft_time:.4f}s')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error analysis
    error = np.abs(conv_direct - conv_fft)
    axes[1, 1].plot(t, error)
    axes[1, 1].set_title(f'Absolute Error (Max: {np.max(error):.2e})')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Speed improvement: {direct_time/fft_time:.1f}x faster")

convolution_example()
```

## Applications

### Polynomial Multiplication

```python
def polynomial_multiplication():
    """Demonstrate fast polynomial multiplication using FFT"""
    # Define two polynomials
    # p(x) = 1 + 2x + 3x² + 4x³
    # q(x) = 2 + 3x + x²
    
    p = [1, 2, 3, 4]  # Coefficients in ascending order
    q = [2, 3, 1]
    
    # Direct multiplication
    def multiply_polynomials_direct(p, q):
        result = [0] * (len(p) + len(q) - 1)
        for i in range(len(p)):
            for j in range(len(q)):
                result[i + j] += p[i] * q[j]
        return result
    
    # FFT-based multiplication
    def multiply_polynomials_fft(p, q):
        n = len(p) + len(q) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        
        # Zero-pad
        p_padded = p + [0] * (n_fft - len(p))
        q_padded = q + [0] * (n_fft - len(q))
        
        # FFT, multiply, inverse FFT
        p_fft = fft(p_padded)
        q_fft = fft(q_padded)
        result_fft = p_fft * q_fft
        result = np.real(ifft(result_fft))
        
        return result[:n]
    
    # Compare results
    direct_result = multiply_polynomials_direct(p, q)
    fft_result = multiply_polynomials_fft(p, q)
    
    print("Polynomial Multiplication:")
    print(f"p(x) coefficients: {p}")
    print(f"q(x) coefficients: {q}")
    print(f"Direct result:     {direct_result}")
    print(f"FFT result:        {[round(x) for x in fft_result]}")
    print(f"Max difference:    {np.max(np.abs(np.array(direct_result) - fft_result[:len(direct_result)])):.2e}")

polynomial_multiplication()
```

## Advanced Topics

### Window Functions and Spectral Leakage

```python
def window_functions_demo():
    """Demonstrate the effect of window functions on spectral analysis"""
    # Create a signal with two close frequencies
    N = 512
    t = np.linspace(0, 1, N, endpoint=False)
    f1, f2 = 50, 55  # Close frequencies
    signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Define window functions
    windows = {
        'Rectangular': np.ones(N),
        'Hanning': np.hanning(N),
        'Hamming': np.hamming(N),
        'Blackman': np.blackman(N)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    freqs = np.fft.fftfreq(N, 1/N)
    freq_range = (freqs >= 0) & (freqs <= 100)
    
    for i, (name, window) in enumerate(windows.items()):
        # Apply window
        windowed_signal = signal * window
        
        # Compute FFT
        spectrum = fft(windowed_signal)
        magnitude = np.abs(spectrum)
        
        # Plot
        axes[i].plot(freqs[freq_range], magnitude[freq_range])
        axes[i].set_title(f'{name} Window')
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Magnitude')
        axes[i].axvline(f1, color='r', linestyle='--', alpha=0.7, label=f'f₁={f1} Hz')
        axes[i].axvline(f2, color='r', linestyle='--', alpha=0.7, label=f'f₂={f2} Hz')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt
