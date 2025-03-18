import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import spectre
from svgpathtools import svg2paths


def setup_parser():
    parser = argparse.ArgumentParser(description="Fourier Epicycles Animation")
    parser.add_argument(
        "image_path", nargs="?", default=None, help="Path to the input image or SVG (optional)"
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=180,
        help="Number of Fourier components to include (default: 180)",
    )
    parser.add_argument(
        "--a", type=float, default=1.0, help="Value for parameter a (default: 1.0)"
    )
    parser.add_argument(
        "--b", type=float, default=1.0, help="Value for parameter b (default: 1.0)"
    )
    parser.add_argument(
        "--curve_strength",
        type=float,
        default=0.5,
        help="Strength of the curve (default: 0.5)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the animation as a video file (default: False)",
    )
    return parser


def process_image(image_path):
    """Extracts the largest contour from a raster image (PNG, JPG, etc.)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("Error: No contours found in the image.")
        sys.exit(1)

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    x_points = contour[:, 0]
    y_points = -contour[:, 1]  # Invert Y-axis for Matplotlib
    return x_points, y_points


def process_svg(svg_path, points_per_segment=80):
    """
    Parses an SVG file and samples points along each path in parametric space.
    Uses direct segment sampling (t ∈ [0,1]) to avoid arc-length calculation issues.
    """
    paths, _ = svg2paths(svg_path)
    all_x, all_y = [], []

    for path in paths:
        for sp in path.continuous_subpaths():
            for segment in sp:
                seg_len = segment.length()
                if seg_len < 1e-9:
                    continue

                # Sample parametric values t ∈ [0,1] instead of distance along path
                for i in range(points_per_segment):
                    t = i / (points_per_segment - 1)
                    pt = segment.point(t)  # Complex number (real=x, imag=y)
                    all_x.append(pt.real)
                    all_y.append(pt.imag)

    return np.array(all_x), np.array(all_y)


def normalize_points(x_points, y_points):
    """Normalizes points to fit within [-1, 1] range in both axes."""
    x_points = x_points.astype(np.float64)  # Convert to float
    y_points = y_points.astype(np.float64)  # Convert to float

    x_points -= np.mean(x_points)
    y_points -= np.mean(y_points)

    max_range = max(np.max(np.abs(x_points)), np.max(np.abs(y_points)))
    if max_range > 0:
        x_points /= max_range
        y_points /= max_range

    return x_points, y_points



def setup_fourier_transform(x_points, y_points, num_components):
    """Computes Fourier coefficients and frequencies."""
    points = x_points + 1j * y_points
    N = len(points)
    num_components = min(num_components, N)

    coefficients = np.fft.fft(points) / N
    freqs = np.fft.fftfreq(N, d=1 / N)
    omega = 2 * np.pi * freqs

    coefficients = np.fft.fftshift(coefficients)
    omega = np.fft.fftshift(omega)

    indices = np.argsort(-np.abs(coefficients))
    coefficients = coefficients[indices]
    omega = omega[indices]

    return coefficients, omega, N, num_components


def init_animation():
    """Initializes the plot for animation."""
    line.set_data([], [])
    path_line.set_data([], [])
    return line, path_line


def stop_animation():
    """Stops animation and saves the final image output."""
    ani.event_source.stop()
    line.set_animated(False)
    path_line.set_animated(False)
    for c in circles:
        c.set_animated(False)

    fig.canvas.draw()
    line.set_visible(False)
    path_line.set_visible(False)
    for c in circles:
        c.remove()

    fig.canvas.draw()
    plt.savefig("final_border.png", facecolor="white", dpi=300)
    ax.fill(xdata, ydata, color="black", alpha=1.0)
    fig.canvas.draw()
    plt.savefig("border_and_filled.png", facecolor="white", dpi=300)


def update_animation(frame, coefficients, omega, N, num_components):
    """Updates the animation frame with Fourier epicycles."""
    global circles, visited_points

    t_current = frame * (2 * np.pi / N)
    x, y = 0.0, 0.0
    xs, ys = [], []

    for c in circles:
        c.remove()
    circles.clear()

    x_prev, y_prev = 0.0, 0.0
    for n in range(num_components):
        coef, freq = coefficients[n], omega[n]
        x_prev, y_prev = x, y
        x += np.real(coef * np.exp(1j * freq * t_current))
        y += np.imag(coef * np.exp(1j * freq * t_current))

        radius = np.abs(coef)
        circle = plt.Circle((x_prev, y_prev), radius, fill=False, alpha=0.3)
        ax.add_patch(circle)
        circles.append(circle)

        xs.append(x_prev)
        ys.append(y_prev)

    xs.append(x)
    ys.append(y)
    line.set_data(xs, ys)

    xdata.append(x)
    ydata.append(y)
    path_line.set_data(xdata, ydata)

    STOP_THRESHOLD = 0.0005
    for vx, vy in visited_points:
        if np.hypot(x - vx, y - vy) <= STOP_THRESHOLD:
            stop_animation()
            return line, path_line, *circles
    else:
        visited_points.append((x, y))

    return line, path_line, *circles


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # 1) Load input data
    if args.image_path:
        if args.image_path.lower().endswith(".svg"):
            x_points, y_points = process_svg(args.image_path, points_per_segment=80)
        else:
            x_points, y_points = process_image(args.image_path)
    else:
        x_points, y_points = spectre.generate_monotile(args.a, args.b, args.curve_strength)

    # 2) Normalize
    x_points, y_points = normalize_points(x_points, y_points)

    # 3) Fourier transform
    coefficients, omega, N, num_components = setup_fourier_transform(
        x_points, y_points, args.num_components
    )

    # 4) Setup plot
    global fig, ax, line, path_line, circles, xdata, ydata, visited_points, ani

    visited_points = []
    xdata, ydata = [], []
    circles = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    (line,) = ax.plot([], [], "black")
    (path_line,) = ax.plot([], [], "black")

    # 5) Create animation
    ani = FuncAnimation(
        fig,
        update_animation,
        frames=N,
        init_func=init_animation,
        fargs=(coefficients, omega, N, num_components),
        blit=True,
        interval=20,
    )

    # 6) Save or show
    if args.save:
        fps = max(1, int(N / 10))  # at least 1 fps
        ani.save("fourier_epicycles.mp4", writer="ffmpeg", fps=fps)
    else:
        plt.show()


if __name__ == "__main__":
    main()
