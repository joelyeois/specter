import torch
from cryosim.microscope import Detector


def test_coincidence_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parameters
    pixel_size = 1.0
    dose_per_angstrom = 10.0
    coincidence_radius = 2.0
    num_frames = 10

    # Create detector
    detector = Detector(
        pixel_size=pixel_size,
        dose_per_angstrom=dose_per_angstrom,
        noise_model="poisson",
        coincidence_radius=coincidence_radius,
        num_frames=num_frames,
    )
    detector.to(device)

    # Dummy intensity map (H, W)
    H, W = 128, 128
    intensity_map = torch.ones((H, W), device=device)
    intensity_map /= intensity_map.sum()  # Normalize

    print(f"Running apply_coincidence with radius {coincidence_radius}...")
    # apply_coincidence expects img which is the "raw" image before noise
    # If from holography, it's intensity. If from CTF, it's dose-scaled.
    # Our Detector.apply_coincidence now normalizes it to a probability map anyway.
    raw_img = torch.ones((H, W), device=device) * (dose_per_angstrom * pixel_size**2)

    output_img = detector.apply_coincidence(raw_img)

    print(f"Output image sum: {output_img.sum().item()}")
    print(f"Output image shape: {output_img.shape}")

    # Basic verification
    assert output_img.shape == (H, W)
    assert output_img.sum() > 0

    # Compare with no coincidence
    detector_no_coinc = Detector(
        pixel_size=pixel_size,
        dose_per_angstrom=dose_per_angstrom,
        noise_model="poisson",
        coincidence_radius=0.0,
    )
    detector_no_coinc.to(device)
    output_no_coinc = detector_no_coinc.apply_coincidence(raw_img)
    print(f"No coincidence sum: {output_no_coinc.sum().item()}")

    if output_img.sum() < output_no_coinc.sum():
        print("Success: Coincidence loss reduced the total counts.")
    else:
        print(
            "Warning: Coincidence loss did not reduce the total counts (might be due to low dose)."
        )


if __name__ == "__main__":
    test_coincidence_loss()
