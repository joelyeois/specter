import torch
import time
from specter.microscope import Detector


def test_fast_detector():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parameters
    pixel_size = 1.0
    dose_per_angstrom = 1.0  # Lower dose for comparison with Original
    coincidence_radius = 2.0
    num_frames = 1
    H, W = 128, 128

    # Create detector
    detector = Detector(
        pixel_size=pixel_size,
        dose_per_angstrom=dose_per_angstrom,
        noise_model="poisson",
        coincidence_radius=coincidence_radius,
        num_frames=num_frames,
    )
    detector.to(device)

    # Raw image (dose-scaled probability map)
    raw_img = torch.ones((H, W), device=device) * (dose_per_angstrom * pixel_size**2)
    intensity_map = raw_img / raw_img.sum()

    # Fixed seed for reproducibility (though poisson and rand will still vary)
    torch.manual_seed(42)

    # 1. Benchmark Original (manually calling old method)
    print("\n--- Running Original apply_detector_physics ---")
    start = time.time()
    # We'll temporarily point the detector to use the old method if we want to test it
    # but here we can just call it directly since it's still in the class
    out_old = detector.apply_detector_physics(
        intensity_map, pixel_size, dose_per_angstrom, coincidence_radius
    )
    end = time.time()
    old_time = end - start
    print(f"Original Time: {old_time:.4f}s, Sum: {out_old.sum().item()}")

    # 2. Benchmark Fast
    print("\n--- Running Optimized apply_detector_physics_fast ---")
    # Warmup
    _ = detector.apply_detector_physics_fast(
        intensity_map, pixel_size, dose_per_angstrom, coincidence_radius
    )

    start = time.time()
    out_new = detector.apply_detector_physics_fast(
        intensity_map, pixel_size, dose_per_angstrom, coincidence_radius
    )
    end = time.time()
    new_time = end - start
    print(f"Optimized Time: {new_time:.4f}s, Sum: {out_new.sum().item()}")

    print(f"\nSpeedup: {old_time / new_time:.2f}x")

    # Correctness check: Since they involve randomness, we can't expect identical pixels
    # with the same seed because of how they might consume the RNG state,
    # but the statistics and total counts should be very similar.
    # To TRULY verify identical logic, we'd need to mock the RNG.

    # Let's check if the counts are in the same ballpark
    diff_sum = abs(out_old.sum() - out_new.sum())
    rel_diff = diff_sum / out_old.sum()
    print(f"Relative difference in total counts: {rel_diff:.4%}")

    assert (
        rel_diff < 0.1
    ), "Total counts deviate too much (though randomness is involved)"


if __name__ == "__main__":
    test_fast_detector()
