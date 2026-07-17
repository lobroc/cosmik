import numpy as np
import pypulseq as pp
import json5

from mrinufft.io.pulseq import acq2opts
from mrinufft.trajectories.utils import Acquisition, Hardware
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from decimal import Decimal, ROUND_HALF_UP

def compute_scanner_params(spatial_dims=[256, 256, 256]):
    with open('../simulations-mr0/scanner_properties.jsonc', 'r') as f:
        properties = json5.load(f)

    # Sequence parameters
    TE = 12e-6  # s
    TR = 2.0e-3  # s
    FA = 5.0  # degrees
    PULSE_DURATION = 16e-6 # s
    SPATIAL_DIMS_PX = spatial_dims

    acq = Acquisition(
        fov=[0.1, 0.1],  # m
        img_size=SPATIAL_DIMS_PX,
        hardware=Hardware(
            n_coils=1, # KomaMRI
            field_strength=properties['B0'],  # T
            gmax=properties['max_grad'], # T/m
            smax=properties['max_slew'], # T/m/s
            grad_raster_time=properties['grad_raster_time'], # s
        ),
        adc_dwell_time=properties['adc_raster_time'], # s
        norm_factor=0.5
    )
    print(acq)
    opts = acq2opts(acq)
    opts.B0 = properties['B0']  # T
    opts.block_duration_raster=properties['block_duration_raster']
    opts.grad_raster_time=properties['grad_raster_time']
    opts.rf_raster_time=properties['rf_raster_time']
    opts.adc_raster_time=properties['adc_raster_time']
    opts.adc_samples_divisor=properties['adc_oversampling']  # oversampling factor for ADC

    return opts, acq, TE, TR, FA, PULSE_DURATION, SPATIAL_DIMS_PX

R2D = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def golden_ratio_traj_2d(Nc: int) -> NDArray:
    angle = 0
    traj_buf = np.zeros((Nc, 2))
    base_vec = np.array([0.5, 0.0])
    for i in range(Nc):
        angle = np.deg2rad(137.51 * i) % (2 * np.pi)
        traj_buf[i] = R2D(angle) @ base_vec
    return traj_buf

if __name__ == '__main__':
    #
    # ----------------------------------
    # Input parameters
    # ----------------------------------
    #
    dummy_time = 5.0
    runtime = 15.0
    img_size = 64

    opts, acq, TE, TR, FA, PULSE_DURATION, SPATIAL_DIMS_PX = compute_scanner_params([img_size, img_size])
    opts.max_slew *= 10.0 # We "cheat" here so we can generate the sequence.
    opts.max_grad *= 2.0
    print(opts)

    quantise_float = lambda x, raster_time: float(Decimal(str(x)).quantize(Decimal(str(raster_time)), rounding=ROUND_HALF_UP))
    snap_to_raster = lambda x, raster_time: int(np.round(x / raster_time)) * raster_time

    #
    # ----------------------------------
    # Find trajectory, with inferred spokes/samples
    # ----------------------------------
    #

    # Compute k-space parameters
    # Use: https://mriquestions.com/field-of-view-fov.html
    FOV = acq.fov[0]
    delta_w = FOV / img_size  # pixel widths in m
    delta_k = 1 / FOV  # in 1/m
    k_fov = 1 / delta_w  # in 1/m
    k_max = k_fov / 2  # in 1/m

    # Infer number of spokes and samples
    max_fov_dir, min_fov_dir = sorted(range(2), key=lambda i: acq.fov[i], reverse=True)

    # Computing Nyquist criterion
    n = 4 # commonly used param
    nyquist_spokes = 16*np.pi*(k_max**2) / (n*np.tan(np.pi/n)) # https://cds.ismrm.org/protected/22MProceedings/PDFfiles/2450.html
    nyquist_size_matrix = 4*np.pi*(img_size**2)

    samples = int(np.ceil(k_max / delta_k))
    dummy_spokes = int(np.ceil(dummy_time / TR))
    spokes = int(np.ceil((runtime - dummy_time) / TR))

    print('Number of spokes for Nyquist criterion:', np.ceil(nyquist_spokes).astype(int))
    print('Number of spokes for Nyquist (alt):', np.ceil(nyquist_size_matrix).astype(int))
    print(f'We are doing {np.round(100 * spokes / nyquist_spokes, 1)} % sampling at worst case (max fov dir).')

    print('Inferred spokes:', spokes, ' and samples:', samples)

    trajectory = golden_ratio_traj_2d(spokes)

    gx, gy = trajectory.T

    # np.save(data_dir / '3d_golden_means_radial_trajectory.npy', trajectory)
    print('Trajectory computed.')

    #
    # ----------------------------------
    # Compute sequence
    # ----------------------------------
    #

    sequence = pp.Sequence(system=opts)

    dt = opts.adc_raster_time # Use adc time for grad alignment to make task easier
    grad_speed_amp = 1

    for i in tqdm(range(dummy_spokes + spokes), desc='Generating sequence blocks'):
        rf_pulse = pp.make_block_pulse(
            flip_angle=np.deg2rad(FA),
            duration=(PULSE_DURATION // opts.rf_raster_time) * opts.rf_raster_time,
            system=opts,
            phase_offset=np.deg2rad((117 * i)) % (2 * np.pi)
        )
        t_rf_center = pp.calc_rf_center(rf_pulse)[0]
        grad_time = quantise_float(samples*opts.grad_raster_time/grad_speed_amp, opts.grad_raster_time)

        final_resting_time = quantise_float(TR - TE - (2 * grad_time) - t_rf_center, opts.grad_raster_time)
        if final_resting_time < 0:
            grad_speed_amp += 1
            if grad_speed_amp >= (opts.grad_raster_time / dt):
                raise RuntimeError("Cannot fit sampling within gradient ramp + flat top. Consider reducing number of samples or increasing grad time.")

            grad_time = quantise_float(samples*opts.grad_raster_time/grad_speed_amp, opts.grad_raster_time)
            final_resting_time = quantise_float(TR - TE - (2 * grad_time) - t_rf_center, opts.grad_raster_time)


        sequence.add_block(rf_pulse)

        if i < dummy_spokes:
            sequence.add_block(pp.make_delay(TR - PULSE_DURATION))
            continue

        sequence.add_block(pp.make_delay(quantise_float(TE - t_rf_center, opts.adc_raster_time)))

        k = (2 * trajectory[i-dummy_spokes]) * k_max # shape (Ns, 3)
        grad_ramp_time = snap_to_raster(grad_time / 8, opts.grad_raster_time)
        grad_fall_time = grad_ramp_time
        grad_flat_time = quantise_float(grad_time - grad_ramp_time - grad_fall_time, opts.grad_raster_time)

        gx = pp.make_trapezoid(
            channel='x',
            area=k[0],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )

        gy = pp.make_trapezoid(
            channel='y',
            area=k[1],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )

        adc_duration = grad_ramp_time + grad_flat_time
        adc_dwell_time = snap_to_raster(adc_duration / samples, opts.adc_raster_time)
        adc = pp.make_adc(
            num_samples=adc_duration / adc_dwell_time,
            duration=adc_duration,
            delay=0,
            system=opts,
            phase_offset=np.deg2rad((117 * i)) % (2 * np.pi)
        )

        sequence.add_block(gx, gy, adc)

        gx2 = pp.make_trapezoid(
            channel='x',
            area=-k[0],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )
        gy2 = pp.make_trapezoid(
            channel='y',
            area=-k[1],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )

        sequence.add_block(gx2, gy2)

        sequence.add_block(pp.make_delay(final_resting_time))

    print(sequence.test_report())

    sequence.write('radial_2d.seq', check_timing=False)
