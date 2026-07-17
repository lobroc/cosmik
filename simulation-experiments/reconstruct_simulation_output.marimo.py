import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import h5py
    import numpy as np
    import cupy as cp
    import pypulseq as pp
    import mrinufft
    import matplotlib.pyplot as plt
    import ipyvolume as ipv

    from scipy.fft import fft, fftfreq, fftshift
    from pathlib import Path
    from tqdm.auto import trange, tqdm

    from typing import Tuple, Union
    from numpy.typing import NDArray

    # BACKEND = 'cufinufft'
    BACKEND = 'gpuNUFFT'

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    return (
        BACKEND,
        NDArray,
        Path,
        Union,
        mrinufft,
        np,
        plots_dir,
        plt,
        pp,
        trange,
    )


@app.cell
def _(Path, np):
    output_dir = Path('.').resolve()
    coil_signals_koma_static = np.load(output_dir / 'sig_koma_nomotion.npy').T
    coil_signals_koma_static = np.squeeze(coil_signals_koma_static)
    coil_signals_mr0_static = np.load(output_dir / 'sig_mr0_no_motion.npy').T
    coil_signals_mr0_static = np.squeeze(coil_signals_mr0_static) * 1j
    coil_signals_mr0_t2s_static = np.load(output_dir / 'sig_mod_mr0_no_motion.npy').T
    coil_signals_mr0_t2s_static = np.squeeze(coil_signals_mr0_t2s_static) * 1j
    print(coil_signals_koma_static.shape, coil_signals_mr0_static.shape)
    return (
        coil_signals_koma_static,
        coil_signals_mr0_static,
        coil_signals_mr0_t2s_static,
        output_dir,
    )


@app.cell
def _(
    coil_signals_koma_static,
    coil_signals_mr0_static,
    coil_signals_mr0_t2s_static,
    np,
    plots_dir,
    plt,
):
    fig3_, axs3_ = plt.subplots(1, 3, figsize=(8, 3))

    axs3_[0].plot(np.real(coil_signals_koma_static)[5:100])
    axs3_[0].plot(np.imag(coil_signals_koma_static)[5:100])
    axs3_[0].set_title('KomaMRI')
    axs3_[0].axis('off')
    axs3_[1].plot(np.real(coil_signals_mr0_static)[5:100])
    axs3_[1].plot(np.imag(coil_signals_mr0_static)[5:100])
    axs3_[1].set_title('MRzero')
    axs3_[1].axis('off')
    axs3_[2].plot(np.real(coil_signals_mr0_t2s_static)[5:100])
    axs3_[2].plot(np.imag(coil_signals_mr0_t2s_static)[5:100])
    axs3_[2].set_title('MRzero (with T2*)')
    axs3_[2].axis('off')
    fig3_.savefig(plots_dir / 'disk_coil_sigs_static.svg', transparent=True)
    plt.show()
    return


@app.cell
def _(np, output_dir):
    coil_signals_koma = np.load(output_dir / 'sig_koma.npy').T
    coil_signals_koma = np.squeeze(coil_signals_koma)
    coil_signals_mr0 = np.load(output_dir / 'sig_mr0.npy').T
    coil_signals_mr0 = np.squeeze(coil_signals_mr0) * 1j
    coil_signals_mr0_t2s = np.load(output_dir / 'sig_mod_mr0.npy').T
    coil_signals_mr0_t2s = np.squeeze(coil_signals_mr0_t2s) * 1j
    print(coil_signals_koma.shape, coil_signals_mr0.shape)
    return coil_signals_koma, coil_signals_mr0, coil_signals_mr0_t2s


@app.cell
def _(
    coil_signals_koma,
    coil_signals_mr0,
    coil_signals_mr0_t2s,
    np,
    plots_dir,
    plt,
):
    fig_, axs_ = plt.subplots(1, 3, figsize=(8, 3))

    axs_[0].plot(np.real(coil_signals_koma)[5:100])
    axs_[0].plot(np.imag(coil_signals_koma)[5:100])
    axs_[0].set_title('KomaMRI')
    axs_[0].axis('off')
    axs_[1].plot(np.real(coil_signals_mr0)[5:100])
    axs_[1].plot(np.imag(coil_signals_mr0)[5:100])
    axs_[1].set_title('MRzero')
    axs_[1].axis('off')
    axs_[2].plot(np.real(coil_signals_mr0_t2s)[5:100])
    axs_[2].plot(np.imag(coil_signals_mr0_t2s)[5:100])
    axs_[2].set_title('MRzero (with T2*)')
    axs_[2].axis('off')
    fig_.savefig(plots_dir / 'disk_coil_sigs_dyn.svg', transparent=True)
    plt.show()
    return


@app.cell
def _(pp):
    sequence = pp.Sequence()
    sequence.read('radial_2d.seq')
    print(sequence)
    return (sequence,)


@app.cell
def _(sequence):
    adc_events = [i for i in range(1, len(sequence.block_events) + 1) if sequence.get_block(i).adc is not None]
    return (adc_events,)


@app.cell
def _(adc_events, sequence):
    sequence_kspace_calc = sequence.calculate_kspace()
    k_traj_adc = sequence_kspace_calc[0] # Kspace trajectory during sampling
    kspace_times = sequence_kspace_calc[4] # Times at which samples are taken
    samples_per_adc = sequence.get_block(adc_events[0]).adc.num_samples
    num_spokes = k_traj_adc.shape[1] // samples_per_adc

    print('Samples per ADC:', samples_per_adc)
    print('Number of spokes:', num_spokes)
    return k_traj_adc, kspace_times, samples_per_adc


@app.cell
def _(coil_signals_koma, coil_signals_mr0, k_traj_adc, np):
    print(k_traj_adc.shape, coil_signals_koma.shape, coil_signals_mr0.shape)

    output_shape = (np.array((64, 64, 1))).astype(int)
    normed_ktraj = k_traj_adc / (2*np.max(np.abs(k_traj_adc))) # normalize to [-0.5, 0.5]
    flat_traj = normed_ktraj.T
    weights = np.sqrt(np.sum(np.abs(normed_ktraj**2), axis=0))

    recon_regularisation = 0.1
    max_recon_iters = 500
    recon_solver = 'lsqr'
    return flat_traj, normed_ktraj, output_shape


@app.cell
def _(BACKEND, flat_traj, mrinufft, normed_ktraj, output_shape):
    dcf = mrinufft.density.nufft_based.pipe(flat_traj[:, :2], shape=output_shape[:2], backend=BACKEND, osf=2)
    nufft = mrinufft.get_operator(BACKEND)(normed_ktraj.T, shape=output_shape, density=dcf, squeeze_dims=True, upsampfac=2.0)
    return dcf, nufft


@app.cell
def _(
    coil_signals_koma_static,
    coil_signals_mr0_static,
    coil_signals_mr0_t2s_static,
    np,
    nufft,
    plots_dir,
    plt,
):
    fig2_, axs2_ = plt.subplots(2, 5, figsize=(11, 3))
    for ax_ in axs2_.flatten():
        ax_.axis('off')

    mmax = 15
    plotimg1 = nufft.adj_op(coil_signals_koma_static)
    pl1 = axs2_[0, 0].imshow(np.abs(plotimg1), vmax=mmax)
    axs2_[0, 0].set_title('Koma static')
    plotimg2 = nufft.adj_op(coil_signals_mr0_static)
    pl2 = axs2_[0, 1].imshow(np.abs(plotimg2), vmax=mmax)
    axs2_[0, 1].set_title('MR0 static')
    plotimg3 = nufft.adj_op(coil_signals_mr0_t2s_static)
    axs2_[0, 2].set_title('MR0 static (with T2*)')
    pl3 = axs2_[0, 2].imshow(np.abs(plotimg3), vmax=mmax)

    comp1 = axs2_[0, 3].imshow(np.abs(plotimg1 - plotimg2), vmax=8)
    axs2_[0, 3].set_title('MR0 vs Koma')
    comp2 = axs2_[0, 4].imshow(np.abs(plotimg3 - plotimg2), vmax=8)
    axs2_[0, 4].set_title('MR0 with vs without T2*')

    for i, p in enumerate([pl1, pl2, pl3, comp1, comp2]):
        axpos = axs2_[0, i].get_position()
        cax = fig2_.add_axes([axpos.x0 + 0.11, axpos.y0, 0.015, axpos.height])
        fig2_.colorbar(p, cax=cax)

    pl1 = axs2_[1, 0].imshow(np.angle(plotimg1), cmap='hsv')
    pl2 = axs2_[1, 1].imshow(np.angle(plotimg2), cmap='hsv')
    pl3 = axs2_[1, 2].imshow(np.angle(plotimg3), cmap='hsv')

    comp1 = axs2_[1, 3].imshow((np.angle(plotimg1) - np.angle(plotimg2)), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    comp2 = axs2_[1, 4].imshow((np.angle(plotimg3) - np.angle(plotimg2)), cmap='hsv', vmin=-np.pi, vmax=np.pi)

    for i, p in enumerate([pl1, pl2, pl3, comp1, comp2]):
        axpos = axs2_[1, i].get_position()
        cax = fig2_.add_axes([axpos.x0 + 0.11, axpos.y0, 0.015, axpos.height])
        fig2_.colorbar(p, cax=cax)

    fig2_.savefig(plots_dir / 'disk_comp_static.svg', transparent=True)
    plt.show()
    return


@app.cell(disabled=True)
def _(coil_signals_koma, nufft, plt):
    adjoint_manual = nufft.adj_op(coil_signals_koma)
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))
    _axs[0].imshow(abs(adjoint_manual)[adjoint_manual.shape[0] // 2, :, :], cmap='viridis')
    _axs[1].imshow(abs(adjoint_manual)[:, adjoint_manual.shape[1] // 2, :], cmap='viridis')
    _axs[2].imshow(abs(adjoint_manual)[:, :, adjoint_manual.shape[2] // 2], cmap='viridis')
    _fig.suptitle('NuFFT dynamic MRI reconstruction (no phase separation)')
    plt.show()
    return


@app.cell(disabled=True)
def _(coil_signals_mr0, nufft, plt):
    adjoint_manual2 = nufft.adj_op(coil_signals_mr0)
    plt.imshow(abs(adjoint_manual2)[30:90, 60:120, adjoint_manual2.shape[2] // 2], cmap='viridis')
    plt.show()
    return


@app.cell(disabled=True)
def _(coil_signals_mr0_t2s, nufft, plt):
    adjoint_manual3 = nufft.adj_op(coil_signals_mr0_t2s)
    plt.imshow(abs(adjoint_manual3)[30:90, 60:120, adjoint_manual3.shape[2] // 2], cmap='viridis')
    plt.show()
    return


@app.cell
def _(kspace_times):
    breath_time, breath_phases = 1.0, 16
    breath_idx_per_sample = (kspace_times / breath_time).astype(int)
    phase_per_sample = ((kspace_times % breath_time) * (breath_phases / breath_time)).astype(int)

    print(phase_per_sample, breath_phases)
    return breath_phases, breath_time, phase_per_sample


@app.cell
def _(breath_phases, phase_per_sample, plt):
    plt.hist(phase_per_sample, bins=breath_phases)
    return


@app.cell
def _(
    BACKEND,
    breath_phases,
    coil_signals_koma,
    coil_signals_mr0,
    coil_signals_mr0_t2s,
    dcf,
    flat_traj,
    mrinufft,
    normed_ktraj,
    np,
    output_shape,
    phase_per_sample,
    plots_dir,
    plt,
    samples_per_adc,
    trange,
):
    def dynreco1(signals, flip_axes=False, filename: str = None):
        phase_resolution = 1
        ncols = 4
        subsamp_plot = 2
        _fig, _axs = plt.subplots(ncols=breath_phases // subsamp_plot, nrows=2, figsize=(8, 2))
        rendered_frames = np.zeros((breath_phases // phase_resolution, *output_shape), dtype=np.complex64)
        rendered_frames = np.squeeze(rendered_frames)
        rendered_frames_optim = np.zeros((breath_phases // phase_resolution, *output_shape), dtype=np.complex64)
        rendered_frames_optim = np.squeeze(rendered_frames_optim)
        nspokes = []
        lastim = None
        for plot_idx, _phase in enumerate(trange(0, breath_phases, phase_resolution, position=0, leave=True)):
            this_phase_traj = normed_ktraj[:flat_traj.shape[0], (phase_per_sample >= _phase) & (phase_per_sample < _phase + phase_resolution)]
            this_weights = dcf[(phase_per_sample >= _phase) & (phase_per_sample < _phase + phase_resolution)]
            this_signals = signals[:flat_traj.shape[0]][(phase_per_sample >= _phase) & (phase_per_sample < _phase + phase_resolution)]
            
            nspokes.append(this_signals.shape[0] / samples_per_adc)
            this_nufft = mrinufft.get_operator(BACKEND)(this_phase_traj.T, shape=output_shape, density=this_weights, squeeze_dims=True, upsampfac=2.0)
            this_recon = this_nufft.adj_op(this_signals)
            if flip_axes:
                this_recon = this_recon[::-1, ::-1]
            rendered_frames[_phase // phase_resolution] = this_recon
            if _phase % subsamp_plot == 0:
                ax = _axs[:, _phase // subsamp_plot]
                lastim = ax[0].imshow(np.abs(this_recon), cmap='viridis')
                ax[0].axis('off')
                lastphase = ax[1].imshow(np.angle(this_recon), cmap='hsv', vmin=-np.pi, vmax=np.pi)
                ax[1].axis('off')
        lastppos = ax[1].get_position()

        # Define a new axes for the colorbar: [left, bottom, width, height] in figure coords
        cax = _fig.add_axes([lastppos.x0 + 0.1, lastppos.y0, 0.015, lastppos.height])

        _fig.colorbar(lastphase, cax=cax)
        print('Spokes per phase:', np.round(nspokes).astype(int))
        
        if filename is not None:
            _fig.savefig(plots_dir / filename, transparent=True)
        plt.show()

        return rendered_frames
    rf_koma = dynreco1(coil_signals_koma, flip_axes=False, filename='disk_img_koma.svg')
    rf_mrzero = dynreco1(coil_signals_mr0, flip_axes=False, filename="disk_img_mr0.svg")
    rf_mrzero_t2s = dynreco1(coil_signals_mr0_t2s, flip_axes=False, filename="disk_img_mr0_with_t2s.svg")
    return rf_koma, rf_mrzero, rf_mrzero_t2s


@app.cell
def _(
    breath_phases,
    np,
    plots_dir,
    plt,
    rf_koma,
    rf_mrzero,
    rf_mrzero_t2s,
    trange,
):
    def comp_reco(signals1, signals2, pmax=1, filename: str = None):
        phase_resolution = 1
        ncols = 4
        subsamp_plot = 2
        _fig, _axs = plt.subplots(ncols=breath_phases // subsamp_plot, nrows=2, figsize=(8, 2))
        rendered_frames = np.zeros_like(signals1, dtype=np.complex64)
        maxdiffs = []
        lastim = None
        lastphase = None
        for plot_idx, _phase in enumerate(trange(0, breath_phases, phase_resolution, position=0, leave=True)):
            if _phase % subsamp_plot == 0:
                ax = _axs[:, _phase // subsamp_plot]
                lastim = ax[0].imshow(np.abs(signals1[_phase]) - np.abs(signals2[_phase]), cmap='viridis', vmin=0, vmax=pmax)
                ax[0].axis('off')
                lastphase = ax[1].imshow(np.angle(signals1[_phase]) - np.angle(signals2[_phase]), cmap='hsv', vmin=-np.pi, vmax=np.pi)
                ax[1].axis('off')

        lastmpos = ax[0].get_position()
        lastppos = ax[1].get_position()
        caxm = _fig.add_axes([lastmpos.x0 + 0.1, lastmpos.y0, 0.015, lastmpos.height])
        caxp = _fig.add_axes([lastppos.x0 + 0.1, lastppos.y0, 0.015, lastppos.height])

        _fig.colorbar(lastim, cax=caxm)
        _fig.colorbar(lastphase, cax=caxp)
        maxdiffs = np.array(maxdiffs)

        if filename is not None:
            _fig.savefig(plots_dir / filename, transparent=True)

        plt.show()

        return rendered_frames
    _ = comp_reco(rf_koma, rf_mrzero, pmax=0.025, filename='disk_diffs_no_t2s.svg')
    _ = comp_reco(rf_mrzero, rf_mrzero_t2s, pmax=0.25, filename='disk_diffs_with_t2s.svg')
    return


@app.cell
def _(breath_phases, breath_time, np, plt, rf_koma, rf_mrzero, rf_mrzero_t2s):
    from imageio.v3 import imwrite
    def make_animation(output_file_path: str, rendered_frames):
        outfile_anim_full = output_file_path
        phase_resolution=1
        clip_value=255
        animation_frames = []
        for _phase, time_sample in enumerate(255 * (np.abs(rendered_frames) / np.max(np.abs(rendered_frames)))):
            time_sample = np.clip(time_sample, 0, clip_value)
            time_fig, time_axs = plt.subplots(1, 1, figsize=(15, 5))
            time_axs.imshow(np.abs(time_sample))
            time_fig.suptitle(f'NuFFT dynamic MRI reconstruction (phase {_phase * phase_resolution} of {breath_phases})')
            time_fig.canvas.draw()
            image = np.frombuffer(time_fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(time_fig.canvas.get_width_height()[::-1] + (4,))
            image = image[..., 1:]
            animation_frames.append(image)
            plt.close(time_fig)
        imwrite(outfile_anim_full, animation_frames, duration=1000 * (breath_time / breath_phases * phase_resolution), extension='.gif', loop=0)

    make_animation('frames_koma.gif', rf_koma)
    make_animation('frames_mrzero.gif', rf_mrzero)
    make_animation('frames_mrzero_t2s.gif', rf_mrzero_t2s)
    return


if __name__ == "__main__":
    app.run()
