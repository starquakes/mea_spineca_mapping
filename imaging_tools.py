import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
from matplotlib.path import Path as mplpath
from tqdm import tqdm
import statsmodels.api as sm
from sklearn import linear_model
from datetime import datetime

import roiextractors as re
import belextractors as be
import spikeextractors as se

import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as mplpath
from matplotlib.image import AxesImage

import scipy.ndimage as ndimg
import neo as neo
import quantities as pq
import bisect as bs
import scipy.stats as stats
from pynwb import NWBHDF5IO
from elephant import kernels
from elephant.statistics import instantaneous_rate
from joblib import Parallel, delayed


class SelectFromImage:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, im, selection_color='y'):
        assert isinstance(im, AxesImage)
        self.ax = im.axes
        self.canvas = self.ax.figure.canvas
        self.selection_color = selection_color
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.shape = im.get_size()
        self.rois = []
        self.lines = []

        self.ax.set_title("Press enter to return ROIs")
        # self.canvas.mpl_connect("key_press_event", accept, )

    def _find_pixels_in_polygon(self, verts):
        x, y = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        p = mplpath(verts)  # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(self.shape[1], self.shape[0]).T

        return np.array(np.where(mask == 1)).T

    def onselect(self, verts):
        roi = self._find_pixels_in_polygon(verts)
        v = np.array(verts)
        l = self.ax.plot(v[:, 0], v[:, 1], color=self.selection_color)
        self.lines.append(l)
        self.rois.append(roi)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        for l in self.lines:
            l.pop(0).remove()
        self.ax.set_title(f"Selected {len(self.rois)} ROIs")
        self.canvas.draw_idle()


def get_frame_number(recording, index):
    bitvals = recording._signals[-2:, index]
    frameno = bitvals[1] << 16 | bitvals[0]
    return frameno


def extract_roi_activity(imaging, roi, return_mean=True):
    n_frames = imaging.get_num_frames()
    if return_mean:
        trace = np.zeros(n_frames)
    else:
        trace = np.zeros((len(roi), n_frames))

    for i in tqdm(np.arange(n_frames), ascii=True, desc=f"Extracting ROI of size {len(roi)}"):
        if return_mean:
            trace[i] = np.mean(imaging.get_frames(i)[roi[:, 1], roi[:, 0]])
        else:
            trace[:, i] = imaging.get_frames(i)[roi[:, 1], roi[:, 0]]

    return trace


# extends df_f by 'winperc' method; previous kept for compatibility
def dff(trace, method='winperc', window, percentile):
    df = np.zeros_like(trace)
    
    if method == 'quantile':
        q = np.percentile(trace, [0.1, 0.7])
        tq = np.where((trace > q[0]) & (trace < q[1]))
        f0 = np.median(trace[tq])
        baseline = f0

        df = (trace - f0) / f0
    elif method == 'window':
        baseline = np.zeros_like(trace)
        half_w = window // 2
        for i, t in enumerate(trace):
            if i - half_w < 0:
                f0 = np.median(trace[: i + half_w])
            elif i + window // 2 > len(trace) - 1:
                f0 = np.median(trace[i - half_w:])
            else:
                f0 = np.median(trace[i - half_w : i + half_w])
            baseline[i] = f0
            df[i] = (t - f0) / f0
    elif method == 'winperc':
        baseline = np.zeros_like(trace)
        half_w = window // 2
        for i, t in enumerate(trace):
            if i - half_w < 0:
                f0 = np.percentile(trace[: i + half_w], percentile)
            elif i + window // 2 > len(trace) - 1:
                f0 = np.percentile(trace[i - half_w:], percentile)
            else:
                f0 = np.percentile(trace[i - half_w : i + half_w], percentile)
            baseline[i] = f0
            cdf = (t - f0) / f0
            df[i] = cdf
    elif method == 'winperc2':
        baseline = ndimg.percentile_filter(trace, percentile, window)
        df = np.divide(np.subtract(trace, baseline), baseline)
            
    return df, baseline


def find_pixels_in_polygon(verts, shape):
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    p = mplpath(verts)  # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(shape[1], shape[0]).T

    return np.array(np.where(mask == 1)).T


def compute_avg_video(imaging, stride=100):
    # compute average video
    avg_video = np.mean(imaging.get_frames(np.arange(0, imaging.get_num_frames(), stride)), 0)

    return avg_video


def demix_spine_shaft(df_shaft, df_spine, method='huber', plot_fit=False):
    """
    Demix spine activity from shaft activity

    Parameters
    ----------
    df_shaft: np.array
        The shaft dff trace
    df_spine: np.array
        The spine dff trace
    method: str
        The method to be used: "rlm" (statsmodel) - "ransac" (sklearn), "huber" (sklearn), "theilsen" (sklearn)
    plot_fit: bool
        If True, the linear regressor is plotted

    Returns
    -------
    demixed_spine: np.array
        The demixed spine trace
    shaft_contribution: np.array
        The shaft contribution to the spine
    slope: float
        Slope of the linear fit
    offset: float
        Intercept of the linear fit

    """
    assert method in ['rlm', 'ransac', 'huber', 'theilsen']

    if method == 'rlm':
        model = sm.RLM(df_shaft, df_spine)
        rlm_results = model.fit()
        slope = rlm_results.params[0]
        offset = 0
        shaft_contribution = rlm_results.params[0] * df_shaft
        demixed_spine = df_spine - shaft_contribution
    elif method == 'ransac':
        ransac = linear_model.RANSACRegressor()
        ransac.fit(df_shaft.reshape(-1, 1), df_spine)
        slope = ransac.estimator_.coef_[0]
        offset = ransac.estimator_.intercept_
        shaft_contribution = slope * df_shaft + offset
        demixed_spine = df_spine - shaft_contribution

    elif method == 'huber':
        huber = linear_model.HuberRegressor()
        huber.fit(df_shaft.reshape(-1, 1), df_spine)
        slope = huber.coef_[0]
        offset = huber.intercept_

        shaft_contribution = slope * df_shaft + offset
        demixed_spine = df_spine - shaft_contribution

    elif method == 'theilsen':
        theil = linear_model.TheilSenRegressor()
        theil.fit(df_shaft.reshape(-1, 1), df_spine)
        slope = theil.coef_[0]
        offset = theil.intercept_

        shaft_contribution = slope * df_shaft + offset
        demixed_spine = df_spine - shaft_contribution

    if plot_fit:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(df_shaft, df_spine, color='gold', marker='.',
                   label='Original')
        ax.plot(df_shaft, shaft_contribution, color='cornflowerblue', label='Shaft contribution')
        ax.set_xlabel("Shaft (a.u.)")
        ax.set_ylabel("Spine (a.u.)")

    return demixed_spine, shaft_contribution, slope, offset


def select_rois(image, ax=None, use_log=True, cmap='viridis'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if use_log:
        axim = ax.imshow(np.log(image), cmap=cmap)
    else:
        axim = ax.imshow(image, cmap=cmap)

    ls = SelectFromImage(axim)

    def accept(event):
        if event.key == "enter":
            ls.disconnect()
            ls.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    return ls.rois



def sync_MEA_imaging(mea_file, tiff_file, sc_folder=None):
    """
    Loads and syncs MEA recording, SpykingCircus sorting and imaging

    Parameters
    ----------
    mea_file: str / Path
    tiff_file: str / Path
    sc_folder: str / Path

    Returns
    -------
    sync_mea: RecordingExtractor
    sync_imag: ImagingExtractor
    sync_sort: SortingExtractor
    ttls: dict
        Dictionary with synced frames for 'rising' and 'falling' frames

    """
    # load image info
    imag = re.TiffImagingExtractor(tiff_file, sampling_frequency=1)  # sampling frequency has to be manually defined
    # load MEA file
    recording = be.Mea1kRecordingExtractor(mea_file, load_spikes=False)

    # load ttl times
    ttl, states = recording.get_ttl_events()
    rising = ttl[states == 1]
    falling = ttl[states == -1]
    rising = rising[0:imag.get_num_frames()]
    falling = falling[0:imag.get_num_frames()]
    start_ttl = int(rising[0])
    end_ttl = int(falling[-1])

    # duration of simultaneous recordings
    duration = (end_ttl - start_ttl) / recording.get_sampling_frequency()
    # correct tiny shifts (due to camera shuttle switch) of sampling rate from imag for alignment
    corrected_imag_sampling = imag.get_num_frames() / duration
    print(f"MEA duration: {duration}")
    print(f"Ca2+ corrected sampling rate: {corrected_imag_sampling}")

    # reload corrected imag
    imag_sync = re.TiffImagingExtractor(tiff_file, sampling_frequency=corrected_imag_sampling)

    if sc_folder is not None:
        # load sorted MEA data
        sc_folder = Path(sc_folder)
        sorting_SC = se.SpykingCircusSortingExtractor(sc_folder)
        sorting_SC_sync = se.SubSortingExtractor(sorting_SC, start_frame=start_ttl,
                                                 end_frame=end_ttl)  # take the cutout with simultaneous imaging
    else:
        sorting_SC_sync = None
    recording_sync = se.SubRecordingExtractor(recording, start_frame=start_ttl,
                                              end_frame=end_ttl)

    ttls = {"rising": rising - start_ttl, "falling": falling - start_ttl}
    
    rising = ttls["rising"][0:imag.get_num_frames()]
    falling = ttls["falling"][0:imag.get_num_frames()]

    shutter_frame_duration = np.median(falling - rising)
    image_frames = (rising - rising[0] + shutter_frame_duration // 2).astype(int)
    imag_sync.set_times(recording_sync.frame_to_time(image_frames))

    return recording_sync, imag_sync, sorting_SC_sync, ttls

def load_nwb_file(nwb_file_path):
    # load nwb files 
    
    mea_fs = 20000
    sorting = se.NwbSortingExtractor(nwb_file_path, sampling_frequency=mea_fs)
    imag = re.NwbImagingExtractor(nwb_file_path)
    
    # get time stamps for downsampling
    
    with NWBHDF5IO(nwb_file_path, "r") as io:
        read_nwbfile = io.read()
        nwbfile = io.read()
        mea_num_frames= nwbfile.acquisition['ElectricalSeries_raw'].data.shape[0]
    
    imag_times = imag.frame_to_time(np.arange(imag.get_num_frames()))
    mea_times = np.arange(mea_num_frames) / mea_fs
    mea_duration = mea_num_frames / mea_fs
    ds_idxs = np.searchsorted(mea_times, imag_times)
    
    return sorting, imag, imag_times, mea_times, mea_duration, ds_idxs
    


def get_recording_start_time(mea_file):

    # get correct start time
    rec = be.Mea1kRecordingExtractor(mea_file)
    date_str = rec._filehandle['time'][0].decode()
    date_str_split = date_str.split("\n")[0][date_str.find(
        "start:") + len("start:") + 1:date_str.find(";")]


    date = datetime.fromisoformat(date_str_split)
    return date

def convert_to_neo(sorting, duration):
    """
    Converts a SortingExtractor to a list of NEO spike trains.

    Parameters
    ----------
    sorting: SortingExtractor
        The spikeinteface sorting extractor object
    duration: float
        Duration in seconds
    Returns
    -------
    spike_trains: list
        List of NEO SpikeTrain objects
    """
    # convert to Neo spiketrains for convolution
    spiketrains = []
    for u in sorting.get_unit_ids():
        spiketrain = sorting.get_unit_spike_train(u) / sorting.get_sampling_frequency()
        neo_st = neo.SpikeTrain(times=spiketrain * pq.s, t_stop=duration * pq.s,
                                sampling_rate=sorting.get_sampling_frequency() * pq.Hz)
        spiketrains.append(neo_st)
    return spiketrains

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    title : bool or string, optional (default = True)
        if True, show standard title. If False or empty string, doesn't show
        any title. If string, shows string as title.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    # >>> from detect_peaks import detect_peaks
    # >>> x = np.random.randn(100)
    # >>> x[60:81] = np.nan
    # >>> # detect all peaks and plot data
    # >>> ind = detect_peaks(x, show=True)
    # >>> print(ind)
    # >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # >>> # set minimum peak height = 0 and minimum peak distance = 20
    # >>> detect_peaks(x, mph=0, mpd=20, show=True)
    # >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    # >>> # set minimum peak distance = 2
    # >>> detect_peaks(x, mpd=2, show=True)
    # >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # >>> # detection of valleys instead of peaks
    # >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)
    # >>> x = [0, 1, 1, 0, 1, 1, 0]
    # >>> # detect both edges
    # >>> detect_peaks(x, edge='both', show=True)
    # >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # >>> # set threshold = 2
    # >>> detect_peaks(x, threshold = 2, show=True)
    # >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # >>> fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    # >>> detect_peaks(x, show=True, ax=axs[0], threshold=0.5, title=False)
    # >>> detect_peaks(x, show=True, ax=axs[1], threshold=1.5, title=False)
    Version history
    ---------------
    '1.0.6':
        Fix issue of when specifying ax object only the first plot was shown
        Add parameter to choose if a title is shown and input a title
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            no_ax = True
        else:
            no_ax = False

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        if title:
            if not isinstance(title, str):
                mode = 'Valley detection' if valley else 'Peak detection'
                title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')" % \
                        (mode, str(mph), mpd, str(threshold), edge)
            ax.set_title(title)
        # plt.grid()
        if no_ax:
            plt.show()
            
def find_bin_id(histogram_output):
    bin_id = []

    for i, n in enumerate(histogram_output):
        bin_id.append([i] * n)

    result_id = np.concatenate(bin_id)
    return result_id

def generate_surrogates(n_surrogates, frame_num, release_probs, spiketrains, best_match,
                        ds_idxs, std_noise, std_signal, bs_s, PLEN, PPRCTL,
                        method='GT_single'):
    """
    Generate ground-truth surrogates and test surrogates.
    
    Prameters
    --------------------------------------------------
    n_surrogates: number of simulated surrogates
    
    frame_num: length of surrogates
    
    release_probs: numpy array, release probabilities
    
    spiketrains: list
     List of NEO SpikeTrain objects
    best_match: best_matching unit
    
    ds_idxs: downsampling indices
    
    std_noise, std_signal: recording signal-to-noise ratio, the standard deviation of recording noise and signal
    
    bs_s: baseline of imaging fluorescence
    
    PLEN, PPRCTL: number of data points of moving stride, moving percentile
    
    method: GT_single for generation of surrogates from best matching unit 
     GT_all for generation of surrogates from all recorded unit spike trains
    """
    # kernel for convolution
    time_array = np.linspace(-1, 4, num=100) * pq.s
    kernel = kernels.ExponentialKernel(sigma=0.5 * pq.s)
    kernel_time = kernel(time_array)

    assert method in ['GT_single', 'GT_all']

    if method == 'GT_single':
        spiketrain = spiketrains[best_match]
        surrogates = np.zeros((n_surrogates, frame_num))

        N = len(spiketrain)
        n = 1  # flipping the coin once per time to get 0 or 1
        for j in tqdm(range(n_surrogates), desc="Generating surrogates for single spike train"):
            Pr = release_probs[j]  # release probility
            idx = sorted(np.random.permutation(np.arange(N))[:np.int(np.ceil(Pr * N))])

            p_spiketrain = spiketrain.times[idx]
            neo_st_p = neo.SpikeTrain(times=p_spiketrain, t_stop=spiketrain.t_stop,
                                      sampling_rate=spiketrain.sampling_rate)

            ifr_p = instantaneous_rate(neo_st_p, sampling_period=neo_st_p.sampling_period,
                                       center_kernel=False, kernel=kernel).squeeze()
            ifr_ds_p = ifr_p[ds_idxs]  # downsampling
            # white noise
            whitenoise = np.random.normal(0, 1, frame_num)

            k = np.array(ifr_ds_p, dtype=np.dtype(float)) / np.std(ifr_ds_p)
            k = k.magnitude * np.median(std_signal - std_noise)
            k = k + bs_s + whitenoise * np.median(std_noise)

            k_dff, bl_k = dff(np.array(k, dtype=np.dtype(float)), 'winperc2', PLEN, PPRCTL)  # delta F over F

            surrogates[j] = k_dff

    elif method == 'GT_all':
        surrogates = np.memmap(filename='all_GT_surrogates.bin', dtype='float', mode='w+',
                               shape=(len(spiketrains), n_surrogates, frame_num))

        def all_surrogates(surrogates, spiketrain, i):

            N = len(spiketrain)

            for j in range(n_surrogates):
                Pr = release_probs[j]  # release probility
                idx = sorted(np.random.permutation(np.arange(N))[:np.int(np.ceil(Pr * N))])

                p_spiketrain = spiketrain.times[idx]
                neo_st_p = neo.SpikeTrain(times=p_spiketrain, t_stop=spiketrain.t_stop,
                                          sampling_rate=spiketrain.sampling_rate)

                ifr_p = instantaneous_rate(neo_st_p, sampling_period=neo_st_p.sampling_period,
                                           center_kernel=False, kernel=kernel).squeeze()
                ifr_ds_p = ifr_p[ds_idxs]  # downsampling

                # white noise
                whitenoise = np.random.normal(0, 1, frame_num)

                # divide by standard deviation to get std = 1
                k = np.array(ifr_ds_p, dtype=np.dtype(float)) / np.std(ifr_ds_p)
                k = k.magnitude * np.median(std_signal - std_noise)
                k = k + bs_s + whitenoise * np.median(std_noise)

                k_dff, bl_k = dff(np.array(k, dtype=np.dtype(float)), 'winperc2', PLEN, PPRCTL)  # delta F over F

                surrogates[i, j] = k_dff

        Parallel(n_jobs=4)(delayed(all_surrogates)(surrogates, spiketrains[i], i)
                           for i in range(len(spiketrains)))

    return surrogates


def compute_corr_r(surrogates, n_surrogates, spiketrains, keep):
    """
    Compute Pearson's correlation R values between surrogates with ground-truth presynaptic unit and without ground_truth 
    presynaptic unit.
    
    Parameters
    ----------
    surrogates: simulated surrogates
    
    n_surrogates: number of surrogates
    
    spiketrains:list
     List of NEO SpikeTrain objects
    keep: indices of included data points out of bursting periods

    Returns
    -------

    """
    # corr_r include all unsorted R-values from correlation tests of each surrogate
    # with all spike trains
    corr_r = np.zeros((len(spiketrains), n_surrogates, len(spiketrains)))
    # corr_noGT_r contains the best Rs from correlation tests of each surrogate
    # with spike trains when removing the GT spike train
    corr_noGT = np.zeros((len(spiketrains), n_surrogates))
    # corr_GT contains the best Rs from correlation of GT surrogates with the best matched unit
    corr_GT = np.zeros((len(spiketrains), n_surrogates))

    for i in tqdm(range(len(surrogates)), desc="Computing correlation R"):
        corr = np.zeros((n_surrogates, len(spiketrains)))
        for x, spine in enumerate(surrogates[i]):
            for y, unit in enumerate(spiketrains):
                r, p = stats.pearsonr(spine[keep], unit[keep])
                corr[x, y] = r

        for k in range(n_surrogates):
            corr_r[i, k] = corr[k, :]
            match1 = np.argsort(corr[k, :])[::-1][0]
            if match1 == i:
                corr_GT[i, k] = np.sort(corr[k, :])[::-1][0]
                corr_noGT[i, k] = np.sort(corr[k, :])[::-1][1]
            else:
                corr_GT[i, k] = np.nan
                corr_noGT[i, k] = np.sort(corr[k, :])[::-1][0]

    return corr_r, corr_GT, corr_noGT


def compute_test_r(surrogates, n_surrogates, corr, spiketrains, keep, N_jobs):
    test_r = np.memmap(filename='test_r.bin', dtype='float', mode='w+',
                       shape=(len(spiketrains), n_surrogates, n_surrogates))
    """
    Compute Pearson's correlation R values between test surrogates across all recorded spike trains. 
    
    Parameters
    -------------------------------------------------------------
    surrogates: simulated surrogates
    
    n_surrogates: number of surrogates
    
    corr: correlation R values between each surrogates with all recorded unit spike trains
    
    keep: indices of included data points out of bursting periods
    
    N_jobs: number of workers to perform parallel computing
    
    """

    def para_test_r(surrogates, n_surrogates, corr, spiketrain, test_r, keep, i):
        for k in range(n_surrogates):
            second_best = np.argsort(corr[i, k])[::-1][1]

            # compute correlation of surrogates from the second best matched unit with the GT spiketrains
            for t, sp in enumerate(surrogates[second_best]):
                r, p = stats.pearsonr(sp[keep], spiketrain[keep])

                test_r[i, k, t] = r

    Parallel(n_jobs= N_jobs)(delayed(para_test_r)(surrogates, n_surrogates, corr, spiketrains[i], test_r, keep, i)
                            for i in range(len(spiketrains)))

    return test_r
