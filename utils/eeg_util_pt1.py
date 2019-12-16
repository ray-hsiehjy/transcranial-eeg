# util func in this py file convert edf to pickle

import glob
import os
import re
import pyedflib
import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
import pickle


def get_seizure_timestamps(path: str) -> dict:
    """
    Get seizure timestamps for an individual subject
    
    Parameters:
    ----------
    path: Path to subject's txt summary file
    
    Return:
    ----------
    ref_dict: Key:file_name. Value:np.ndarray, shape==(num of seizures, 2)
    """
    ref_dict = {}
    with open(path, "r") as f:
        paragraphs = f.read().split("\n\n")
        # remove extra lines and strip whitespaces
        paragraphs = [p.strip() for p in paragraphs if p != ""]
        for paragraph in paragraphs:
            # every paragraph has metadata info for one edf file
            if not paragraph.startswith("File Name:"):
                continue
            file_name = re.findall(r"chb\d{2}.*_\d{2}.*\.edf", paragraph)[0]
            # event_times is a list of timestamps of seizure events [start_1, end_1, start_2, end_2...]
            event_times = re.findall(r"\d+\sseconds", paragraph)
            # if no event in the recording file, event_times == empty list
            event_times = [int(timestamp.split(" ")[0]) for timestamp in event_times]
            # reshape into [[start_1, end_1], [start_2, end_2], ...]
            event_times = np.asarray(event_times).reshape(-1, 2)

            # assign seizure timestamps into reference dictionary
            ref_dict[file_name] = event_times

        return ref_dict


def read_edf(file_path: str, ch_names=None) -> (int, int, int, np.ndarray):
    """
    Load recoding data into a np.ndarray and extract some metadata
    
    Parameters:
    ----------
    file_path: str. path to a single edf file
    ch_names: 
        if None: return all channels in the file
        if a list. eg ["FP1-F7", "F7-T7"...]. Use standard 10-20 naming system.
    
    Return:
    ----------
    num_channel, sampling_freq, Nsamples, signal
    """
    # Context manager "with" does not close EdfReader
    f = pyedflib.EdfReader(file_path)

    # get Nsamples
    Nsamples = f.getNSamples()
    assert len(set(Nsamples)) == 1
    Nsamples = Nsamples[0]

    # get sampling_freq
    sampling_freq = f.getSampleFrequencies()
    assert len(set(sampling_freq)) == 1
    sampling_freq = sampling_freq[0]

    # get signal labels
    sig_labels = f.getSignalLabels()
    # if no specific channels, return all in original order
    if ch_names == None:
        ch_names = sig_labels
    num_channel = len(ch_names)

    # create an all-zero 2d-array
    signal = np.zeros((num_channel, Nsamples))
    # get index of wanted channels
    ch_idx = [sig_labels.index(ch) for ch in ch_names]
    # read in specified channel one at a time
    for c, ch in enumerate(ch_idx):
        signal[c, :] = f.readSignal(ch)

    # explicitly close file
    f._close()
    del f

    return num_channel, sampling_freq, Nsamples, signal


def ictal_idx(
    timestamps: np.ndarray, sec_per_segment: int, num_segment: int, preictal_length: int
) -> (list, list):
    """
    Find segments with seizure.
    
    Parameters:
    ----------
    timestamps: 2d-array from get_seizure_timestamps. 
    sec_per_segment: int in seconds
    num_segment: number of total segment for this individual edf file. eg 3600 sec // 6 sec = 600 segments
    preictal_length: int in seconds, number of seconds before seizure episodes defined as preictal

    Return:
    ----------
    ictal_lst: a list of all segments within ictal phase
    preictal_lst: a list of all segments defined as pretical
    """

    ictal_lst = []
    preictal_lst = []
    # timestamp shape (num_seizure_episodes, 2)
    for i in range(timestamps.shape[0]):
        # find segments where ictal phase starts and ends
        ictal_start, ictal_end = timestamps[i, :] // sec_per_segment
        # get all segments within ictal phase
        ictal = list(np.arange(ictal_start, ictal_end + 1))
        for element in ictal:
            if element > num_segment:
                ictal.remove(element)
        ictal_lst += ictal

        # convert seconds to number of bins
        num_preictal = preictal_length // sec_per_segment
        # get all segments within preictal phase
        preictal = list(np.arange(ictal[0] - num_preictal, ictal[0]))
        preictal_lst += preictal

    return ictal_lst, preictal_lst


def bandpower(data, sf: float, bands: list, window_sec: float, relative=False) -> list:
    """
    Compute the average power of the signal x in a specific frequency band.

    Parameters:
    ----------
    data : 1d-array. Input signal in the time-domain.
    sf : Sampling frequency of the data.
    band : Lower and upper frequencies of the band of interest.
    window_sec : Length of each window in seconds.
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return:
    ----------
    bps : a list of absolute or relative power in decible.
    """

    # Define window length. nperseg: number of samples per segment
    nperseg = window_sec * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    bps = []
    for band in bands:
        low, high = band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            bp /= simps(psd, dx=freq_res)
        # convert power unit to decible
        bp = 10 * np.log10(bp)
        bps.append(bp)
    return bps


def Edf_to_PickledArray(
    subject_id: int,
    main_folder: str,
    sec_per_segment: int = 4,
    preictal_length: int = 20,
    band_range: list = None,
    ch_names: list = None,
):
    """
    Convert all edf file from one subject to concatenated numpy array and pickle the array
    
    Parameters:
    ----------
    subject_id: int
    main_folder: main folder that holds all subfolders
    sec_per_segment: int in seconds. User defined segment length. Must >= 4
    preictal_length: int in second. User defined preictal length. Must >= 4. Better increment at sec_per_segment
    band_range: list of ranges. 
        If None, band_range = [delta, theta, alpha, beta, gamma] = [[0.5, 4], [4, 8], [8,12], [12,30], [30, 128]]
    ch_names: EEG 10-20 system names. 
        If None, ch_names = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ",
        ]

    Return:
    ----------
    create a subfolder "pickle_preictalXX" under main_folder and put three pickle files in it
        data X, shape (num_segment, num_ch, len(band_range))
        label y, shape (num_segment,): 0==interictal, 0.5==preictal, 1==ictal
        ref_dict: 
            key: filenames, value: timestamps of seizure events. 
            Key=="seg_lst", value: marks of where files are concatenated
    """

    if band_range == None:
        band_range = [[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 128]]
    if ch_names == None:
        ch_names = [
            # left four
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            # left-center
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            # right_center four
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            # right four
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            # center two
            "FZ-CZ",
            "CZ-PZ",
        ]

    # subject id, 2 digit string
    subject = f"chb{str(subject_id).zfill(2)}"

    # find all subject summary txt files
    raw_folder = os.path.join(main_folder, "raw_data")
    txt_lst = sorted(
        glob.glob(os.path.join(raw_folder, "**", "*summary.txt"), recursive=True)
    )
    # get one individual summary file
    subject_summary = [txt for txt in txt_lst if subject in txt][0]

    # get all edf recording files for a subject
    edf_lst = sorted(glob.glob(os.path.join(raw_folder, subject, "*.edf")))

    # ref_dict, key: file_name, value:2d-array [num_event, [start, end]]
    ref_dict = get_seizure_timestamps(subject_summary)

    # power spectrum list, each element is the power spectrum over time from one edf file
    ps_lst = []
    # label list, each element is 0 or 1 seizure labels for segments from one edf file
    label_lst = []
    # seg_size lst, each element is the number of segments from one edf file
    seg_lst = []

    for edf_f in edf_lst:
        file_name = os.path.split(edf_f)[1]
        # not all edf files are useful
        if file_name not in ref_dict.keys():
            continue
        print(f"Processing {file_name}")

        # read recordings and metadatea from single edf
        num_channel, sampling_freq, Nsamples, rec = read_edf(edf_f, ch_names=ch_names)

        # segment recording into fragments
        sample_per_segment = sec_per_segment * sampling_freq
        # truncate recording so that total_samples % sample_per_segment == 0
        num_segment = Nsamples // (sample_per_segment)
        rec = rec[:, : num_segment * sample_per_segment]
        # reshape rec from (channel, Nsamples) to (channel, num_segment, sample_per_segment)
        rec = rec.reshape(num_channel, num_segment, sample_per_segment)
        # move num_segment to axis 0 for later purpose
        rec = np.moveaxis(rec, 1, 0)

        # get seizure labels, 1d-array in 0s and 1s, shape (num_segment)
        timestamps = ref_dict.get(file_name, None)
        # in case there is error parsing summary.txt file
        if timestamps is None:
            print(
                f"Cannot get seizure timestamps from {file_name}, check summary.txt file"
            )
            quit()
        ictal_lst, preictal_lst = ictal_idx(
            timestamps, sec_per_segment, num_segment, preictal_length=preictal_length
        )
        assert len(set.intersection(set(ictal_lst), set(preictal_lst))) == 0

        # create an all zero-array
        label = np.zeros(num_segment)
        # replace seizure segments with 1
        label[ictal_lst] = 2
        label[preictal_lst] = 1
        label_lst.append(label)

        # calculate power for every bandwidth, value unit in decible(db)
        bandpower_param = {"sf": 256, "bands": band_range, "window_sec": 4}
        ps = np.apply_along_axis(bandpower, -1, rec, **bandpower_param)
        ps_lst.append(ps)

        seg_lst.append(num_segment)
    ps_concat = np.concatenate(ps_lst, axis=0)
    label_concat = np.concatenate(label_lst, axis=0)

    ref_dict["seg_lst"] = seg_lst

    assert ps_concat.shape[0] == label_concat.shape[0]
    assert ps_concat.shape[-2:] == (num_channel, len(band_range))

    # create destination folder if not already exists
    des_folder = os.path.join(main_folder, f"pickle_preictal{preictal_length}")
    if not os.path.isdir(des_folder):
        os.mkdir(des_folder)

    psd_fname = os.path.join(des_folder, f"{subject}_PSD.pickle")
    label_fname = os.path.join(des_folder, f"{subject}_label.pickle")
    refdict_fname = os.path.join(des_folder, f"{subject}_ref_dict.pickle")

    for fname, obj in zip(
        [psd_fname, label_fname, refdict_fname], [ps_concat, label_concat, ref_dict]
    ):
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    return f"{subject} pickled"

