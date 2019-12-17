# util func in this py file prepare data for training
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def clean_powerspectra(array: np.ndarray, threshold=-25) -> np.ndarray:
    """
    Fix recording segments where power is abnormally low across ALL channels and all spectra-bands. 
    Likely instrumental artifacts. Replace with mean of same spectral band (eg alpha, beta...) from same individual. 
    
    Parameters:
    ----------
    ary: array to be cleaned. shape (num_segment, num_channel=18, band_range=5)
    threhold: power level in decibel. 

    Return:
    ----------
    cleaned array
    """
    num_channels = array.shape[1]
    num_bands = array.shape[2]

    # find where power is abnormally low across ALL channels
    mask = (array.reshape(-1, num_channels * num_bands) < threshold).prod(axis=-1)
    mask = np.array(mask, dtype="bool")

    # replace abnormal measuring with mean of other segments.
    # array[mask].shape == (num_abnormal_segment, num_channels, num_bands)
    # array[~mask].mean(axis=0).shape == (num_channels, num_bands)
    array[mask] = array[~mask].mean(axis=0)

    return array


def get_pickled_data(
    subject_ids: list,
    preictal_length: int,
    clean_spectra=True,
    threshold=-25,
    z_score=True,
    local=False,
):
    """
    Load pickle files into concatenated array

    Parameters:
    ----------
    ids: a list of subject id in int used for testing
    clean_spectra: bool, if true apply clean_powerspectra function when loading pickle file
    threshold: only apply when clean_spectra is set True, unit in decibel
    z_score: if ture convert power to z_score 
    local: if True, set main_fd path to local main_folder. Otherwise, main_fd on google drive


    Return:
    ----------
    powers: feature matrix X shape==(number of segments, num_ch*num_band=18*5=90)
    labels: 0: interictal, 1:preictal, 2:ictal
    """
    power_lst = []
    label_lst = []
    main_fd = "/content/drive/My Drive/Insight/"
    if local == True:
        main_fd = os.path.join("D:", "\\", "ML_projects", "EEG_project")

    for subject_id in subject_ids:
        subject = f"chb{str(subject_id).zfill(2)}"
        psd_f = os.path.join(main_fd, "pickle_psd", f"{subject}_PSD.pickle")
        label_f = os.path.join(
            main_fd, "pickle_labels", f"{subject}_label{preictal_length}.pickle"
        )

        with open(psd_f, "rb") as f:
            power = pickle.load(f)
        with open(label_f, "rb") as f:
            label = pickle.load(f)

        if clean_spectra:
            power = clean_powerspectra(power, threshold=threshold)

        power = power.reshape(power.shape[0], -1)
        if z_score:
            power = StandardScaler().fit_transform(power)

        power_lst.append(power)
        label_lst.append(label)

    powers = np.concatenate(power_lst, axis=0)
    labels = np.concatenate(label_lst, axis=0)
    labels = np.array(labels, dtype="int")

    return powers, labels


def reduce_interictal(X: np.ndarray, y: np.ndarray, keep_size: int, seed=None):
    """
    Reduce the number of segments of interictal while keep that of ictal and preictal the same

    Parameters:
    ----------
    X: feature matrix X shape==(number of segments, num_ch*num_band=18*5=90)
    y: labels for X
    keep_size: number of interictal segment to be kept
    seed: np.random.seed

    Return:
    ----------
    reduced_array: reduced sized array.
    """
    interictal_idx = np.where(y == 0)[0]

    if seed != None:
        np.random.seed(seed)
    to_drop = np.random.choice(
        interictal_idx, size=len(interictal_idx) - keep_size, replace=False
    )

    reduced_X = np.delete(X, to_drop, axis=0)
    reduced_y = np.delete(y, to_drop, axis=0)

    return reduced_X, reduced_y


def construct_timesteps(
    X: np.ndarray, y: np.ndarray, label: int, Tx=2, num_interictal=3000
):
    """
    Bin every Tx consecutive segments for LSTM 

    Parameters:
    ----------
    X: feature matrix X shape==(number of segments, num_ch*num_band=18*5=90)
    y: labels for X
    label: 0==intericatl, 1==preictal, 2==ictal
    Tx: number of steps for LSTM training
    num_interictal: number of binned interictal to be returned. Only apply when label==0

    Return:
    ----------
    binned_array: binned_array. shape
    """
    idx = np.where(y == label)[0]
    indices = []  # a list of indices

    if label != 0:
        for i in idx:
            if np.prod(y[i : i + Tx]) == label ** Tx:
                indices.append(i)

    else:
        while len(indices) < num_interictal:
            i = int(np.random.choice(idx, size=1))
            if (np.prod(y[i : i + Tx]) == 0) and (i not in indices):
                indices.append(i)

    indices = np.asarray(indices)
    indices = np.concatenate([indices + i for i in range(Tx)])  # concat 1-d arrays
    indices = np.sort(indices)  # sort array

    binned_X = X[indices].reshape(-1, Tx, 90)
    binned_y = np.zeros(binned_X.shape[0], dtype="int") + label

    return binned_X, binned_y


def bin_steps(power: np.ndarray, label: np.ndarray, Tx: int):
    """
    Construct sliding window with timesteps for each sample == Tx

    """

    label_Tx2 = np.asarray(
        [np.max(label[i : i + Tx]) for i in range(len(label) - Tx + 1)], dtype="int"
    )
    power_Tx2 = np.zeros(shape=(len(label_Tx2), Tx, 90))
    for i in range(power.shape[0] - Tx + 1):
        power_Tx2[i, :, :] = power[i : i + Tx]

    return power_Tx2, label_Tx2


def split_data(
    power_Tx: np.ndarray,
    label_Tx: np.ndarray,
    num_episodes: int,
    preictal_length: int,
    seed=None,
    keep_size=(300, 75),
    shuffle=True,
):
    """
    Split train and test data for individual subject.
    Random select a fraction of interictal segments for training and testing
    shuffle data before returning 
    ----------

    Parameters:
    ----------
    power: feature matrix X shape = (num_samples, Tx, num_ch*num_bands) = (num_samples, 2, 18*5)
    label: label y
    num_episodes: number of episodes from last used for testing
    preictal_length: number of second added to "abnormal" category
    keep_size: number of interictal samples used for train and test in tuple eg, (300, 75)
    seed: randam seed for selecting interictal samples and for shuffle data

    Ruturn:
    ----------
    train_X, train_y, test_X, test_y all shuffled
    """

    # Use last several seizure edpisodes for testing
    preictal_starts = np.where(label_Tx == 2)[0][:: (preictal_length // 4)]
    cutoff = preictal_starts[-num_episodes]

    # cutoff splits train and test
    train_X, train_y = power_Tx[:cutoff], label_Tx[:cutoff]
    test_X, test_y = power_Tx[cutoff:], label_Tx[cutoff:]

    # reduce samples of interictal
    train_X, train_y = reduce_interictal(
        train_X, train_y, keep_size=keep_size[0], seed=seed
    )
    test_X, test_y = reduce_interictal(
        test_X, test_y, keep_size=keep_size[1], seed=seed
    )

    # shuffle train data
    if shuffle:
        np.random.seed(seed)
        train_order = np.random.permutation(len(train_y))
        train_X, train_y = train_X[train_order], train_y[train_order]

        np.random.seed(seed)
        test_order = np.random.permutation(len(test_y))
        test_X, test_y = test_X[test_order], test_y[test_order]

    return train_X, train_y, test_X, test_y


def merge_preictal(label: np.ndarray, how: str):
    """
    Merge preictal segment either into interictal or ictal category
    ----------

    Parameters:
    ----------
    label: the label array to be modified
    how: either "to_interictal" or "to_ictal"

    Return:
    ----------
    label: modified label array 
    label_OH: label one-hot
    class_weights: the ratio between 0 and 1 in label_m. For later training purpose
    """

    if how == "to_interictal":
        label = np.where(label == 2, 0, label)
    if how == "to_ictal":
        label = np.where(label == 2, 1, label)

    # label_OH = np.eye(2)[label]
    class_weights = compute_class_weight("balanced", np.arange(2), label)

    return label, class_weights
