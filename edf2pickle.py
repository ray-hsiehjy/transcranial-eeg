import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from eeg_util import Edf_to_PickledArray

main_folder = os.path.join(".")
args = [[i, main_folder] for i in range(1, 25, 1)]

for sec in range(4, 36, 4):
    kwarg = {"preictal_length": sec}

    with ProcessPoolExecutor() as executor:
        results = [executor.submit(Edf_to_PickledArray, *arg, **kwarg) for arg in args]

        for future in as_completed(results):
            print(future.result())
