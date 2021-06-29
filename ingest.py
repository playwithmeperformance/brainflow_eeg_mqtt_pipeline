#!/usr/bin/python3
import argparse
import time
import numpy as np
import pandas as pd

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    board = BoardShim(8, params)
    board.prepare_session()
    board.start_stream(45000, 'streaming_board://224.0.0.1:6666')
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    # time.sleep(5)
    # data = board.get_board_data()
    # print(data)
    # BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'dumping first 5 seconds of measurement')

    time.sleep(60)
    data = board.get_board_data()
    
    board.stop_stream()
    board.release_session()

    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(8)
    df = pd.DataFrame(np.transpose(data))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Last Data From the Board')
    print(df.tail(10))

    # # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    # DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
    # restored_data = DataFilter.read_file('test.csv')
    # restored_df = pd.DataFrame(np.transpose(restored_data))
    # print('Data From the File')
    # print(restored_df.tail(30))


if __name__ == "__main__":
    main()