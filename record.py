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
    params.other_info = '8'
    params.ip_address = '224.0.0.1'
    params.ip_port = 6666
    print(params)
    board = BoardShim(-2, params)
    board.prepare_session()
    board.start_stream(45000, 'file:///home/a0n/Unicorn-Suite-Hybrid-Black/brainflow_start/Testdump.gtec:w')
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    # time.sleep(5)
    # data = board.get_board_data()
    # print(data)
    # BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'dumping first 5 seconds of measurement')
    while True:
        time.sleep(5)
        data = board.get_board_data()
        print(data)
    input("Press Enter to stop streaming...")

    # # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    # DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
    # restored_data = DataFilter.read_file('test.csv')
    # restored_df = pd.DataFrame(np.transpose(restored_data))
    # print('Data From the File')
    # print(restored_df.tail(30))


if __name__ == "__main__":
    main()