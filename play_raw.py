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
    board.start_stream(45000)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    while True:
        time.sleep(5)
        data = board.get_board_data()
        print(data)
    # BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'dumping first 5 seconds of measurement')
    # input("Press Enter to stop streaming...")
    # time.sleep(60)
    # data = board.get_board_data()
    
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()