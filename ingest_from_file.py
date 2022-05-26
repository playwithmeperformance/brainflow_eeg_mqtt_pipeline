#!/usr/bin/python3
import argparse
import time
import numpy as np
import pandas as pd
import signal

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import os, sys

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    self.kill_now = True


def main():
    BoardShim.enable_dev_board_logger()
    original_board_id = BoardIds.UNICORN_BOARD.value
    board_id = BoardIds.PLAYBACK_FILE_BOARD.value

    # use synthetic board for demo
    params = BrainFlowInputParams()
    params.other_info = str(original_board_id)
    params.file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Testdump1.gtec')

    board = BoardShim(board_id, params)

    board.prepare_session()
    board.config_board ('loopback_true')
    board.start_stream(5000, 'streaming_board://224.0.0.1:6666')
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')

    killer = GracefulKiller()
    while not killer.kill_now:
        time.sleep(1)
        print(board.get_board_data_count())
        data = board.get_board_data()
        # print(board.get_board_data_count())
        # print(len(data.flatten()))

    print("Closing Stream") 
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()