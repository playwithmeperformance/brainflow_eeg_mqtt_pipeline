#!/usr/bin/python3
import argparse
import time
import logging
import random
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 100
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.num_points_big = self.window_size * self.sampling_rate * 2

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot',size=(1600, 1200))

        # self.nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()


    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-100,100)
            
            p.setTitle(f"EEG #{i}")
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
        print("Initialized Time series with", len(self.eeg_channels), "Channels")
        print("Rendering", len(self.curves), "Curves")


    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points_big)
        filtered_data = np.ndarray(shape=(19,self.num_points), dtype=float)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            DataFilter.remove_environmental_noise(data[channel], self.sampling_rate, NoiseTypes.FIFTY.value)
            DataFilter.perform_highpass(data[channel], self.sampling_rate, 0.5, 1,
                                      FilterTypes.BUTTERWORTH.value, 1)
            DataFilter.perform_lowpass(data[channel], self.sampling_rate, 60.0, 1,
                                      FilterTypes.BUTTERWORTH.value, 1)
            filtered_data[channel] = np.array(data[channel][-self.num_points:])
            self.curves[count].setData(filtered_data[channel])

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.other_info = '8'
    params.ip_address = '224.0.0.1'
    params.ip_port = 6666
    print(params)
    # params.other_info = '8'
    # params.file = '/home/a0n/Unicorn-Suite-Hybrid-Black/MNE-Testdump.gtec'

    try:
        board_shim = BoardShim(-2, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        g = Graph(board_shim)
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()