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
        filtered_data = np.ndarray(shape=(19,1000), dtype=float)
        avg_bands = [0, 0, 0, 0, 0]
        band_power_alpha = []
        band_power_beta = []

        # print("render")
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            DataFilter.remove_environmental_noise(data[channel], self.sampling_rate, NoiseTypes.FIFTY.value)
            DataFilter.perform_highpass(data[channel], self.sampling_rate, 1.0, 1,
                                      FilterTypes.BUTTERWORTH.value, 1)
            filtered_data[channel] = np.array(data[channel][-int(data[channel].size/2):])
            print(f"asdasd: {type(filtered_data[channel])}")
            DataFilter.perform_lowpass(filtered_data[channel], self.sampling_rate, 60.0, 1,
                                      FilterTypes.BUTTERWORTH.value, 1)
                        
            # DataFilter.perform_bandpass(data[channel], self.sampling_rate, 31.0, 59.0, 2,
            #                           FilterTypes.BUTTERWORTH.value, 1)
            
            DataFilter.perform_wavelet_denoising(filtered_data[channel], 'coif3', 2)
            DataFilter.detrend(filtered_data[channel], DetrendOperations.LINEAR.value)

            # nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            # psd = DataFilter.get_psd_welch(filtered_data, nfft, nfft // 2, self.sampling_rate,
            #                        WindowFunctions.BLACKMAN_HARRIS.value)
            # band_power_alpha.append(DataFilter.get_band_power(psd, 7.0, 13.0))
            # band_power_beta.append(DataFilter.get_band_power(psd, 14.0, 30.0))
            if filtered_data[channel].size == self.num_points:
                self.curves[count].setData(filtered_data[channel])

        # print("aa")
        # print(f"{np.mean(band_power_alpha)} alpha, {np.mean(band_power_beta)}: {np.mean(band_power_beta) / np.mean(band_power_beta)}")
        # print(np.mean(band_power_beta))
        # print(band_power_beta)
        # print("aa")
        bands = DataFilter.get_avg_band_powers(data, self.eeg_channels, self.sampling_rate, True)
        # print("alpha/beta:%f", np.mean(band_power_alpha))
        print(bands)
        # feature_vector = np.concatenate((bands[0], bands[1]))
        # print(feature_vector)
        # #calc concentration
        # concentration_params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        # concentration = MLModel(concentration_params)
        # concentration.prepare()
        # print('Concentration: %f' % concentration.predict(feature_vector))
        # concentration.release()

        # calc relaxation
        # relaxation_params = BrainFlowModelParams(BrainFlowMetrics.RELAXATION.value, BrainFlowClassifiers.REGRESSION.value)
        # relaxation = MLModel(relaxation_params)
        # relaxation.prepare()
        # print('Relaxation: %f' % relaxation.predict(feature_vector))
        # relaxation.release()
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