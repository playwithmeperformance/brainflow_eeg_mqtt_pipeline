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

from pythonosc.udp_client import SimpleUDPClient
oscTidal = SimpleUDPClient("127.0.0.1", 6010)  # Create client


def send_message_to_tidal(name, value):
    oscTidal.send_message("/ctrl", [name, value])

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)
        self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.num_points_big = self.window_size * self.sampling_rate * 2

        self.eeg_names = BoardShim.get_eeg_names(self.board_id)
        print(self.eeg_names)
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot',size=(1600, 1200))
        self.win.show()

        concentration_params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        self.concentration = MLModel(concentration_params)
        self.concentration.prepare()

        relaxation_params = BrainFlowModelParams(BrainFlowMetrics.RELAXATION.value, BrainFlowClassifiers.REGRESSION.value)
        self.relaxation = MLModel(relaxation_params)
        self.relaxation.prepare()

        # self.nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    # def create_eeg_curves(channel_range, ):


    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        print("init time series")
        for i in self.eeg_channels:
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-100,100)
            
            p.setTitle(self.eeg_names[i])
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

        current_curves_count = len(self.curves)
        for i in self.accel_channels:
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-0.1,0.1)
            
            p.setTitle(f"ACCEL #{i-current_curves_count}")
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
        
        current_curves_count = len(self.curves)
        for i in self.gyro_channels:
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-8,8)
            
            p.setTitle(f"GYRO #{i-current_curves_count}")
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

        print("Initialized Time series with", len(self.eeg_channels), "Channels")
        print("Rendering", len(self.curves), "Curves")


    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points_big)
        filtered_data = np.zeros(shape=(8,self.num_points), dtype=float)
        nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        # print(f"nearest: {nfft}")
        band_power = np.zeros(shape=(8,7), dtype=float)        
        # print(filtered_data)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 30, 58, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.remove_environmental_noise(data[channel], self.sampling_rate, NoiseTypes.FIFTY.value)

            # print(np.zeros(fill_count, dtype=float))

            if data[channel].size < self.num_points:
              fill_count = self.num_points - data[channel].size
              # print(fill_count)
              filtered_data[channel] = np.concatenate((np.zeros(fill_count, dtype=float), np.array(data[channel][-self.num_points:])), axis=0)
            else:
              filtered_data[channel] = np.array(data[channel][-self.num_points:])
            
            DataFilter.detrend(filtered_data[channel], DetrendOperations.CONSTANT.value)

            psd = DataFilter.get_psd_welch(filtered_data[channel], nfft, nfft // 2, self.sampling_rate,
                                   WindowFunctions.BLACKMAN_HARRIS.value)
            # DataFilter.perform_wavelet_denoising(filtered_data[channel], 'coif3', 2)

            band_power[channel][0] = DataFilter.get_band_power(psd, 1, 3) # delta
            band_power[channel][1] = DataFilter.get_band_power(psd, 4, 7) # theta
            band_power[channel][2] = DataFilter.get_band_power(psd, 7.5, 12.5) # mu
            band_power[channel][3] = DataFilter.get_band_power(psd, 12.5, 15.5) # smr
            band_power[channel][4] = DataFilter.get_band_power(psd, 8.0, 12.0) # alpha
            band_power[channel][5] = DataFilter.get_band_power(psd, 12.0, 30.0) # beta
            band_power[channel][6] = DataFilter.get_band_power(psd, 32.00, 59) # gamma
            # print(len(filtered_data[channel]) % 2)
            # fft_data = DataFilter.perform_fft(filtered_data[channel][-nfft:], WindowFunctions.NO_WINDOW.value)
            # print(fft_data)
            self.curves[count].setData(filtered_data[channel])

        for count, channel in enumerate(self.accel_channels):
            DataFilter.perform_highpass(data[channel], self.sampling_rate, 0.1, 1,
                                      FilterTypes.BUTTERWORTH.value, 1)
            DataFilter.perform_lowpass(data[channel], self.sampling_rate, 20, 1,
                                      FilterTypes.BUTTERWORTH.value, 1)

            self.curves[count+len(self.eeg_channels)].setData(data[channel][-self.num_points:])

        for count, channel in enumerate(self.gyro_channels):
            self.curves[count+len(self.eeg_channels)+len(self.accel_channels)].setData(data[channel][-self.num_points:])


        average_band_power = np.mean(band_power, axis=0)
        print(average_band_power)
        # oscTidal.send_message("/ctrl", ['delta', average_band_power[0]])
        # oscTidal.send_message("/ctrl", ['theta', average_band_power[0]])
        # oscTidal.send_message("/ctrl", ['mu', relaxation_value])
        # oscTidal.send_message("/ctrl", ['smr', relaxation_value])
        # oscTidal.send_message("/ctrl", ['alpha', relaxation_value])
        # oscTidal.send_message("/ctrl", ['beta', relaxation_value])
        # oscTidal.send_message("/ctrl", ['gamma', relaxation_value])

        # print(f"alpha/beta {average_band_power[4]/average_band_power[5]}")
        
        bands = DataFilter.get_avg_band_powers(filtered_data, self.eeg_channels, self.sampling_rate, True)
        print(bands)
        feature_vector = np.concatenate((bands[0], bands[1]))
        # print(feature_vector)
        #calc concentration
        # concentration_params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        # concentration = MLModel(concentration_params)
        # concentration.prepare()
        concentration_value = self.concentration.predict(feature_vector)
        # print('Concentration: %f' % concentration_value)
        # concentration.release()

        #calc relaxation
        
        relaxation_value = self.relaxation.predict(feature_vector)
        # relaxation.release()
        # print(f"Relaxation: {relaxation_value}, Concentration: {concentration_value}")
        send_message_to_tidal('concentration', concentration_value)
        send_message_to_tidal('relaxation', relaxation_value)

        # oscTidal.send_message("/ctrl", ['relaxation', relaxation_value])

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