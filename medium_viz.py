#!/usr/bin/python3
import argparse
import time
import logging
import random
import numpy as np
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

from pythonosc.udp_client import SimpleUDPClient
oscTidal = SimpleUDPClient("127.0.0.1", 6010)  # Create client

import paho.mqtt.publish as publish
import random

def send_message_to_tidal(name, value):
    # publish.single(f"/brain/{name}", str(value), hostname="crystal.local")
    oscTidal.send_message("/ctrl", [name, value])

def send_message_to_servo(number, value):
    publish.single(f"/servos/{str(number)}", str(value), hostname="crystal.local")

def send_message_to_mqtt(path, value):
    publish.single(path, str(value), hostname="192.168.1.1")

last_feature_vector = ["0","0","0","0","0","0","0","0","0","0"];

def send_feature_vector_to_mqtt(features):
    messages = []    
    for count, feature in enumerate(reversed(features)):
        formatted_feature = "{:.2f}".format(feature)
        if formatted_feature != last_feature_vector[count]:
            messages.append({"topic": f"/mirror/{10-count}/position", "payload": formatted_feature})
    publish.multiple(messages, hostname="192.168.1.1")

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)
        self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
        self.feature_bands = [1,2,3,4,5,6,7,8,9,10]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 100
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.num_points_big = self.window_size * self.sampling_rate * 2
        self.filtered_feature_data = np.zeros(shape=(len(self.feature_bands),int(self.window_size*(1000 / self.update_speed_ms))), dtype=float)

        self.mental_states = ['relaxed','concentrated']
        self.mental_state_data = np.zeros(shape=(len(self.mental_states),int(self.window_size*(1000 / self.update_speed_ms))), dtype=float)
        

        self.lastMoveTime = time.time()
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
        self.lastDirection = -1

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
            # p.setYRange(-100,100)
            
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
            # p.setYRange(-0.1,0.1)
            
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
            # p.setYRange(-8,8)
            
            p.setTitle(f"GYRO #{i-current_curves_count}")
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

        # current_curves_count = len(self.curves)
        # for i in self.feature_bands:
        #     p = self.win.addPlot(row=i+current_curves_count,col=0)
        #     p.showAxis('left', False)
        #     p.setMenuEnabled('left', False)
        #     p.showAxis('bottom', False)
        #     p.setMenuEnabled('bottom', False)
        #     # p.setYRange(0,1.5)
            
        #     p.setTitle(f"Feature #{i}")
        #     self.plots.append(p)
        #     curve = p.plot()
        #     self.curves.append(curve)

        current_curves_count = len(self.curves)
        for count,name in enumerate(self.mental_states):
            p = self.win.addPlot(row=count+current_curves_count+1,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(0,1)
            
            p.setTitle(f"{name}")
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

            # band_power[channel][0] = DataFilter.get_band_power(psd, 1, 3) # delta
            # band_power[channel][1] = DataFilter.get_band_power(psd, 4, 7) # theta
            # band_power[channel][2] = DataFilter.get_band_power(psd, 7.5, 12.5) # mu
            # band_power[channel][3] = DataFilter.get_band_power(psd, 12.5, 15.5) # smr
            # band_power[channel][4] = DataFilter.get_band_power(psd, 8.0, 12.0) # alpha
            # band_power[channel][5] = DataFilter.get_band_power(psd, 12.0, 30.0) # beta
            # band_power[channel][6] = DataFilter.get_band_power(psd, 32.00, 59) # gamma
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
            # if count == 0:
            #     if np.average(abs(data[channel][-1])) > 0.05:
            #         # send_message_to_mqtt("/themotor/move", str(int(np.average(data[channel][-1])*5000)))
            #         print(np.average(data[channel][-1]))

        for count, channel in enumerate(self.gyro_channels):
            self.curves[count+len(self.eeg_channels)+len(self.accel_channels)].setData(data[channel][-self.num_points:])
        
        # print(f"feature band {self.feature_bands}")

        #     if count == 2:
        #         movement = np.average(data[channel][-100:])
        #         if abs(movement) > 20:
        #             send_message_to_mqtt("/theground/rotation", str(int(movement*10000)))

            # if count == 0:
            #     movement = np.average(data[channel][-100:])
            #     if abs(movement) > 20:
            #         send_message_to_mqtt("/themotor/move", str(int(movement*100)))


            # if count == 1:
            #     movement = np.average(data[channel][-100:])
            #     if abs(movement) > 20:
            #         send_message_to_mqtt("/thesun/rotation", str(int(movement*10000)))

            # print(np.average(data[channel][-100:]))



        # average_band_power = np.mean(band_power, axis=0)
        # print(average_band_power)
        # oscTidal.send_message("/ctrl", ['delta', average_band_power[0]])
        # oscTidal.send_message("/ctrl", ['theta', average_band_power[0]])
        # oscTidal.send_message("/ctrl", ['mu', relaxation_value])
        # oscTidal.send_message("/ctrl", ['smr', relaxation_value])
        # oscTidal.send_message("/ctrl", ['alpha', relaxation_value])
        # oscTidal.send_message("/ctrl", ['beta', relaxation_value])
        # oscTidal.send_message("/ctrl", ['gamma', relaxation_value])

        # print(f"alpha/beta {average_band_power[4]/average_band_power[5]}")
        
        bands = DataFilter.get_avg_band_powers(filtered_data, self.eeg_channels, self.sampling_rate, True)
        print(f"AvgBand: {bands[0]}, StdBand: {bands[1]}")

        # send_message_to_mqtt("/servos/1", bands[1][0])
        # send_message_to_mqtt("/servos/2", bands[1][1])
        # send_message_to_mqtt("/servos/0", bands[1][2])
        # send_message_to_mqtt("/servos/5", 1.0 - bands[1][3])
        # send_message_to_mqtt("/servos/6", 1.0 - bands[1][4])


        # send_message_to_mqtt("/servos/3", bands[0][0])
        # send_message_to_mqtt("/servos/4", bands[0][1])

        # send_message_to_mqtt("/servos/0", 1.0 - bands[1][1])
        

        feature_vector = np.concatenate((bands[0], bands[0]))
        send_feature_vector_to_mqtt(feature_vector)

        # print(f"feature_vector {feature_vector}")


        self.filtered_feature_data = np.append(self.filtered_feature_data, feature_vector[:,None], axis=1)
        self.filtered_feature_data = np.delete(self.filtered_feature_data, 0, axis=1)

        # for count, channel in enumerate(self.feature_bands):
        #     self.curves[count+len(self.eeg_channels)+len(self.accel_channels)+len(self.gyro_channels)].setData(self.filtered_feature_data[count][-self.num_points:])
        # print(len(self.filtered_feature_data[0]))

        # print(f"Feature: {feature_vector[:,None]}")
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

        # send_message_to_mqtt("/mirror/10/position", bands[1][0]);

        self.mental_state_data = np.delete(np.append(self.mental_state_data, [[relaxation_value],[concentration_value]], axis=1), 0, axis=1)
        
        for count, channel in enumerate(self.mental_states):
            self.curves[count+len(self.eeg_channels)+len(self.accel_channels)+len(self.gyro_channels)].setData(self.mental_state_data[count][-self.num_points:])
        
        # send_message_to_tidal('concentration', concentration_value)
        # send_message_to_tidal('relaxation', relaxation_value)
        # send_message_to_servo(2,relaxation_value)
        # send_message_to_mqtt("/thelight/1/brightness", concentration_value)
        # send_message_to_mqtt("/thesun/light/1/brightness", relaxation_value)
        # send_message_to_mqtt("/thesun/light/2/brightness", relaxation_value)
        # timeNow = time.time()
        # if (relaxation_value < 0.3 and (timeNow - self.lastMoveTime) > 3.5):
        #     self.lastMoveTime = timeNow
        #     print("move")
        #     if self.lastDirection == -1:
        #         # send_message_to_mqtt("/themotor/move", "-" + str(random.randrange(1000,10000)))
        #         self.lastDirection = 1
        #     elif self.lastDirection == 1:
        #         # send_message_to_mqtt("/themotor/move", str(random.randrange(1000,10000)))
        #         self.lastDirection = -1
        print("relax " + str(relaxation_value));

        print("con " + str(concentration_value));
        oscTidal.send_message("/ctrl", ['relaxation', relaxation_value])
        oscTidal.send_message("/ctrl", ['concentration', concentration_value])
        oscTidal.send_message("/ctrl", ['irelaxation', int(relaxation_value*100)])
        oscTidal.send_message("/ctrl", ['iconcentration', int(concentration_value*10)])

        oscTidal.send_message("/ctrl", ['alpha', bands[0][0]])
        oscTidal.send_message("/ctrl", ['beta', bands[0][1]])
        oscTidal.send_message("/ctrl", ['theta', bands[0][2]])
        oscTidal.send_message("/ctrl", ['delta', bands[0][3]])
        oscTidal.send_message("/ctrl", ['gamma', bands[0][4]])

        oscTidal.send_message("/ctrl", ['ialpha', int(bands[0][0]*10)])
        oscTidal.send_message("/ctrl", ['ibeta', int(bands[0][1]*10)])
        oscTidal.send_message("/ctrl", ['itheta', int(bands[0][2]*10)])
        oscTidal.send_message("/ctrl", ['idelta', int(bands[0][3]*10)])
        oscTidal.send_message("/ctrl", ['igamma', int(bands[0][4]*10)])

        self.app.processEvents()

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    # params.other_info = '8'
    # params.ip_address = '224.0.0.1'
    # params.ip_port = 6666
    print(params)
    # params.other_info = '8'
    # params.file = '/home/a0n/Unicorn-Suite-Hybrid-Black/MNE-Testdump.gtec'

    try:
        board_shim = BoardShim(8, params)
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



# def main():
#     BoardShim.enable_dev_board_logger()
#     logging.basicConfig(level=logging.DEBUG)

#     params = BrainFlowInputParams()
#     params.other_info = '8'
#     params.ip_address = '224.0.0.1'
#     params.ip_port = 6666
#     print(params)
#     # params.other_info = '8'
#     # params.file = '/home/a0n/Unicorn-Suite-Hybrid-Black/MNE-Testdump.gtec'

#     try:
#         board_shim = BoardShim(-2, params)
#         board_shim.prepare_session()
#         board_shim.start_stream(250000)
#         g = Graph(board_shim)
#     except BaseException as e:
#         logging.warning('Exception', exc_info=True)
#     finally:
#         logging.info('End')
#         if board_shim.is_prepared():
#             logging.info('Releasing session')
#             board_shim.release_session()


if __name__ == '__main__':
    main()