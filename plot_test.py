#!/usr/bin/python3
import pyqtgraph as pg
import numpy as np
import time

x = np.random.normal(size=1000)
y = np.random.normal(size=1000)
plotWidget = pg.plot(title="Three plot curves")
# pg.plot(x)   # data can be a list of values ora numpy array
plotWidget.plot(x, y[0], pen=(0,1))
time.sleep(10)