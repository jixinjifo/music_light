import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import json
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

def _normalized_linspace(size):
    return np.linspace(0, 1, size)

def interpolate(y, new_length):
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z

with open('../../in_the_spring.json', 'rb') as f:
    data = json.loads(f.read())
segments = data['segments']
duration = []
for i in segments:
	duration.append(i['duration'])
duration = np.array(duration)
pitches = []
for i in segments:
    pitches.append(i['pitches'])
pitches = np.array(pitches)

app = QtGui.QApplication([])
view = pg.GraphicsView()
layout = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(layout)
view.show()
view.setWindowTitle('Visualization')
view.resize(800,600)

plot = layout.addPlot(title='Pitches Output', colspan=3)
plot.setRange(yRange=[-0.1, 1])
plot.disableAutoRange(axis=pg.ViewBox.YAxis)
n_light = 100
x_data = np.array(range(1,n_light+1))
pen = pg.mkPen((255, 255, 255, 200), width=4)
pitch_curve = pg.PlotCurveItem(pen=pen)
pitch_curve.setData(x=x_data, y=x_data*0)
plot.addItem(pitch_curve)

for i, pitch in enumerate(pitches):
	pitch = interpolate(pitch, n_light)
	pitch = gaussian_filter1d(pitch, sigma=1)
	pitch_curve.setData(x=x_data, y=pitch)
	app.processEvents()
	time.sleep(duration[i])