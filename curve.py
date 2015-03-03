#! /usr/bin/env python
# coding: utf-8

from PyQt4.Qt import *
from PyQt4.QtGui import *

import sys
import cairo
import numpy as np
import math

def norm(array, axis=0):
	return np.sqrt(np.sum(array ** 2, axis=1))

def gaussian_window(X, mu, sigma):
	G = np.exp(-(X - mu) ** 2 / (2 * sigma ** 2))
	return G / np.sum(G, axis=0)

# let's try some smoothing with springs and complicated integration methods...
def compute_v1(points, velocity_threshold, stiffness, friction, dt=0.1):
	eps = 1E-6
	
	P = np.array(points, dtype=np.float)
	V = np.zeros(P.shape)
	A = np.zeros(P.shape)
	
	L0_1 = norm(P[1:, :] - P[:-1, :], axis=1) + eps
	L0_2 = L0_1[1:] + L0_1[:-1]
	
	for iteration in xrange(1000):
		L1 = norm(P[1:, :] - P[:-1, :], axis=1) + eps
		L2 = norm(P[2:, :] - P[:-2, :], axis=1) + eps
	
		D1 = (P[1:, :] - P[:-1, :]) / L1[:, np.newaxis]
		D2 = (P[2:, :] - P[:-2, :]) / L2[:, np.newaxis]
		
		F1 = D1 * (stiffness * (L1 - L0_1))[:, np.newaxis]
		F2 = D2 * (stiffness * (L2 - L0_2))[:, np.newaxis]
		
		F1a = np.concatenate([F1, np.array([[0., 0.]])], axis=0)
		F1b = -np.concatenate([np.array([[0., 0.]]), F1], axis=0)
		F2a = np.concatenate([F2, np.array([[0., 0.], [0., 0.]])], axis=0)
		F2b = -np.concatenate([np.array([[0., 0.], [0., 0.]]), F2], axis=0)
		
		F_spring = F1a + F1b + F2a + F2b
		F_friction = 0*-F_spring * (np.minimum(friction / (norm(F_spring) + eps), 1.0))[:, np.newaxis]
		
		A = F_spring + F_friction
		#~A[0, :] = A[-1, :] = np.array([[0., 0.]])
		
		V += A * dt
		P += V * dt
		
		#~if np.all(norm(V) < velocity_threshold):
			#~break
	
	return P.tolist()

# ...well, maybe less complicated.
def compute_v2(points, velocity_threshold, stiffness, dt=0.1):
	null_vec = np.array([[0., 0.]])
	
	P = np.array(points, dtype=np.float)
	V = np.zeros(P.shape)
	A = np.zeros(P.shape)
	
	for iteration in xrange(1000):
		D1 = (P[1:] - P[:-1])		
		F1 = D1 * stiffness
		
		F1a = np.concatenate([F1, null_vec], axis=0)
		F1b = -np.concatenate([null_vec, F1], axis=0)
		
		A = F1a + F1b
		A[0] = A[-1] = null_vec
		
		V += A * dt
		P += V * dt
		
		if np.all(norm(A, axis=1) < velocity_threshold):
			break
	
	return P.tolist()

# ok, screw it. better use gaussian filter.
def compute_v3(points, sigma):
	P = np.array(points, dtype=np.float)
	
	L = norm(P[1:] - P[:-1], axis=1)
	X = np.zeros(P.shape[0])
	for i in xrange(P.shape[0] - 1):
		X[i + 1] = X[i] + L[i]
	
	X2, X1 = np.meshgrid(X, X)
	G = gaussian_window(X1, X2, sigma)

	GP = np.einsum('kj,ki->ijk', P, G)
		
	P = np.sum(GP, axis=2)
	return P.tolist()

def oversampled(points, threshold=10):
	def samples(p0, p1):
		x0, y0 = p0
		x1, y1 = p1
		length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
		n = int(length / threshold) + 1
		dx, dy = (x1 - x0) / n, (y1 - y0) / n
		return [(x0 + dx * i, y0 + dy * i) for i in xrange(n)]
		
	return sum([samples(p0, p1) for p0, p1 in
		zip(points, points[1:] + [points[-1]])], [])

def draw_path(ctx, points):
	if points:
		ctx.move_to(*points[0])
		for pt in points[1:]:
			ctx.line_to(*pt)
		ctx.stroke()
	
class Window(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		
		self.draw_mode = False
		self.points = []
		self.new_points = []
		
		# smoothing slider
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setRange(1, 1000)
		self.slider.setValue(50)

		self.slider.valueChanged.connect(lambda value: self._compute_curve())
		self.slider.valueChanged.connect(self._set_label)
		
		self.slider_label = QLabel()
		self.slider_label.setMinimumWidth(170)
		self.slider_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
		
		self._set_label(self.slider.value())

		# drawing zone
		self.image_label = QLabel()
		self.image_label.setFocus()
		self.image_label.setMinimumSize(1, 1)
		self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		
		# layouts
		slider_layout = QHBoxLayout()
		slider_layout.addWidget(self.slider_label)
		slider_layout.addWidget(self.slider)
		
		self.layout = QVBoxLayout()
		self.layout.addWidget(self.image_label)
		self.layout.addLayout(slider_layout)
		
		self.setLayout(self.layout)
		self.resize(800, 600)
		
		self._redraw()

	def _set_label(self, value):
		self.slider_label.setText('smooth radius: %s px' % str(value))

	def resizeEvent(self, event):
		self._redraw()
	
	def mousePressEvent(self, event):
		if event.button() == Qt.LeftButton:
			self.draw_mode = True
			self.points = []
			self.new_points = []
			self._add_point(event)
	
	def mouseMoveEvent(self, event):
		if self.draw_mode:
			self._add_point(event)
			
	def mouseReleaseEvent(self, event):
		if event.button() == Qt.LeftButton:
			self._add_point(event)
			self.draw_mode = False
			self.points = oversampled(self.points)
			self._compute_curve()
	
	def _add_point(self, mouse_event):
		self.points.append((
			mouse_event.x() - self.image_label.x(),
			mouse_event.y() - self.image_label.y()
		))
		self._redraw()
	
	def _redraw(self):
		# create image
		im = cairo.ImageSurface(
			cairo.FORMAT_ARGB32,
			self.image_label.width(),
			self.image_label.height()
		)
		ctx = cairo.Context(im)
		
		ctx.rectangle(0, 0, im.get_width(), im.get_height())
		ctx.set_source_rgb(1.0, 1.0, 1.0)
		ctx.fill()
		
		ctx.set_line_width(5)
		
		# draw curve path
		if self.new_points:
			ctx.set_source_rgb(0.95, 0.95, 0.95)
			draw_path(ctx, self.points)
			
			ctx.set_source_rgb(0.0, 0.0, 0.0)
			draw_path(ctx, self.new_points)
		else:
			ctx.set_source_rgb(0.0, 0.0, 0.0)
			draw_path(ctx, self.points)
		
		# transform cairo image to qt image
		image = QImage(
			str(im.get_data()),
			im.get_width(),
			im.get_height(),
			QImage.Format_ARGB32
		)
		self.pixmap = QPixmap.fromImage(image)
		self.image_label.setPixmap(self.pixmap)
		
		self.update()
		
	def _compute_curve(self):
		if self.points:
			#~self.points = compute_v1(self.points, 0.001, 9, 0.1)
			#~self.points = compute_v2(self.points, 0.001, 0.1)
			self.new_points = compute_v3(self.points, self.slider.value())
		
		self._redraw()
	
def main():
	app = QApplication(sys.argv)

	win = Window()
	win.show()
	
	app.exec_()

if __name__ == '__main__':
	main()




