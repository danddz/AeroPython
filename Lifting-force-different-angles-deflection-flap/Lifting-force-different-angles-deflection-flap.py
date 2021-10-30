import os
import io
import pandas as pd
import math
import numpy
from scipy import integrate
from matplotlib import pyplot
import itertools
import requests

%matplotlib inline

class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya 
        self.xb, self.yb = xb, yb 
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2         
        self.length = numpy.sqrt((xb - xa)**2 + (yb - ya)**2)
        
        if xb-xa <= 0.0:
            self.beta = numpy.arccos((yb - ya) / self.length)
        elif xb-xa > 0.0:
            self.beta = numpy.pi + numpy.arccos(-(yb - ya) / self.length)
        
        if self.beta <= numpy.pi:
            self.loc = 'upper' 
        else:
            self.loc = 'lower' 
        
        self.sigma = 0.0
        self.vt = 0.0   
        self.cp = 0.0    
        self.gamma = 0.0 
        
 def define_panels(x, y, N):
    panels = numpy.empty(N, dtype = object)
    for i in range(N):
        panels[i] = Panel(x[i], y[i], x[i + 1], y[i + 1])
    return panels
    
class Freestream:
    def __init__(self, u_inf, alpha):
        self.u_inf = u_inf
        self.alpha = alpha * numpy.pi / 180.0 
        
        
def integral(x, y, panel, dxdk, dydk):
    def integrand(s):
        return (((x - (panel.xa - numpy.sin(panel.beta) * s)) * dxdk + (y - (panel.ya + numpy.cos(panel.beta) * s)) * dydk) / ((x - (panel.xa - numpy.sin(panel.beta) * s))**2 + (y - (panel.ya + numpy.cos(panel.beta) * s))**2))
    return integrate.quad(integrand, 0.0, panel.length)[0]
    
def source_contribution_normal(panels):
    A = numpy.empty((panels.size, panels.size), dtype = float)
    numpy.fill_diagonal(A, 0.5)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, panel_j, numpy.cos(panel_i.beta), numpy.sin(panel_i.beta))
    return A
    
def vortex_contribution_normal(panels):
    A = numpy.empty((panels.size, panels.size), dtype = float)
    numpy.fill_diagonal(A, 0.0)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, panel_j, numpy.sin(panel_i.beta), -numpy.cos(panel_i.beta))
    return A
    
def get_velocity_field(panels, freestream, X, Y, gamma1, gamma2):
    u = freestream.u_inf * numpy.cos(freestream.alpha) * numpy.ones_like(X, dtype = float)
    v = freestream.u_inf * numpy.sin(freestream.alpha) * numpy.ones_like(X, dtype = float)
    vec_intregral = numpy.vectorize(integral)
    
    for panel in panels:
        u += panel.sigma / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 1, 0) - gamma1 / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 0, -1)
        v += panel.sigma / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 0, 1) - gamma1 / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 1, 0)
        u += panel.sigma / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 1, 0) - gamma2 / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 0, -1)
        v += panel.sigma / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 0, 1) - gamma2 / (2.0 * numpy.pi) * vec_intregral(X, Y, panel, 1, 0)
    
    return u, v
    
 def compute_tangential_velocity(panels, freestream, gamma1, gamma2, A_source, B_vortex, N):
    A = numpy.empty((panels.size, panels.size + 2), dtype = float)
    A[:, :-2] = B_vortex
    A[:, -2] = -numpy.sum(A_source[:, :N], axis = 1)
    A[:, -1] = -numpy.sum(A_source[:, N:], axis = 1)

    b = freestream.u_inf*numpy.sin([freestream.alpha-panel.beta for panel in panels])
    strengths = numpy.append([panel.sigma for panel in panels], [gamma1, gamma2])
    tangential_velocities = numpy.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]
        
        
def compute_pressure_coefficient(panels, freestream):
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2
        
def kutta_condition(A_source, B_vortex, N):
    b2 = numpy.empty(A_source.shape[0] + 2, dtype = float)
    b1 = numpy.empty(A_source.shape[0] + 2, dtype = float)
    
    b1[:-2] = B_vortex[0, :] + B_vortex[N-1, :]
    b1[-2] = - numpy.sum(A_source[0, :N] + A_source[N-1, :N])
    b1[-1] = - numpy.sum(A_source[0, N:] + A_source[N-1, N:])
    
    b2[:-2] = B_vortex[N, :] + B_vortex[-1, :]
    b2[-2] = - numpy.sum(A_source[N, :N] + A_source[-1, :N])
    b2[-1] = - numpy.sum(A_source[N, N:] + A_source[-1, N:])
    
    return b1, b2
    
def build_singularity_matrix(A_source, B_vortex, N):
    A = numpy.empty((A_source.shape[0] + 2, A_source.shape[1] + 2), dtype = float)
    A[:-2, :-2] = A_source
    A[:-2, -2] = numpy.sum(B_vortex[:,0:N], axis=1)
    A[:-2, -1] = numpy.sum(B_vortex[:,N:], axis=1)
    A[-2, :], A[-1, :] = kutta_condition(A_source, B_vortex, N)
    return A
    
def build_freestream_rhs(panels_first, panels_second, freestream):
    b = numpy.empty(panels_first.size + panels_second.size + 2, dtype = float)
    for i, panel in enumerate(itertools.chain(panels_first, panels_second)):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    b[-2] = -freestream.u_inf * (numpy.sin(freestream.alpha - panels_first[0].beta) + numpy.sin(freestream.alpha - panels_first[-1].beta))
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panels_second[0].beta) + numpy.sin(freestream.alpha - panels_second[-1].beta))
    return b
    
naca_main_url = "https://raw.githubusercontent.com/ikursakov/ru_AeroPython/master/lessons/resources/NACA23012_MainFoil.csv"
naca_flap_url = "https://raw.githubusercontent.com/ikursakov/ru_AeroPython/master/lessons/resources/NACA23012_FlapFoil.csv"

N = 150 
x_rotation = 1.03
y_rotation = -0.054
angle_rotation = 20.0 * math.pi / 180 

main_s = requests.get(naca_main_url).content
main_coord = pd.read_csv(io.StringIO(main_s.decode('utf-8')), names = ('x', 'y'))
x_main = main_coord.x.values
y_main = main_coord.y.values

flap_s = requests.get(naca_flap_url).content
flap_coord = pd.read_csv(io.StringIO(flap_s.decode('utf-8')), names = ('x', 'y'))
x_const_flap = flap_coord.x.values
y_const_flap = flap_coord.y.values
x_flap = x_rotation + (x_const_flap - x_rotation) * math.cos(angle_rotation) + (y_const_flap - y_rotation) * math.sin(angle_rotation)
y_flap = y_rotation - (x_const_flap - x_rotation) * math.sin(angle_rotation) + (y_const_flap - y_rotation) * math.cos(angle_rotation)

panels_main = define_panels(x = x_main, y = y_main, N = N)
panels_flap = define_panels(x = x_flap, y = y_flap, N = N)
angle_freestreem_a, angle_freestreem_b = -14.0, 14.0
angle_freestreem_best, cl_best = float('inf'), float('inf')

while abs(angle_freestreem_a - angle_freestreem_b) > 0.01:
    freestream = Freestream(u_inf = 1.0, alpha = (angle_freestreem_a + angle_freestreem_b) / 2)
    panels = numpy.concatenate((panels_main, panels_flap))
    A_source = source_contribution_normal(panels)
    B_vortex = vortex_contribution_normal(panels)
    
    A = build_singularity_matrix(A_source, B_vortex, numpy.size(panels_main))
    b = build_freestream_rhs(panels_main, panels_flap, freestream)
    strengths = numpy.linalg.solve(A, b)
    
    for i , panel in enumerate(panels_main):
        panel.sigma = strengths[i]
        panel.gamma = strengths[-2]
    for i , panel in enumerate(panels_flap):
        panel.sigma = strengths[i + panels_flap.size]
        panel.gamma = strengths[-1]
    
    compute_tangential_velocity(panels, freestream, panels_main[0].gamma, panels_flap[0].gamma, A_source, B_vortex, numpy.size(panels_main))
    compute_pressure_coefficient(panels_main, freestream)
    compute_pressure_coefficient(panels_flap, freestream)
    
    c_main = abs(max(panel.xa for panel in panels_main) - min(panel.xa for panel in panels_main))
    c_flap = abs(max(panel.xa for panel in panels_flap) - min(panel.xa for panel in panels_flap))

    cl_new = (panels_main[0].gamma * sum(panel.length for panel in panels_main) + panels_flap[0].gamma * sum(panel.length for panel in panels_flap) ) / (0.5 * freestream.u_inf * (c_main + c_flap))
    
    if abs(cl_new) < abs(cl_best):
        cl_best = cl_new
        angle_freestreem_best = (angle_freestreem_a + angle_freestreem_b) / 2
    if cl_new > 0:
        angle_freestreem_b = (angle_freestreem_a + angle_freestreem_b) / 2
    else:
        angle_freestreem_a = (angle_freestreem_a + angle_freestreem_b) / 2
        
u_inf, alpha = 1.0, 0.0
angle_rotation_a = -20.0 * math.pi / 180
angle_rotation_b = 20.0 * math.pi / 180
angle_flap_best, cl_flap_best = float('inf'), float('inf')
freestream = Freestream(u_inf, alpha)

while abs((angle_rotation_a) - (angle_rotation_b)) > 0.001:
    angle_rotation = (angle_rotation_a + angle_rotation_b) / 2
    x_flap = x_rotation + (x_const_flap - x_rotation) * math.cos(angle_rotation) - (y_const_flap - y_rotation) * math.sin(angle_rotation)
    y_flap = y_rotation + (x_const_flap - x_rotation) * math.sin(angle_rotation) + (y_const_flap - y_rotation) * math.cos(angle_rotation)
    
    panels_main = define_panels(x = x_main, y = y_main, N = N)
    panels_flap = define_panels(x = x_flap, y = y_flap, N = N)
    panels = numpy.concatenate((panels_main, panels_flap))
    
    A_source = source_contribution_normal(panels)
    B_vortex = vortex_contribution_normal(panels)
    
    A = build_singularity_matrix(A_source, B_vortex, numpy.size(panels_main))
    b = build_freestream_rhs(panels_main, panels_flap, freestream)
    strengths = numpy.linalg.solve(A, b)
    
    for i , panel in enumerate(panels_main):
        panel.sigma = strengths[i]
        panel.gamma = strengths[-2]
    for i , panel in enumerate(panels_flap):
        panel.sigma = strengths[i + panels_flap.size]
        panel.gamma = strengths[-1]
    
    compute_tangential_velocity(panels, freestream, panels_main[0].gamma, panels_flap[0].gamma, A_source, B_vortex, numpy.size(panels_main))
    compute_pressure_coefficient(panels_main, freestream)
    compute_pressure_coefficient(panels_flap, freestream)
    
    c_main = abs(max(panel.xa for panel in panels_main) - min(panel.xa for panel in panels_main))
    c_flap = abs(max(panel.xa for panel in panels_flap) - min(panel.xa for panel in panels_flap))

    cl_new = ( panels_main[0].gamma * sum(panel.length for panel in panels_main) + panels_flap[0].gamma * sum(panel.length for panel in panels_flap) ) / (0.5 * freestream.u_inf * (c_main + c_flap))
    
    if abs(cl_new) < abs(cl_flap_best):
        cl_flap_best = cl_new
        angle_flap_best = angle_rotation / math.pi * 180
    
    if cl_new > 0:
        angle_rotation_a = angle_rotation
    else:
        angle_rotation_b = angle_rotation
        
u_inf, alpha = 1.0, 4.0
freestream = Freestream(u_inf, alpha)
angle_flaps = numpy.array([0, 5, 10, 15], dtype = 'float') * math.pi / 180
angle_flap_best = -angle_flap_best
cl_0 = 0.0
results = []

for i_angle, angle_flap in enumerate(angle_flaps):
    x_flap = x_rotation + (x_const_flap - x_rotation) * math.cos(angle_flap) + (y_const_flap - y_rotation) * math.sin(angle_flap)
    y_flap = y_rotation - (x_const_flap - x_rotation) * math.sin(angle_flap) + (y_const_flap - y_rotation) * math.cos(angle_flap)
    
    panels_main = define_panels(x = x_main, y = y_main, N = N)
    panels_flap = define_panels(x = x_flap, y = y_flap, N = N)
    panels = numpy.concatenate((panels_main, panels_flap))
    
    A_source = source_contribution_normal(panels)
    B_vortex = vortex_contribution_normal(panels)
    
    A = build_singularity_matrix(A_source, B_vortex, numpy.size(panels_main))
    b = build_freestream_rhs(panels_main, panels_flap, freestream)
    
    strengths = numpy.linalg.solve(A, b)
    
    for i , panel in enumerate(panels_main):
        panel.sigma = strengths[i]
        panel.gamma = strengths[-2]
    for i , panel in enumerate(panels_flap):
        panel.sigma = strengths[i + panels_flap.size]
        panel.gamma = strengths[-1]
    
    compute_tangential_velocity(panels, freestream, panels_main[0].gamma, panels_flap[0].gamma, A_source, B_vortex, numpy.size(panels_main))
    compute_pressure_coefficient(panels_main, freestream)
    compute_pressure_coefficient(panels_flap, freestream)

    c_main = abs(max(panel.xa for panel in panels_main) - min(panel.xa for panel in panels_main))
    c_flap = abs(max(panel.xa for panel in panels_flap) - min(panel.xa for panel in panels_flap))

    cl_new = (panels_main[0].gamma * sum(panel.length for panel in panels_main) + panels_flap[0].gamma * sum(panel.length for panel in panels_flap)) / (0.5 * freestream.u_inf * (c_main + c_flap))
    
    if i_angle == 0:
        results.append([cl_new, angle_flap / math.pi * 180])
        cl_0 = cl_new
    else:
        results.append([cl_new, angle_flap / math.pi * 180, ((cl_new / cl_0)**(0.5) - 1) * 100])
        
print("Подъемная сила = {:.4f}, угол атаки = {:.4f} \n".format(cl_best, angle_freestreem_best))
print("Подъемная сила = {:.4f}, угол отклонения закрылка = {:.4f} \n".format(cl_flap_best, angle_flap_best))
print("\n".join(["Подъемная сила = {:.4f}. угол отклонения закрылка = {:.4f}".format(results[0][0], results[0][1])] + ["Подъемная сила = {:.4f}. угол отклонения закрылка = {:.4f}. Скорость больше на {:.4f}%".format(result[0], result[1], result[2]) for result in results[1:]]))
