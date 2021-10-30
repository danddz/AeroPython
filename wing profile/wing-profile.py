import math
import numpy
from matplotlib import pyplot

%matplotlib inline

def get_velocity_doublet(strength, xd, yd, X, Y):   
    u = - strength/(2*math.pi)*((X-xd)**2-(Y-yd)**2)/((X-xd)**2+(Y-yd)**2)**2
    v = - strength/(2*math.pi)*2*(X-xd)*(Y-yd)/((X-xd)**2+(Y-yd)**2)**2
    return u, v

def get_stream_function_doublet(strength, xd, yd, X, Y):
    psi = - strength/(2*math.pi)*(Y-yd)/((X-xd)**2+(Y-yd)**2)
    return psi

def get_velocity(strength, xs, ys, X, Y):   
    u = strength/(2*numpy.pi)*(X-xs)/((X-xs)**2+(Y-ys)**2)
    v = strength/(2*numpy.pi)*(Y-ys)/((X-xs)**2+(Y-ys)**2)    
    return u, v

def get_stream_function(strength, xs, ys, X, Y):
    psi = strength/(2*numpy.pi)*numpy.arctan2((Y-ys), (X-xs))    
    return psi
  
  
x = numpy.loadtxt("https://raw.githubusercontent.com/ikursakov/ru_AeroPython/master/lessons/resources/NACA0012_x.txt")
y = numpy.loadtxt("https://raw.githubusercontent.com/ikursakov/ru_AeroPython/master/lessons/resources/NACA0012_y.txt")
sigma = numpy.loadtxt("https://raw.githubusercontent.com/ikursakov/ru_AeroPython/master/lessons/resources/NACA0012_sigma.txt")

size = 15
N = 510 * 5
x_start, x_end = -1, 2
y_start, y_end = -0.5, 0.5

X, Y = numpy.meshgrid(numpy.linspace(x_start, x_end, N), numpy.linspace(y_start, y_end, N))
pyplot.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)

u_inf = 1.0
u = u_inf * numpy.ones((N, N), dtype = float)
v = numpy.zeros((N, N), dtype = float)
psi = u_inf * Y

for i in range(len(sigma)):
    u_sourse, v_sourse = get_velocity(sigma[i], x[i], y[i], X, Y)
    psi_sourse = get_stream_function(sigma[i], x[i], y[i], X, Y)
    u += u_sourse
    v += v_sourse
    psi += psi_sourse
    
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1,arrowstyle='->')
pyplot.contour(X, Y, psi, levels=[-0.01, 0.01], colors='#CD2305', linewidths=2,linestyles='solid')
pyplot.scatter(x, y, color = "blue")
#pyplot.plot(x,y)

cp = 1.0 - (u**2+v**2)/u_inf**2

size = 10
pyplot.figure(figsize=(1.1*size, (y_end-y_start)/(x_end-x_start)*size))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
contf = pyplot.contourf(X, Y, cp, levels=numpy.linspace(-0.7, 0.5, 100),extend='both')
cbar = pyplot.colorbar(contf)
cbar.set_label('$C_p$', fontsize=16)
cbar.set_ticks([-0.7, 0.0, 0.5])
pyplot.scatter(X[numpy.unravel_index(numpy.argmax(cp), cp.shape)], Y[numpy.unravel_index(numpy.argmax(cp), cp.shape)], color='g', s = 20, marker='o')


print(numpy.max(cp))
print(numpy.unravel_index(numpy.argmax(cp), cp.shape))
