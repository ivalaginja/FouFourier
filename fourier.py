"""
Simple functions for my Fourier notebooks.
"""

import numpy as np
from scipy import signal


# From notebook 3

def ft1d(func):
    """Calculate the Fourier Transform of func shifting the zer-frequency index to the center where needed."""
    ft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(func)))
    return ft


def ift1d(func):
    """Calculate the inverse Fourier Transform of func shifting the zer-frequency index to the center where needed."""
    ift = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(func)))
    return ift


def ft1d_freq(x):
    """Calculate the (spatial) frequency array based on the spatial array x."""
    s = np.fft.fftshift(np.fft.fftfreq(x.size, d=x[-1]-x[-2]))
    return s


def rect1d(x, ampl, tint):
    """Create the rectangle function with amplitude ample and an interval tint from -tint/2 to tint/2."""
    
    if tint/2 >= np.max(x):
        raise("Your interval is larger than the provided x array.")
    
    func = ampl * np.ones_like(x)
    leftzero = np.where(x < -tint/2)
    rightzero = np.where(x > tint/2)
    func[leftzero] = 0
    func[rightzero] = 0
    
    return func


def rect2d(size):
    """Rectangluar aperture. size is a tuple (x,y)."""
    rect = (np.abs(xx) <= (size[0]/2)) * (np.abs(yy) <= (size[1]/2))
    return rect.astype('float')


def triangle(x, ampl, tint):
    """Create the triangle function with amplitude ampl on interval tint from -tint/2 to tint/2."""
    pt = np.pi / tint
    tri = ampl/2 * signal.sawtooth(pt*x + np.pi, width=0.5) + ampl/2
    
    # Discard all but the central triangle
    indl = np.where(x < -tint)    # nulling left side
    tri[indl] = 0
    indr = np.where(x > tint)     # nulling right side
    tri[indr] = 0
    
    return tri


def gaussian(x, ampl, c):
    """Calculate a simple Gaussian with amplitude ampl and FWHM c."""
    func = ampl * np.exp(-np.pi * np.square(x) / (2 * c/2))
    return func


def gaussian2d(x, y, ampl, c):
    """Calculate a simple 2D Gaussian with amplitude ampl and FWHM c."""
    #func = ampl * np.exp(-np.square(x) / (2*np.square(c)))
    func = ampl * np.exp(-np.pi * (np.square(x) + np.square(y)) / (2 * c/2))
    return func


def sinusoid1d(x, nu, phi, ampl):
    func = A * np.cos(2*np.pi * nu * x - phi)
    return func


def sinusoid2d(x, y, nu, phi, A, theta=0):
    xr = x * np.cos(theta)
    yr = y * np.sin(theta)
    
    func = A * np.cos(2 * np.pi * nu * (xr+yr) - phi)
    return func






# From Leiden HCI class


def displC(c,trim=0):
    # displC - display a Complex number c as four plots
    #          as a (Real, Imaginary) pair and as
    #          an Amplitude, Phase plot
    #          optionally cut out the central square with size trim by trim pixels
    c2 = np.copy(c)
    if (trim>0): # if the user specifies a trim value, cut out the centre of the image
        (nx,ny) = c.shape
        dx = (nx-trim) / 2 + 1
        dy = (nx-trim) / 2 + 1
        c2 = c[dx:dx+trim,dy:dy+trim]
    
    # set up the plot panels
    fig=plt.figure(figsize=(10,8))
    axre = fig.add_subplot(221)
    axim = fig.add_subplot(222)
    axamp = fig.add_subplot(223)
    axpha = fig.add_subplot(224)
    # plot out the panels
    im = axre.imshow(c2.real)
    im = axim.imshow(c2.imag)
    im = axamp.imshow(np.abs(c2))
    im = axpha.imshow(np.angle(c2))
    
    axre.set_title('Real')
    axim.set_title('Imag')
    axamp.set_title('Amplitude')
    axpha.set_title('Phase')
    plt.show()


def padcplx(c, pad=5):
    """Puts a Complex array in the centre of a zero-filled Complex array.
    pad defines the padding multiplier for the output array."""
    (nx, ny) = c.shape
    bignx = nx * pad + 1
    bigny = ny * pad + 1
    big_c = np.zeros((bignx, bigny),dtype=complex)
    
    dx = int((nx * (pad-1)) / 2 + 1)
    dy = int((ny * (pad-1)) / 2 + 1)
    
    big_c[dx:dx+nx,dy:dy+ny] = c
    return(big_c)


def circle_mask(im, xc, yc, rcirc):
    """Create a circular aperture centered on (xc, yc) with radius rcirc."""
    x, y = np.shape(im)
    newy, newx = np.mgrid[:y,:x]
    circ = (newx-xc)**2 + (newy-yc)**2 < rcirc**2
    return circ


def zoom(im,x,y,bb):
    # cut out a square box from image im centered on (x,y) with half-box size bb
    return(im[y-bb:y+bb,x-bb:x+bb])


def box(c,x,y,trim=0):
    # chop out a square box from an array
    c2 = np.copy(c)
    (nx,ny) = c.shape
    dx = x - trim
    dy = y - trim
    c2 = c[dy:dy+2*trim,dx:dx+2*trim]
    
    return(c2)


def rotate2(img, angle, c_in):
    # rotate image img by angle degrees about point c_in
    # c_in should be an np.array((y,x))
    # returns the rotated image with zeroes for unknown values
    from scipy.ndimage.interpolation import affine_transform
    a=angle*np.pi/180.0
    transform=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    offset=c_in-c_in.dot(transform)
    dst=affine_transform(img,transform.T,order=2,offset=offset,output_shape=(img.shape),cval=0.0)
    return(dst)


def r_theta(im, xc, yc):
    # returns the radius rr and the angle phi for point (xc,yc)
    ny, nx = im.shape
    yp, xp = np.mgrid[0:ny,0:nx]
    yp = yp - yc
    xp = xp - xc
    rr = np.sqrt(np.power(yp,2.) + np.power(xp,2.))
    phi = np.arctan2(yp, xp)
    return(rr, phi)


def phi_ramp(im, npx, npy):
    ny, nx = im.shape
    ly = np.linspace(-0.5, 0.5, ny) * np.pi * npy * 2
    lx = np.linspace(-0.5, 0.5, nx) * np.pi * npx * 2
    
    x, y = np.meshgrid(lx, ly)
    return(x+y)
