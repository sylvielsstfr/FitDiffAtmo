"""
Module instrument to simulate the auxtel instrument

"""
#pylint: disable=trailing-whitespace

import jax.numpy as jnp

DCCD   = 181.45 # mm
PIXELW = 0.01   # mm (10 microns)
NEFF=156. # lines per mm
MM_TO_NM = 1000000.



class Hologram():
    """A class containing the properties of the optical device called Hologram

    :param wlmin: minimum wavelength measurable by the device in nm, defaults to 380.
    :type wlmin: float, optional

    :param wlmax: maximum wavelength measurable by the device in nm, defaults to 1000. 
    :type wlmax: float, optional 

    :param rebin: rebinning parameter chosen for libradtran, defaults 1, no rebining
    :type rebin: int, optional
    """
    def __init__(self,wlmin=380.,wlmax=1000.,rebin=1):
        """
        init the hologram object from its wavelength range of operation and its pixel resolution
        """
        self.wlmin = wlmin
        self.wlmax = wlmax
        self.rebin = rebin

    @staticmethod
    def dispersion ( wl:float,neff:float=NEFF,dccd:float=DCCD,p:int=1)->float:
        """dispersion(wl,neff,dccd,p) : number of dx per wavelength

        :param wl: wavelength in mm
        :type wl:  float or array of floats
        :param neff: line density of the disperser at the beam impact center per mm, defaults to NEFF
        :type neff: float, optional
        :param dccd: distance between the device to the CCD, default DCCD
        :type dccd: float, optional
        :param p: order of dispersion, defaults 1
        :type p: int, optional
        :return: the dispersion shift x 
        :rtype: float or array of float of the same size as wl

        recommended : all input arguments should be expressed in the same unit,
        either mm, or microns, thus the output will be in the same unit.
        """
        x = dccd*neff*wl*p/jnp.sqrt(1-(wl*neff*p)**2)
        return x
          
    @staticmethod
    def disp_to_wavelength( x:float, neff:float=NEFF,dccd:float=DCCD,p:int=1)->float:
        """Get a list of wavelength from the x spacing  number of dx per wavelength

        :param x: position spacing along dispersion axis
        :type x:  float or array of floats
        :param neff: line density of the disperser at the beam impact center per mm, defaults to NEFF
        :type neff: float, optional
        :param dccd: distance between the device to the CCD, default DCCD
        :type dccd: float, optional
        :param p: order of dispersion, defaults 1
        :type p: int, optional
        :return: the wavelength
        :rtype: float or array of float of the same size as wl

        recommended : all input arguments should be expressed in the same unit,
        either mm, or microns, thus the output will be in the same unit.
        """
        xx = x/dccd
        wl = 1/(neff*p)*xx/jnp.sqrt(1+xx**2)
        return wl
         
  
    @staticmethod
    def dispersion_rate( wl:float, neff=NEFF, dccd=DCCD,p:int=1)->float:
        """dispersion_rate(wl) : number of dx per wavelength

        :param wl: wavelength in mm
        :type wl:  float or array of floats
        :param neff: line density of the disperser at the beam impact center per mm, defaults to NEFF
        :type neff: float, optional
        :param dccd: distance between the device to the CCD, default DCCD
        :type dccd: float, optional
        :param p: order of dispersion, defaults 1
        :type p: int, optional
        :return: the dispersion shift rate dx/dlambda 
        :rtype: float or array of float of the same size as wl

        recommended : all input arguments should be expressed in the same unit,
        either mm, or microns, thus the output will be in the same unit.
        """


        #dxdlambda=D*neff*p*(np.sqrt(1-(wl*neff*p)**2)+ 
        #(wl*neff*p)**2/np.sqrt(1-(wl*neff*p)**2))/(1-(wl*neff*p)**2)
   
        dxdlambda=dccd*neff*p/(jnp.sqrt(1-(wl*neff*p)**2))**3
        return dxdlambda 

    def get_wavelength_sample(self):
        """ Must provide a list of wavelength sampled at the pixel spacing 
            These wavelength are considered as sampled wavelengths by the instrument
        """

        npixels = jnp.arange(0.,5000)
        pix_dist = npixels* PIXELW * self.rebin # distances in mm from 0th order point

        wl_samples = Hologram.disp_to_wavelength(pix_dist) # wavelengths in mm
        wl_samples *= MM_TO_NM
        indexes = jnp.where(jnp.logical_and(wl_samples>self.wlmin,wl_samples<self.wlmax))[0]
        wl_selected = wl_samples[indexes]
        return wl_selected
        