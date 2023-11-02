import jax.numpy as jnp

DCCD   = 181.45 # mm
#DCCD   = 200.0 # mm
PIXELW = 0.01   # mm (10 microns)
NEFF=156. # lines per mm




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
    def Dispersion( wl:float,neff:float=NEFF,D:float=DCCD,p:int=1)->float:
        """Dispersion(wl,a,D) : number of dx per wavelength

        :param wl: wavelength in mm
        :type wl:  float or array of floats
        :param neff: line density of the disperser at the beam impact center per mm, defaults to NEFF
        :type neff: float, optional
        :param D: distance between the device to the CCD, default DCCD
        :type D: float, optional
        :param p: order of dispersion, defaults 1
        :type p: int, optional
        :return: the dispersion shift x 
        :rtype: float or array of float of the same size as wl

        recommended : all input arguments should be expressed in the same unit,
        either mm, or microns, thus the output will be in the same unit.
        """
        X=D*neff*wl*p/jnp.sqrt(1-(wl*neff*p)**2)
        return X
    
    @staticmethod
    def Dispersion_Rate(wl:float,neff=NEFF,D=DCCD,p:int=1)->float:
        """Dispersion_Rate(wl) : number of dx per wavelength

        :param wl: wavelength in mm
        :type wl:  float or array of floats
        :param neff: line density of the disperser at the beam impact center per mm, defaults to NEFF
        :type neff: float, optional
        :param D: distance between the device to the CCD, default DCCD
        :type D: float, optional
        :param p: order of dispersion, defaults 1
        :type p: int, optional
        :return: the dispersion shift rate dx/dlambda 
        :rtype: float or array of float of the same size as wl

        recommended : all input arguments should be expressed in the same unit,
        either mm, or microns, thus the output will be in the same unit.
        """


        #dxdlambda=D*neff*p*(np.sqrt(1-(wl*neff*p)**2)+ (wl*neff*p)**2/np.sqrt(1-(wl*neff*p)**2))/(1-(wl*neff*p)**2)
   
        dxdlambda=D*neff*p/(jnp.sqrt(1-(wl*neff*p)**2))**3
        return dxdlambda 

    def GetWavelengthSample(self):
        """does nothing """
        pass