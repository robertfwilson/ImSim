from astropy.io import fits
from scipy.signal import fftconvolve, oaconvolve


from .utils import *
from .wfi import WFI_Properties
#from .romsim import ROMSIM_PACKAGE_DIR


def bin_psf(psfdata, oversample=5):

    binned_psf = np.zeros(shape=(psfdata.shape[0]//oversample,psfdata.shape[1]//oversample) )

    for row in range(oversample):
        for col in range(oversample):

            binned_psf+=psfdata[row::oversample, col::oversample]

    return binned_psf


def make_prf_model(psfdata, oversample=5,):

    prfmodel = np.empty(shape=(oversample,oversample,psfdata.shape[0]//oversample-1, psfdata.shape[1]//oversample-1))
    
    for i in range(0,oversample):
        for j in range(0,oversample):

            shifted_psf = np.roll(psfdata, shift=(i,j), axis=(0,1) )[oversample:,oversample:]            
            prfmodel[i,j] = bin_psf(shifted_psf, oversample)
    
    return prfmodel



def make_prf_scene(prf_model, x, y, fluxes, size=(15,15), trim_prf_model=True,edge_buffer=5):

    os = prf_model.shape[0]
    os_frac = 1./(os)
    
    prf_size = prf_model.shape[2:]
    img_prf_weights = np.zeros(shape=(int(os), int(os), size[0]+edge_buffer*2, size[1]+edge_buffer*2) )

    

    
    for i in range(len(fluxes)):

        # How much into the pixel the centroid is
        d_col, d_row = x[i]%1, y[i]%1

        # Get the indices of the 4 PRF models surrounding the centroid
        prf_rows = int(d_row//os_frac), int((d_row//os_frac+1)%(os))
        prf_cols = int(d_col//os_frac), int((d_col//os_frac+1)%(os))

        #print('prf_row', d_row, prf_rows, 'prf_col', d_col, prf_cols)

        # Distance of centroid from PRF definitions
        row_dist = (d_row%os_frac)
        col_dist = (d_col%os_frac)

        # weights for upper left PRF        
        prf_up_le_w = row_dist - col_dist+os_frac
        # Weights for upper right PRF
        prf_up_ri_w = os_frac-row_dist - col_dist+os_frac
        # Weights for lower left PRF
        prf_lo_le_w = row_dist + col_dist
        #weights for lower right PRF
        prf_lo_ri_w = os_frac-row_dist + col_dist

        #print(np.sum([prf_up_le_w, prf_up_ri_w, prf_lo_le_w, prf_lo_ri_w]))

        i_row_le = int(y[i] )+edge_buffer
        i_row_ri =  i_row_le + int(prf_rows[1]==0) #int(i_row_le + (d_row+os_frac)//1)

        #print('i_row', i_row_le, i_row_ri)
        
        i_col_lo = int(x[i])+edge_buffer
        i_col_up = i_col_lo + int(prf_cols[1]==0) #int(i_col_lo + (d_col+os_frac)//1) #i_col_lo + int(d_col>(1.-os_frac))

        #print(d_col, prf_cols, i_col_lo, i_col_up)

        
        #print('i_col', i_col_lo, i_col_up)


        # Add weighted flux of star to the 4 PRFs that define its position
        img_prf_weights[prf_cols[0], prf_rows[0], i_col_lo, i_row_le] += fluxes[i]*prf_lo_le_w
        img_prf_weights[prf_cols[1], prf_rows[0], i_col_up, i_row_le] += fluxes[i]*prf_up_le_w
        img_prf_weights[prf_cols[0], prf_rows[1], i_col_lo, i_row_ri] += fluxes[i]*prf_lo_ri_w
        img_prf_weights[prf_cols[1], prf_rows[1], i_col_up, i_row_ri] += fluxes[i]*prf_up_ri_w


    scene_array = np.zeros(img_prf_weights.shape[2:])

    #print(prf_model.shape)

    if trim_prf_model:
        xx=int(size[0]+edge_buffer*2)
        yy= int(size[1]+edge_buffer*2)

        if max(xx,yy)>max(prf_size):
            prf_model_trimmed = prf_model
        else:

            pix2trim_xx = (prf_size[0]-xx)//2
            pix2trim_yy = (prf_size[0]-yy)//2
            
            prf_model_trimmed = prf_model[:,:,pix2trim_xx:-pix2trim_xx,pix2trim_yy:-pix2trim_yy]
    else:
        prf_model_trimmed = prf_model


    for j in range(os):
        for k in range(os):

            #print(img_prf_weights[j,k].shape, prf_model[j,k].shape, img_array.shape)
            scene_array += fftconvolve(img_prf_weights[j,k], prf_model_trimmed[j,k], mode='same')

    return scene_array[edge_buffer:-edge_buffer,edge_buffer:-edge_buffer]



class RomanPRF(object):

    def __init__(self, bandpass, sca, spectype='M0V',prf_filename=None, prf_model=None):

        self.bandpass=bandpass
        self.sca=sca
        self.spectype=spectype
        self.wfi_props = WFI_Properties(bandpass=bandpass, sca=sca)
        self.ipc_array= self.wfi_props.interpixel_capacitance

        if prf_model is None:
            self.prf_model = self._get_prf_model(prf_filename)

        

    def _get_prf_model(self, fname=None, add_ipc=True):

        if fname is None:
            fname = RIMTIMSIM_DIRECTORY + '/data/psfmodels/'+self.bandpass+'/rimtimsim_wfi_psfmodel_'+self.bandpass+'_'+self.sca+'_spectype_'+self.spectype+'_jitter_12mas_nlambda_10.fits'

        hdul = fits.open(fname)
        self.psf_fits = hdul
        self.psf_data = self.psf_fits[0].data
        self.prf_data = self.psf_fits[1].data
        hdul.close()

        
        self.oversample = self.psf_data.shape[0]//self.prf_data.shape[0]
        
        prf_model = make_prf_model(self.psf_data, oversample=self.oversample)

        if add_ipc:
            prf_model_w_ipc = np.ones_like(prf_model)
            
            for i in range(prf_model.shape[0]):
                for j in range(prf_model.shape[1]):
                    prf_model_w_ipc[i,j] = oaconvolve(prf_model[i,j], self.ipc_array, mode='same')

            self.prf_model = prf_model_w_ipc
            return prf_model_w_ipc
                            
        self.prf_model = prf_model
        return prf_model        

        
    def _build_PRF_scene_FFT(self, coords, fluxes, size, trim=True, edge_buffer=5):

        prf_mod = self.prf_model
        x,y = coords.T
        
        scene = make_prf_scene(prf_mod, x, y, fluxes, size=size, trim_prf_model=trim, edge_buffer=edge_buffer )
        
        return scene

    def build_PRF_scene(self, coords, mags, size=(25,25), edge_buffer=5, trim=True):

        zp_mag = self.wfi_props.zeropoint_mag
        fluxes = 10.**(-0.4 * (mags-zp_mag) )

        return self._build_PRF_scene_FFT(coords, fluxes, size=size , edge_buffer=edge_buffer, trim=trim)

    def _interpolate_PRF(self, dx, dy):

        return 1.




class RomanVariablePRF(object):

    def __init__(self):

        
        return 1.

    




class RomanSceneModeler(RomanPRF):


    def __init__(self, coords, mags, bandpass, sca, spectype='M0V', prf_filename=None, prf_model=None):

        self.bandpass=bandpass
        self.sca=sca
        self.spectype=spectype
        self.wfi_props = WFI_Properties(bandpass=bandpass, sca=sca)
        self.ipc_array= self.wfi_props.interpixel_capacitance
        #np.array([[0.21,  1.62,  0.20],
        #        [1.88, 91.59,  1.87],
        #        [0.21,  1.66,  0.22]]) / 100.0

        if prf_model is None:
            self.prf_model = self._get_prf_model(prf_filename)

        
        self.coords = coords
        self.mags = mags




    def _build_scene_model(self, ):

        


        self.scene_model = 0

        
    def _interpolate_scene(self, d_row, d_col):


        
        
        return 1. 

    

