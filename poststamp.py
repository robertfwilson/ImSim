import numpy as np
from tqdm import tqdm
import pandas as pd

from .injection import update_stellar_catalog
from .utils import *


from .romsim import RomanImage



class TimeSeriesCutout(RomanImage):

    def __init__(self, star, frame_size, bkg_stars=None, **kw):

        self.frame_size=frame_size
        self.star = star
        self.bkg_stars=bkg_stars
        self.subframe=frame_size

        super().__init__(subframe=frame_size, bandpass=star.bandpass, **kw)

    

    def generate_random_bkg_stars(self, detector_catalog=None, detector_size=(4096,4096)):

        if detector_catalog is None:
            detector_catalog = pd.read_csv(RIMTIMSIM_DIRECTORY+'/data/starcats/bulge_catalog_mags.dat', index_col=0)
    
        n_stars_exp = len(detector_catalog) * (self.frame_size[0]*self.frame_size[1]) / (detector_size[0] * detector_size[1])
    
        n_stars = np.random.poisson(n_stars_exp)
    
        bkg_stars = detector_catalog.iloc[np.random.choice(np.arange(len(detector_catalog)), size=n_stars)].copy()
    
        #random_position = *frame_size[1]
        bkg_stars['xcol']=np.random.rand(n_stars) * self.frame_size[0]
        bkg_stars['ycol']=np.random.rand(n_stars) * self.frame_size[1]
    
        bkg_stars=bkg_stars.reset_index()
        self.bkg_stars=bkg_stars
        
        return bkg_stars

    def _make_source_catalog(self, star_xy=None, bandpass='F146'):

        if star_xy is None:
            try:
                star_xy = self.star_xy
            except AttributeError:
                star_xy = self.frame_size[0]/2., self.frame_size[1]/2.
                self.star_xy=star_xy
        else:
            self.star_xy = star_xy
            

        
        target_star = pd.DataFrame({'sicbroid':[-1], 'xcol':[star_xy[0]], 'ycol':[star_xy[1]],self.bandpass:[self.star.mag]}, index=[-1])

        try:
            bkg_catalog=self.bkg_stars.reset_index()
        except:
            self.generate_random_bkg_stars()
            bkg_catalog=self.bkg_stars.reset_index()
            
        all_stars = pd.concat([target_star, bkg_catalog[['sicbroid','xcol','ycol',self.bandpass]]])

        self.SourceCatalog = all_stars

        return all_stars 


    def get_simulated_cutout(self, star, frame_size=(32,32), dither_random=1.,
                             bandpass='F146',multi_accum=[1,2,3,4,4,4], n_zodi=5.):

        
        target_mag_timeseries = pd.DataFrame([star._dflux_to_dmag()], index=[-1])

        #print(star._dflux_to_dmag(), sum(star._dflux_to_dmag()))


        if len(self.SourceCatalog.loc[-1])==0:
            target_star = pd.DataFrame({'sicbroid':[-1], 'xcol':[star_xy[0]], 'ycol':[star_xy[1]],self.bandpass:[star.mag]}, index=[-1])
        else:
            target_star = self.SourceCatalog.loc[-1]

        
        bkg_catalog=self.bkg_stars.reset_index()
        all_stars = pd.concat([target_star, bkg_catalog[['sicbroid','xcol','ycol',self.bandpass]]])

        #self.SourceCatalog = all_stars

        all_stars = self.SourceCatalog.copy()
        imgs = []
        errs=[]
        
        i=min(target_mag_timeseries.columns)

        if dither_random != 0.:
            dx_t = np.random.randn(len(star.time)) * dither_random
            dy_t = np.random.randn(len(star.time)) * dither_random
        
        for j,t in tqdm(enumerate(star.time)):

            delta_mags = target_mag_timeseries[i]


            new_star_cat = update_stellar_catalog(all_stars, delta_mags, mag_col=bandpass,
                                                  delta_xcol=dx_t[j], delta_ycol=dy_t[j])


            data, data_err = self.make_realistic_image(oversample=True, bandpass=bandpass, \
                                                    read_style='ramp', return_err=True, \
                                                    multiaccum_table=multi_accum, \
                                                    star_list=new_star_cat[:,1:],
                                                    trim_psf_kernel=True,  )

            i+=1
            imgs.append(data)
            errs.append(data_err)

        self.cutouts = imgs
        self.cutout_errs=errs
            
        return imgs


    def get_PSF_lightcurve(self, stars=[-1], assume_constant_bkg=True,
                           assume_constant_bkg_stars=True, dithered=False):

        target_stars = self.SourceCatalog.loc[stars]        
        
        if assume_constant_bkg:
            # Calculate the Zodiacal and thermal background in the simulated images
            sky_bkg = self.n_min_zodiacal_background * self.wfiprops.minzodi_background
            sky_bkg += self.wfiprops.thermal_background
            
        if assume_constant_bkg_stars:
            bkg_stars = self.SourceCatalog.drop(stars)

        target_scene = self._get_target_scene(target_stars)
        bkg_scene = self._get_background_scene(bkg_stars )

        self.target_scene = target_scene
        self.bkg_scene = bkg_scene
        
        data = np.array(self.cutouts)
        data_err = np.sqrt(np.array(self.cutout_errs))

        #Replace Saturated Pixels
        sat_mask = ~np.isfinite(data)
        data[sat_mask]=1e5
        data_err[sat_mask] = np.inf


        #if np.shape(data_err)!=np.shape(data):
        #data_err = np.ones_like(data)
            
        
        #if dithered:
        # Currently only works for non-dithered data. Whch is cool for right now.     
        n_frames = np.array(data).shape[0]

        # Create Design Matrix
        #A =  np.vstack([target_scene.ravel()]).T

        # Solve for the Flux in each frame
        flux_weights = np.array([ matrix_solve(x=target_scene, y=data[i]-bkg_scene, y_err=data_err[i]) for i in range(len(data)) ])
        
        flux, flux_err = flux_weights.reshape(-1, 2).T

        #print(flux, flux_err)

        lightcurve = {'time':self.star.time, 'psf_flux':flux, 'psf_flux_err':flux_err, 'injected_flux': self.star.d_flux+1.}
        self.lightcurve = lightcurve
        
        return lightcurve

    def get_quick_lightcurve(self, stars=[-1], multi_accum=[1,2,3,4,4,4], n_zodi=5):


        psf_flux, psf_flux_err = self.estimate_psf_lightcurve_noise(stars, multi_accum, n_zodi)

        injected_light_curve = self.star.d_flux + 1.
        
        return injected_light_curve + np.random.randn(len(injected_light_curve))*psf_flux_err


    

    def estimate_psf_lightcurve_noise(self,  stars=[-1], multi_accum=[1,2,3,4,4,4], n_zodi=5):


        try:
            star_list =self.SourceCatalog.to_numpy()
        except AttributeError:
            self._make_source_catalog()
            star_list =self.SourceCatalog.to_numpy()

        data, data_err = self.make_realistic_image(oversample=True, bandpass=self.star.bandpass, \
                                                    read_style='ramp', return_err=True, \
                                                    multiaccum_table=multi_accum, \
                                                    star_list=star_list,trim_psf_kernel=True,  )


        target_stars = self.SourceCatalog.loc[stars]
        bkg_catalog = self.bkg_stars[['sicbroid','xcol','ycol',self.bandpass]]
        
        target_scene = self._get_target_scene(target_stars)
        bkg_scene = self._get_background_scene(bkg_catalog)

        self.target_scene = target_scene
        self.bkg_scene = bkg_scene


         #Replace Saturated Pixels
        sat_mask = ~np.isfinite(data)
        data[sat_mask]=1e5
        data_err[sat_mask] = np.inf

        self.cutouts = data
        self.cutout_errs=data_err
        
        
        psf_flux, psf_flux_err = matrix_solve(x=target_scene, y=data-bkg_scene, y_err=data_err) 

        return psf_flux, psf_flux_err


    
    def _get_background_scene(self, bkg_stars):
        
        bkg_scene = self._make_expected_source_scene(star_list=bkg_stars.to_numpy(),)
        self.bkg_scene = bkg_scene
        
        return bkg_scene
    
    def _get_target_scene(self, target_stars):
        
        target_scene = self._make_expected_source_scene(star_list=target_stars.to_numpy(), include_sky=False, )
        
        self.target_scene = target_scene
        
        return target_scene

        
    
    def set_base_target_catalog(self, catalog):

        self.targ_stars = catalog



    def get_timeseries(self, ):

        return 1. 


    def calc_img_cutout(self, ):

        return 1. 


   
