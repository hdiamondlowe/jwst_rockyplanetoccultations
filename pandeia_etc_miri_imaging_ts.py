import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.modeling.models import BlackBody
from scipy import stats
import astrotools.orbitparams as orb
import pickle
import json
import starry as st
import copy
import os
st.config.lazy = False
st.config.quiet = True
from jwst_backgrounds import jbt
import pandeia.engine
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.perform_calculation import perform_calculation


#filter    = 'f1500w'
#subarray  = 'sub256'
pull_default_dict = False # needs to be true if you want to pull a new default dictionary from paneia
make_new_bkg = True # should be true if working with different stars; otherwise set to False for speed

def get_bkg(targ, ref_wave, make_new_bkg=True):
    """
    Code to retrieve sky background from the jwst backgrounds database based on system coordinates
    JWST backgrounds: https://jwst-docs.stsci.edu/jwst-general-support/jwst-background-model
    
    Inputs:
    targ          -- dictionary of target; must include RA and Dec of system as strings
    ref_wave      -- reference wavelength for jbt to return 
    make_new_bkg  -- default=True; otherwise you can load up the last background called, 
                                   use if only working with one system
                                   
    Returns:
    background    -- list of two lists containing wavelength (um) and background counts (mJy/sr)
    """
    
    print('Computing background')

    sys_coords = targ['rastr']+' '+targ['decstr']

    if make_new_bkg:
        # make astropy coordinates object
        c = SkyCoord(sys_coords, unit=(u.hourangle, u.deg))

        # use jwst backgrounds to compute background at this point
        bg = jbt.background(c.ra.deg, c.dec.deg, ref_wave)

        # background is computed for many days; choose one
        ndays = bg.bkg_data['calendar'].size
        assert ndays > 0  # make sure object is visible at some point in the year; if not check coords
        middleday = bg.bkg_data['calendar'][int(ndays / 2)] # picking the middle visible date; somewhat arbitrary!
        middleday_indx = np.argwhere(bg.bkg_data['calendar'] == middleday)[0][0]

        tot_bkg = bg.bkg_data['total_bg'][middleday_indx]
        wav_bkg = bg.bkg_data['wave_array']

        # background is [wavelength, total_background] in [micron, mJy/sr]
        background = [list(np.array(wav_bkg)), list(np.array(tot_bkg))]

        ascii.write(background, "background.txt", overwrite=True)

    else: 
        background = ascii.read("background.txt")
        background = [list(background['col0']), list(background['col1'])]
    
    print('Returning background')
    return background


def make_miri_dict(filter, subarray, targ, pull_default_dict=True):
    """
    Code to make the initial miri dictionally for imaging_ts
    
    Inputs:
    filter            -- which photometric filter to use (e.g., f1500w)
    subarray          -- which subarray readout ot use (e.g., sub256)
    targ              -- 
    sys_coords        -- string of the coordinates of the system in RA Dec; e.g. "23h06m30.33s -05d02m36.46s";
                         to be passed to get_bkg function
    pull_default_dict -- default=True; can re-use a saved one but this doesn't save much time.
    """

    print('Creating MIRI dictionary')

    # grab default imaging ts dictionary (I think this only works online?)
    if pull_default_dict:
        miri_imaging_ts = build_default_calc('jwst', 'miri', 'imaging_ts')

        # Serializing json
        json_object = json.dumps(miri_imaging_ts, indent=4)

        # Writing to sample.json
        with open("miri_imaging_ts.json", "w") as outfile:
            outfile.write(json_object)

    else: 
        with open("miri_imaging_ts.json", "r") as f:
            miri_imaging_ts = json.load(f)
            
    if   filter == 'f1500w': ref_wave = 15 * u.micron
    elif filter == 'f1800w': ref_wave = 18 * u.micron
            
    # update with basic parameters
    miri_imaging_ts['configuration']['instrument']['filter'] = filter
    miri_imaging_ts['configuration']['detector']['subarray'] = subarray
       
    miri_imaging_ts['configuration']['detector']['ngroup']   = 2    
    miri_imaging_ts['configuration']['detector']['nint'] = 1 
    miri_imaging_ts['configuration']['detector']['nexp'] = 1
    miri_imaging_ts['configuration']['detector']['readout_pattern'] = 'fastr1'
    try: miri_imaging_ts['configuration'].pop('max_filter_leak')
    except(KeyError): pass

    miri_imaging_ts['scene'][0]['spectrum']['normalization'] = {}
    miri_imaging_ts['scene'][0]['spectrum']['normalization']['type']          = 'photsys'
    miri_imaging_ts['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'vegamag'
    miri_imaging_ts['scene'][0]['spectrum']['normalization']['bandpass']      = '2mass,ks'
    miri_imaging_ts['scene'][0]['spectrum']['normalization']['norm_flux']     = targ['sy_kmag']           # change this for different stars

    miri_imaging_ts['scene'][0]['spectrum']['sed']['key']          = 'm5v'
    miri_imaging_ts['scene'][0]['spectrum']['sed']['sed_type']     = 'phoenix'
    try: miri_imaging_ts['scene'][0]['spectrum']['sed'].pop('unit')
    except(KeyError): pass

    miri_imaging_ts['background'] = get_bkg(targ, ref_wave)
    miri_imaging_ts['background_level'] = 'high'

    miri_imaging_ts['strategy']['aperture_size']  = 0.7
    miri_imaging_ts['strategy']['sky_annulus']    = [2, 2.8]

    print('Returning MIRI dictionary')
    return miri_imaging_ts

def make_miri_calib_dict(miri_dict):

    print('Creating MIRI calibration dictionary')

    miri_imaging_ts_calibration = copy.deepcopy(miri_dict)

    miri_imaging_ts_calibration['scene'][0]['spectrum']['sed']['sed_type']     = 'flat'
    miri_imaging_ts_calibration['scene'][0]['spectrum']['sed']['unit']         = 'flam'
    miri_imaging_ts_calibration['scene'][0]['spectrum']['sed'].pop('key')

    miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['type']          = 'at_lambda'
    miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_wave']     = 2
    miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_waveunit'] = 'um'
    miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_flux']     = 1e-18
    miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'flam'
    miri_imaging_ts_calibration['scene'][0]['spectrum']['normalization'].pop('bandpass')
    

    print('Returning MIRI calibration dictionary')

    return miri_imaging_ts_calibration

def get_timing(miri_dict):
    
    report = perform_calculation(miri_dict)
    tframe   = report['information']['exposure_specification']['tframe']        * u.s
    nframe   = report['information']['exposure_specification']['nframe']
    nskip    = report['information']['exposure_specification']['nsample_skip']
    
    i = InstrumentFactory(miri_dict['configuration'])
    det_pars = i.read_detector_pars()
    fullwell = det_pars['fullwell']
    sat_level = 0.8 * fullwell

    report = perform_calculation(miri_dict, dict_report=False)
    report_dict = report.as_dict()
    
    
    #report = perform_calculation(miri_imaging_ts, dict_report=False)
    report_dict = report.as_dict() 

    # count rate on the detector in e-/second/pixel
    #det = report_dict['2d']['detector']
    det = report.signal.rate_plus_bg_list[0]['fp_pix']

    timeinfo = report_dict['information']['exposure_specification']
    #totaltime = timeinfo['tgroup']*timeinfo['ngroup']*timeinfo['nint']

    maxdetvalue = np.max(det)

    #maximum time before saturation per integration 
    #based on user specified saturation level

    try:
        maxexptime_per_int = sat_level/maxdetvalue
    except: 
        maxexptime_per_int = np.nan

    transit_duration = tdur.to(u.s).value
    frame_time = tframe.value
    overhead_per_int = tframe #overhead time added per integration 
    min_nint_trans = 3
    max_ngroup = 100
    #try: 
        #are we starting with a exposure time ?
    #    maxexptime_per_int = m['maxexptime_per_int']
    #except:
        #or a pre defined number of groups specified by user
    #    ngroups_per_int = m['ngroup']

    flag_default = "All good"
    flag_high = "All good"
    if 'maxexptime_per_int' in locals():
        #Frist, if maxexptime_per_int has been defined (from above), compute ngroups_per_int

        #number of frames in one integration is the maximum time beofre exposure 
        #divided by the time it takes for one frame. Note this does not include 
        #reset frames 

        nframes_per_int = np.floor(maxexptime_per_int/frame_time)

        #for exoplanets nframe =1 an nskip always = 0 so ngroups_per_int 
        #and nframes_per_int area always the same 
        ngroups_per_int = np.floor(nframes_per_int/(nframe + nskip)) 

        #put restriction on number of groups 
        #there is a hard limit to the maximum number groups. 
        #if you exceed that limit, set it to the maximum value instead.
        #also set another check for saturation

        if ngroups_per_int > max_ngroup:
            ngroups_per_int = max_ngroup
            flag_high = "Groups/int > max num of allowed groups"

        if (ngroups_per_int < mingroups) | np.isnan(ngroups_per_int):
            ngroups_per_int = mingroups  
            nframes_per_int = mingroups
            flag_default = "NGROUPS<"+str(mingroups)+"SET TO NGROUPS="+str(mingroups)

    elif 'ngroups_per_int' in locals(): 
        #if it maxexptime_per_int been defined then set nframes per int 
        nframes_per_int = ngroups_per_int*(nframe+nskip)

        #if that didn't work its because maxexptime_per_int is nan .. run calc with mingroups
    else:
        #if maxexptime_per_int is nan then just ngroups and nframe to 2 
        #for the sake of not returning error
        ngroups_per_int = mingroups
        nframes_per_int = mingroups
        flag_default = "Something went wrong. SET TO NGROUPS="+str(mingroups)


    #the integration time is related to the number of groups and the time of each 
    #group 
    exptime_per_int = ngroups_per_int*tframe

    #clock time includes the reset frame 
    clocktime_per_int = (ngroups_per_int+1.0)*tframe

    #observing efficiency (i.e. what percentage of total time is spent on soure)
    eff = (exptime_per_int)/(clocktime_per_int)

    #this says "per occultation" but this is just the in transit frames.. See below
    # transit duration / ((ngroups + reset)*frame time)
    nint_per_occultation =  transit_duration/((ngroups_per_int+1.0)*frame_time)

    #figure out how many integrations are in transit and how many are out of transit 
    nint_in = np.ceil(nint_per_occultation)
    nint_out = np.ceil(nint_in/expfact_out)

    #you would never want a single integration in transit. 
    #here we assume that for very dim things, you would want at least 
    #3 integrations in transit 
    if nint_in < min_nint_trans:
        ngroups_per_int = np.floor(ngroups_per_int/min_nint_trans)
        exptime_per_int = (ngroups_per_int)*tframe
        clocktime_per_int = ngroups_per_int*tframe
        eff = (ngroups_per_int - 1.0)/(ngroups_per_int + 1.0)
        nint_per_occultation =  tdur/((ngroups_per_int+1.0)*tframe)
        nint_in = np.ceil(nint_per_occultation)
        nint_out = np.ceil(nint_in/expfact_out)

    if nint_out < min_nint_trans:
        nint_out = min_nint_trans

    timing = {
        #"Transit Duration" : (transit_duration)/60.0/60.0,
        "Seconds per Frame" : tframe,
        "Time/Integration incl reset (sec)":clocktime_per_int,
        "APT: Num Groups per Integration" :int(ngroups_per_int), 
        #"Num Integrations Out of Transit":int(nint_out),
        "Num Integrations In Transit":int(nint_in),
        "APT: Num Integrations per Occultation":int(nint_out+nint_in),
        "Observing Efficiency (%)": eff*100.0,
        #"Transit+Baseline, no overhead (hrs)": (nint_out+nint_in)*clocktime_per_int/60.0/60.0, 
        "Number of Transits": noccultations
    }
    
    return timing



# ## Provide stellar parameters

# grab a table of transiting terrestrial exoplanets from NASA Exoplante Archive
try:
    sample = ascii.read('RockyPlanetSample_final.csv')
except: 
    sample = ascii.read('sample_final.csv')
print("number of planets", len(sample))

# Creating a standard star
def make_star(targ):
    M_s  = targ['st_mass']
    R_s  = targ['st_rad']
    prot = 1

    star = st.Primary(
        st.Map(ydeg = 1, udeg = 2, nw = 1, amp = 1.0), 
        m    = M_s,
        r    = R_s, 
        prot = prot
    )

    return star

# adapted from Mette's code
def make_planet(plnt, phase=0, t0=0, tidally_locked=True):

    planet  = st.kepler.Secondary(
        st.Map(ydeg = 5, nw = 1, amp = 5e-3),               
        m      = plnt['pl_bmasse'],                                
        r      = plnt['pl_rade'],                          
        porb   = plnt['pl_orbper'],                                                      
        prot   = plnt['pl_orbper'],                                       
        Omega  = 0,                                         
        ecc    = 0,                                                         
        w      = 90,                                  
        t0     = t0,                           
        theta0 = 180,                           
        inc    = plnt['pl_orbincl'],
        length_unit = u.Rearth,
        mass_unit   = u.Mearth

    )
    
    if tidally_locked:
        planet.map.spot(contrast = -1, radius = 60)
        
    return planet

def flux_amplitude(T_s, T_p, wavelength, R_s, R_p):
    
    ''' This function will take in the Temperature in Kelvin, 
    and the wavelength range that we are looking at,
    as well as the the radius of the star and the planet. '''
    
    bb_s = BlackBody(T_s, scale=1*u.erg/u.s/u.cm**2/u.AA/u.sr)
    bb_p = BlackBody(T_p, scale=1*u.erg/u.s/u.cm**2/u.AA/u.sr)
    
    Flux_ratio = bb_p(wavelength)/bb_s(wavelength) * (R_p/R_s)**2
        
    return Flux_ratio.decompose()

def T_day(T_s, R_s, a, albedo, atmo='bare rock'):
    # can be 'bare rock' or 'equilibrium'
    if   atmo == 'bare rock': f = 2/3
    elif atmo == 'equilibrium': f = 1/4
    
    T_day = T_s * np.sqrt((R_s/a).decompose()) * (1 - albedo)**(1/4) * f**(1/4)
    
    return T_day

def process_target(targ):
    print(targ)

    try:
        filter = targ['filter']
        subarray = targ['subarray']
        nobs = targ['nobs']

    except: 
        filter = 'f1500w'
        subarray = 'sub256'
        nobs = 4

    # star_params
    star_name = targ['hostname']
    k_mag = targ['sy_kmag']

    # planet params
    tdur = orb.Tdur(P=targ['pl_orbper']*u.day, 
                    Rp_Rs=((targ['pl_rade']*u.R_earth)/(targ['st_rad']*u.R_sun)).decompose(),
                    a_Rs = ((targ['pl_orbsmax']*u.AU)/(targ['st_rad']*u.R_sun)).decompose(),
                    i = targ['pl_orbincl']
                   ) # event duration

    # obs params
    tfrac   = 1             # how many times occuldation duration to observe
    tsettle = 45 * u.min    # should be specific to MIRI
    tcharge = 1 * u.hr      # amount of charged time becuase JWST will not start observations right away
    noccultations = 1       # can always scale later
    mingroups = 5           # suggested for MIRI Imaging TSO
    expfact_out = 1         # bare minimum of out-of-transit baseline; but will be asking for more so not a big deal



    # for MIRI, this calculation underestimates the number of groups; 
    # experiment with adding extra groups and checking warnings
    miri_imaging_ts = make_miri_dict(filter, subarray, targ)

    # below is some of Natasha's code to determine the timing, but for some reason the resulting number of groups is always less than the saturation limit
    #timing = get_timing(miri_imaging_ts)
    #print(timing)
    #ngroup = timing['APT: Num Groups per Integration']
    #miri_imaging_ts['configuration']['detector']['ngroup'] = ngroup

    report = perform_calculation(miri_imaging_ts)
    ngroup = int(report['scalar']['sat_ngroups'])  # use as many groups as possible without saturating
    if ngroup > 300: ngroup = 300        # ngroups > 300 not recommended due to cosmic rays
    miri_imaging_ts['configuration']['detector']['ngroup'] = ngroup

    report = perform_calculation(miri_imaging_ts)

    print('ETC Warnings:')
    print(report['warnings'])

    tframe  = report['information']['exposure_specification']['tframe'] * u.s
    tint    = tframe * ngroup                         # amount of time per integration
    treset  = 1*tframe                                # reset time between each integration
    cadence = tint + treset
    nint    = (tdur/(tint + treset)).decompose()      # number of in-transit integrations
    ref_wave = report['scalar']['reference_wavelength']                         * u.micron

    print('number of groups per integration', ngroup)
    print('time per single integration:', tint)
    print('cadence (integration time plus reset):', cadence)
    print('number of in-occultation integrations:', nint.decompose())
    print('observing efficiency (%):', (tint/cadence).decompose()*100)


    miri_imaging_ts_calibration = make_miri_calib_dict(miri_imaging_ts)
    report_calibration = perform_calculation(miri_imaging_ts_calibration)
    print('Calibartion Warnings:')
    print(report_calibration['warnings'])


    T_rock = T_day(targ['st_teff']*u.K, targ['st_rad']*u.R_sun, targ['pl_orbsmax']*u.AU, 0, atmo='bare rock')
    amp_rock = flux_amplitude(targ['st_teff']*u.K, T_rock, ref_wave, targ['st_rad']*u.R_sun, targ['pl_rade']*u.R_earth)

    T_atmo = T_day(targ['st_teff']*u.K, targ['st_rad']*u.R_sun, targ['pl_orbsmax']*u.AU, 0, atmo='equilibrium')
    amp_atmo = flux_amplitude(targ['st_teff']*u.K, T_atmo, ref_wave, targ['st_rad']*u.R_sun, targ['pl_rade']*u.R_earth)

    planet = make_planet(targ)
    planet.map.amp = amp_rock

    star = make_star(targ)


    system = st.System(star, planet)#, planet_c)


    snr = report['scalar']['sn']
    extracted_flux = report['scalar']['extracted_flux'] / u.s
    extracted_noise = report['scalar']['extracted_noise'] / u.s

    calibration_extracted_flux = report_calibration['scalar']['extracted_flux'] / u.s
    calibration_norm_value = report_calibration['input']['scene'][0]['spectrum']['normalization']['norm_flux']

    extracted_flux, extracted_noise, calibration_extracted_flux * calibration_norm_value

    #signal = extracted_flux * tint
    #noise  = extracted_noise * tint
    signal = extracted_flux / calibration_extracted_flux * calibration_norm_value  * u.erg/u.s/u.cm**2/u.AA
    noise  = extracted_noise / calibration_extracted_flux * calibration_norm_value  * u.erg/u.s/u.cm**2/u.AA

    noise /= np.sqrt(nobs)

    tstart = (targ['pl_orbper']*u.day)*0.5 - (tdur/2) - (tdur*tfrac/2)
    tend   = (targ['pl_orbper']*u.day)*0.5 + (tdur/2) + (tdur*tfrac/2)
    trange = tend - tstart
    total_int = int(np.ceil((trange/cadence).decompose()))

    signal_ts = np.ones(total_int)*signal
    scatter_ts = np.random.normal(0, noise.value, total_int) * u.erg/u.s/u.cm**2/u.AA
    signal_ts_scatter = signal_ts.value + scatter_ts.value

    time = np.linspace(tstart.value, tend.value, total_int) # times in... days?

    flux = np.hstack(system.flux(time))

    #plt.figure(figsize=(15, 4))
    #plt.plot(time/targ['pl_orbper'], flux)
    #plt.axvline(0.5, color='k', alpha=0.5)
    #plt.ylim(0.988, 1.001)
    #plt.grid()
    #plt.show()


    signal_ts_scatter_binned, time_bin_edges, _ = stats.binned_statistic(time, signal_ts_scatter*flux, bins=25)
    time_bin_width = np.mean(np.diff(time_bin_edges))
    time_binned = time_bin_edges[:-1] + time_bin_width/2

    fig = plt.figure(figsize=(15,6))
    gs = gridspec.GridSpec(1, 3, left=0.07, right=0.99, bottom=0.1, top=0.93)

    figure = {}
    figure['lc'] = fig.add_subplot(gs[0,0:2])
    figure['FpFs'] = fig.add_subplot(gs[0,2])


    figure['lc'].plot(time/targ['pl_orbper'], signal_ts_scatter*flux/signal, '.', color='k', alpha=0.5, label=f'Cadence={np.round(cadence, 2)}; ngroups={ngroup}')
    figure['lc'].plot(time_binned/targ['pl_orbper'], signal_ts_scatter_binned/signal, 'o', color='k', alpha=1)
    figure['lc'].plot(time/targ['pl_orbper'], signal_ts*flux/signal, '-', lw=3, color='C3', label=f'NO atmo; Tday={np.round(T_rock, 0)}')

    # compare to depth of full equilibrium atmosphere
    planet.map.amp = amp_atmo
    flux = np.hstack(system.flux(time))
    figure['lc'].plot(time/targ['pl_orbper'], signal_ts*flux/signal, '--', lw=3, color='C0', alpha=0.8, label=f'YES atmo; Tday={np.round(T_atmo, 0)}')


    figure['lc'].axvline(0.5, ls=':', color='k', alpha=0.5)
    figure['lc'].axvline(0.5-tdur.value/targ['pl_orbper']/2, ls='--', color='k', alpha=0.5)
    figure['lc'].axvline(0.5+tdur.value/targ['pl_orbper']/2, ls='--', color='k', alpha=0.5)

    figure['lc'].legend(loc='upper right')
    per = targ['pl_orbper']
    figure['lc'].set_title(targ['pl_name']+f', Kmag={k_mag}, {nobs} obs, Tdur = {np.round(tdur.to(u.min), 2)}, P={np.round(per, 3)} days', fontsize=16)

    figure['lc'].set_xlabel('Phase', fontsize=14)
    figure['lc'].set_ylabel('Normalized Flux', fontsize=14)

    figure['lc'].grid(alpha=0.4)

    wave_range = np.linspace(0.7, 25, 100) *u.micron
    Fp_Fs_rock = flux_amplitude(targ['st_teff']*u.K, T_rock, wave_range, targ['st_rad']*u.R_sun, targ['pl_rade']*u.R_earth)
    Fp_Fs_atmo = flux_amplitude(targ['st_teff']*u.K, T_atmo, wave_range, targ['st_rad']*u.R_sun, targ['pl_rade']*u.R_earth)

    yerr = 1/report['scalar']['sn'] / np.sqrt(nint) / np.sqrt(nobs)

    figure['FpFs'].plot(wave_range, Fp_Fs_rock *1e6, lw=3, color='C3', label='NO atmo (bare rock)')
    figure['FpFs'].plot(wave_range, Fp_Fs_atmo *1e6, lw=3, color='C0', ls='--', label='YES atmo (equilibrium temp)')

    figure['FpFs'].errorbar(ref_wave.value, amp_rock *1e6, yerr=yerr.value *1e6, fmt='.', color='k', alpha=0.8)

    figure['FpFs'].legend(loc='lower right')
    figure['FpFs'].set_ylabel('$F_p$/$F_s$ (ppm)', fontsize=14)
    figure['FpFs'].set_xlabel('Wavelength ($\mu$m)', fontsize=14)
    figure['FpFs'].set_title(f'T_day,rock = {np.rint(T_rock)}, {nobs} obs', fontsize=16)

    figure['FpFs'].set_xlim(0.7, 25)
    figure['FpFs'].grid(alpha=0.4)
    
    plname = targ['pl_name'].replace(' ','')  # w/o spaces
    plt.savefig(f'../sample/{plname}_{filter}_{subarray}_{nobs}obs.png')
    #plt.show()
    plt.close()

#for targ in sample: process_target(targ)
process_target(sample[-2])