import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
import matplotlib as mpl
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column
import astrotools.orbitparams as orb
import astrotools.generateExoplanetTable as genTable
import pickle
import os
import copy

# grab a table of transiting terrestrial exoplanets from NASA Exoplante Archive
try: smallPlanetSystems = ascii.read('./NASAExoArchive_TransitingExoplanetTable.dat')
except:
    thiswd = os.getcwd()
    smallPlanetSystems = genTable.generateTransitingExoTable(outputpath=thiswd, 
                                                             sy_dist_upper=40, 
                                                             st_rad_upper=0.5, 
                                                             pl_rade_upper=1.85, 
                                                             pl_per_upper=30)

print("number of planets", len(smallPlanetSystems))
smallPlanetSystems.pprint(show_unit=True)

ind = np.argwhere(smallPlanetSystems['hostname'] == 'TRAPPIST-1')
smallPlanetSystems['st_raderr1'][ind] = 0.0013  # Agol+ 2020
smallPlanetSystems['st_raderr2'][ind] = -0.0013  # Agol+ 2020
smallPlanetSystems['st_mass'][ind] = 0.0898  # Mann+ 2019

ind = np.argwhere(smallPlanetSystems['hostname'] == 'LHS 1140')
smallPlanetSystems['st_raderr1'][ind] = 0.0035  # Lillo-Box+ 2020
smallPlanetSystems['st_raderr2'][ind] = -0.0035  # Lillo-Box+ 2020

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'TOI-1468 b')
smallPlanetSystems['pl_orbsmax'][ind] = 0.02102086  # Charturvedi+ 2022

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'GJ 1132 b')
smallPlanetSystems['pl_orbincl'][ind] = 88.68   # Dittmann+ (2017)

# refining parameters from Luque & Palle (2022)
ind = np.argwhere(smallPlanetSystems['pl_name'] == 'TOI-1634 b')
smallPlanetSystems['pl_rade'][ind]      = 1.773
smallPlanetSystems['pl_radeerr1'][ind]  = 0.077
smallPlanetSystems['pl_radeerr2'][ind]  = -0.077
smallPlanetSystems['pl_bmasse'][ind]     = 7.57
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.71
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.72

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'TOI-1685 b')
smallPlanetSystems['pl_rade'][ind]      = 1.70
smallPlanetSystems['pl_radeerr1'][ind]  = 0.07
smallPlanetSystems['pl_radeerr2'][ind]  = -0.07
smallPlanetSystems['pl_bmasse'][ind]     = 3.09
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.59
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.58

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'LHS 1815 b')
smallPlanetSystems['pl_rade'][ind]      = 1.088
smallPlanetSystems['pl_radeerr1'][ind]  = 0.064
smallPlanetSystems['pl_radeerr2'][ind]  = -0.064
smallPlanetSystems['pl_bmasse'][ind]     = 1.58
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.64
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.60

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'L 98-59 c')
smallPlanetSystems['pl_rade'][ind]      = 1.34
smallPlanetSystems['pl_radeerr1'][ind]  = 0.07
smallPlanetSystems['pl_radeerr2'][ind]  = -0.07
smallPlanetSystems['pl_bmasse'][ind]     = 2.42
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.35
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.34

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'L 98-59 d')
smallPlanetSystems['pl_rade'][ind]      = 1.58
smallPlanetSystems['pl_radeerr1'][ind]  = 0.08
smallPlanetSystems['pl_radeerr2'][ind]  = -0.08
smallPlanetSystems['pl_bmasse'][ind]     = 2.31
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.46
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.45

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'TOI-1235 b')
smallPlanetSystems['pl_rade'][ind]      = 1.69
smallPlanetSystems['pl_radeerr1'][ind]  = 0.08
smallPlanetSystems['pl_radeerr2'][ind]  = -0.08
smallPlanetSystems['pl_bmasse'][ind]     = 6.69
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.67
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.69

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'LTT 3780 b')
smallPlanetSystems['pl_rade'][ind]      = 1.32
smallPlanetSystems['pl_radeerr1'][ind]  = 0.06
smallPlanetSystems['pl_radeerr2'][ind]  = -0.06
smallPlanetSystems['pl_bmasse'][ind]     = 2.47
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.24
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.24

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'LTT 3780 c')
smallPlanetSystems['pl_rade'][ind]      = 2.33
smallPlanetSystems['pl_radeerr1'][ind]  = 0.135
smallPlanetSystems['pl_radeerr2'][ind]  = -0.135
smallPlanetSystems['pl_bmasse'][ind]     = 7.02
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.69
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.67

ind = np.argwhere(smallPlanetSystems['pl_name'] == 'GJ 1252 b')
smallPlanetSystems['pl_rade'][ind]      = 1.193
smallPlanetSystems['pl_radeerr1'][ind]  = 0.74
smallPlanetSystems['pl_radeerr2'][ind]  = -0.74
smallPlanetSystems['pl_bmasse'][ind]     = 1.32
smallPlanetSystems['pl_bmasseerr1'][ind] = 0.28
smallPlanetSystems['pl_bmasseerr2'][ind] = -0.28


Rp_Rs = ((np.array(smallPlanetSystems['pl_rade'])*u.R_earth)/(np.array(smallPlanetSystems['st_rad'])*u.R_sun)).decompose()
Teq = orb.Teq(np.array(smallPlanetSystems['st_teff'])*u.K, 0, np.array(smallPlanetSystems['st_rad'])*u.R_sun, np.array(smallPlanetSystems['pl_orbsmax'])*u.AU).decompose()

c = Column(Rp_Rs, name='pl_rp_rs')
smallPlanetSystems.add_column(c)
c = Column(Teq, name='pl_teq_a0')
smallPlanetSystems.add_column(c)

#######################################################################################################################
## Basic rejection of data ############################################################################################
#######################################################################################################################

vol_b = 4/3 * np.pi * np.array(smallPlanetSystems['pl_rade'])*u.R_earth**3      # volume of planets
rho_p = np.array(smallPlanetSystems['pl_bmasse'])*u.M_earth / vol_b
rho_p = rho_p.to(u.g/u.cm**3)

mask = np.ones(len(smallPlanetSystems))
#rho_mask = rho_p.value > 3
#mask *= rho_mask

pl_rad_err = np.mean([smallPlanetSystems['pl_radeerr1'], abs(smallPlanetSystems['pl_radeerr2'])], axis=0)
pl_mass_err = np.mean([smallPlanetSystems['pl_bmasseerr1'], abs(smallPlanetSystems['pl_bmasseerr2'])], axis=0)

mask *= pl_rad_err/smallPlanetSystems['pl_rade'] < 0.1
mask *= pl_mass_err/smallPlanetSystems['pl_bmasse'] < 0.15

mask = mask.astype(bool)

sample_intermediate = copy.deepcopy(smallPlanetSystems)
sample_intermediate = sample_intermediate[mask]

print(len(sample_intermediate))
sample_intermediate.pprint(show_unit=True)
print('saving out first pass at sample')
ascii.write(sample_intermediate, 'sample_intermediate.csv', format='csv', overwrite=True)

#######################################################################################################################
##### Selected sample by hand #########################################################################################
#######################################################################################################################

pl_names = ['GJ 1132 b', 'GJ 367 b', 'GJ 486 b', 'L 98-59 c', 'LHS 1140 b', 'LHS 1140 c', 'LHS 1478 b', 'LTT 1445 A b', 'LTT 3780 b', 'TOI-1468 b', 'TOI-1634 b', 'TRAPPIST-1 d']
pl_inds = [i for i, e in enumerate(sample_intermediate['pl_name']) if e in set(pl_names)]

sample = copy.deepcopy(sample_intermediate)
mask = np.zeros(len(sample['pl_name'])).astype(bool)
mask[pl_inds] = True
sample = sample[mask]

print(len(sample))
sample.pprint(show_unit=True)
ascii.write(sample, 'sample_final.dat', overwrite=True)

cmap = mpl.cm.inferno
norm = mpl.colors.Normalize(vmin=300, vmax=1000)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

#sample chosen by:
#high_rprs = np.argwhere(((pl_rad*u.R_earth)/(st_rad*u.R_sun)).decompose() > .03)

plt.figure(figsize=(15, 5))

Fe100 = ascii.read('/home/hannah/Research/Library/ZengCompositions/Fe_100.txt')
plt.plot(Fe100['col1'], Fe100['col2'], color='#873e23', lw=2, alpha=0.8)

Earthlike = ascii.read('/home/hannah/Research/Library/ZengCompositions/Fe_32p5_MgSiO3_67p5_Earth.txt')
plt.plot(Earthlike['col1'], Earthlike['col2'], color='#e28743', lw=2, alpha=0.8)

MgSiO3 = ascii.read('/home/hannah/Research/Library/ZengCompositions/MgSiO3_100.txt')
plt.plot(MgSiO3['col1'], MgSiO3['col2'], color='#eab676', lw=2, alpha=0.8)

MgSiO3_50_H2O_50 = ascii.read('/home/hannah/Research/Library/ZengCompositions/MgSiO3_50_H2O_50.txt')
plt.plot(MgSiO3_50_H2O_50['col1'], MgSiO3_50_H2O_50['col2'], color='#448fa2', lw=1.5, alpha=0.8)

plt.errorbar(smallPlanetSystems['pl_bmasse'], 
             smallPlanetSystems['pl_rade'], 
             xerr=np.mean([smallPlanetSystems['pl_bmasseerr1'], abs(smallPlanetSystems['pl_bmasseerr2'])], axis=0), 
             yerr=np.mean([smallPlanetSystems['pl_radeerr1'], abs(smallPlanetSystems['pl_radeerr2'])], axis=0),
             fmt='.',
             color='grey',
             ecolor='grey',
             elinewidth=3,
             alpha=0.7,
             zorder=500,
             )

for i in range(len(sample)):
    plt.plot(sample['pl_bmasse'][i],
             sample['pl_rade'][i], 
             'o',
             markersize=sample['pl_rp_rs'][i]*300,
             markeredgecolor='grey',
             color=mapper.to_rgba(sample['pl_teq_a0'][i]),
             alpha=0.9,
             zorder=1000
            )
    plt.text(sample['pl_bmasse'][i],
             sample['pl_rade'][i]+0.04, 
             sample['pl_name'][i],
             zorder=2000)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', fraction=0.08, pad=0.02, aspect=12).set_label(label='Planet Equilibrium Temperature (K)', size=14)

plt.ylim(0.65, 1.9) # radius
plt.xlim(0.3, 9.8) # mass

plt.xscale('log')
#massrad['massrad'].set_yscale('log')
plt.xlabel('Planet Mass ($M_{\oplus}$)', fontsize=15)
plt.ylabel('Planet Radius ($R_{\oplus}$)', fontsize=15)
#plt.title('', fontsize=16)

plt.tick_params(axis='both', labelsize=12)
plt.tick_params(axis='both', which='minor')#, labelbottom=False)
plt.grid(axis='both', which='both', alpha=0.4)

plt.tick_params(axis='x', which='minor')
ax = plt.gca()
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.1f}'))

plt.tight_layout()

plt.savefig('Figure_TerrestrialSample.png', dpi=800)
plt.show()