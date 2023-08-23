import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# GENERAL PLOT OPTIONS
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

#parameter = [u'omegam', u'chi2',u'LSST_DZ_S1',u'LSST_DZ_S2',u'LSST_DZ_L1',u'LSST_DZ_L2']
parameter = [u'omegam', u'chi2',u'LSST_DZ_S1',u'LSST_DZ_L1']
chaindir=os.getcwd()

analysissettings={'smooth_scale_1D':0.05, 'smooth_scale_2D':0.2,'ignore_rows': u'0.5',
'range_confidence' : u'0.0005'}

analysissettings2={'smooth_scale_1D':0.05,'smooth_scale_2D':0.2,'ignore_rows': u'0.0',
'range_confidence' : u'0.0005'}

root_chains = (
  'EXAMPLE_MCMC18',
  'EXAMPLE_MCMC19',
)


# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.addDerived(10*p.omegam,name='om10',label='{10 \\Omega_m}')
samples.addDerived(100*p.omegab,name='ob100',label='{100 \\Omega_b}')
samples.addDerived(10*p.ns,name='ns10',label='{10 n_s}')
samples.saveAsText(chaindir + '/.VM_P22_TMP5')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[1],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.addDerived(10*p.omegam,name='om10',label='{10 \\Omega_m}')
samples.addDerived(100*p.omegab,name='ob100',label='{100 \\Omega_b}')
samples.addDerived(10*p.ns,name='ns10',label='{10 n_s}')
samples.saveAsText(chaindir + '/.VM_P22_TMP6')

#GET DIST PLOT SETUP
g=gplot.getSubplotPlotter(chain_dir=chaindir,
  analysis_settings=analysissettings2,width_inch=6.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 13.0
g.settings.legend_fontsize = 14.0
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

param_3d = None
g.triangle_plot([
  chaindir + '/.VM_P22_TMP5',
  chaindir + '/.VM_P22_TMP6'],
parameter,
plot_3d_with_param=param_3d,line_args=[
{'lw': 1.2,'ls': 'solid', 'color':'green'},
{'lw': 1.2,'ls': 'solid', 'color':'lightcoral'},
{'lw': 1.2,'ls': '--', 'color':'black'},
{'lw': 1.6,'ls': '-.', 'color': 'maroon'},
{'lw': 1.6,'ls': 'solid', 'color': 'indigo'},
{'lw': 1.6,'ls': 'solid', 'color': 'blue'},
],
contour_colors=['green','lightcoral','black','maroon','indigo','blue'],
contour_ls=['solid','solid','--','-.','solid','solid'], 
contour_lws=[1.0,1.0,1.5,1.5,1.0,1.0],
filled=[True,False,False,False,False,False],
shaded=False,
legend_labels=[
'LSST-Y1 3x2pt CR + Bias/M fixed flat prior LZ/SZ',
'LSST-Y1 3x2pt CR + Bias/M fixed',
],
legend_loc=(0.35, 0.8))


g.export()