#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import datetime
import numpy as np
import pandas as pd

from zipfile import ZipFile
from bs4 import BeautifulSoup #Needs conda install -c anaconda lxml

import matplotlib as mpl
from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from harc_plot import geopack
import harc_plot

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

Re = 6371 # Radius of the Earth in km

regions = {}
rgn = {}
rgn['lat_0'] =  -90
rgn['lat_1'] =   90
rgn['lon_0'] = -180
rgn['lon_1'] =  180
regions['World'] = rgn

rgn = {}
rgn['lat_0'] =   20
rgn['lat_1'] =   50
rgn['lon_0'] = -130
rgn['lon_1'] =  -60
regions['US'] = rgn


def load_proposed_grapes(kmz_fpath):
    """
    Load the locations of proposed Grapes from a Google KMZ File.
    kmz_fpath: path to kmz file with Placemarks
    """
    with ZipFile(kmz_fpath, 'r') as kmz:
        kml = kmz.open(kmz.filelist[0].filename, 'r').read()
    soup = BeautifulSoup(kml, 'xml')

    # iterate over each TimeStep element in dwd:ForecastTimeSteps
    df_list = []
    steps = soup.find("kml:Document")
    for step in steps.find_all("kml:Placemark"):
        coords_str = step.find('kml:coordinates').text.split(',')
        lon        = float(coords_str[0])
        lat        = float(coords_str[1])
        name       = step.find('kml:name').text

        tmp = {}
        tmp['name'] = name
        tmp['lat']  = lat
        tmp['lon']  = lon
        df_list.append(tmp)

    df = pd.DataFrame(df_list)
    
    return df

def load_and_filter_existing_nodes(fpath,region=None,active_only=True):
    """
    Load Grape Locations from nodelist_status.csv.
    fpath: path to nodelist_status.csv
    region: None or string matching entry in regions dictionary.
            If in regions dictionary, only stations withing the specified
            region will be included.
    active_only: If True, will only return stations with Status == Data Logged.

    Returns DataFrame
    """

    nodes   = pd.read_csv(fpath)

    # Rename Latitude/Longitude Columns to match rest of this code convetion.
    cols    = {'Latitude':'lat','Longitude':'lon'}
    nodes   = nodes.rename(columns=cols)

    if active_only:
        # Get only active notes
        tf      = nodes['Status'] == 'Data logged'
        nodes   = nodes[tf].copy()

    if region is not None:
        # Filter by region.
        rgn_dct   = regions[region]
        rgn_lat_0 = rgn_dct['lat_0']
        rgn_lat_1 = rgn_dct['lat_1']
        rgn_lon_0 = rgn_dct['lon_0']
        rgn_lon_1 = rgn_dct['lon_1']

        tf = np.logical_and(nodes['lat'] >= rgn_lat_0,nodes['lat'] <= rgn_lat_1)
        nodes = nodes[tf].copy()


        tf = np.logical_and(nodes['lon'] >= rgn_lon_0,nodes['lon'] <= rgn_lon_1)
        nodes = nodes[tf].copy()

    return nodes

def compute_midpoints(nodes,txs,lat_key_in='lat',lon_key_in='lon'):
    """
    Compute midpoints between transmitters and receiver nodes and append to nodes
    data frame.

    nodes:  DataFrame containing receivers
    txs:    DataFrame containing transmitter information
    lat_key_in: Column name identifying latitudes in nodes DataFrame
    lon_key_in: Column name identifying longitudes in nodes DataFrame

    """
    # Compute TX Midpoints
    nodes_tmp = []
    for rinx,row in nodes.iterrows():
        for call,tx in txs.items():
            lat_0 = tx['lat']
            lon_0 = tx['lon']
            lat_1 = row[lat_key_in]
            lon_1 = row[lon_key_in]

            dist  = Re*harc_plot.geopack.greatCircleDist(lat_0,lon_0,lat_1,lon_1)
            azm   =    harc_plot.geopack.greatCircleAzm(lat_0,lon_0,lat_1,lon_1)
            res   = geopack.greatCircleMove(lat_0,lon_0,dist/2.,azm,alt=0,Re=Re)

            row[call+'_mid_lat'] = res[0][0]
            row[call+'_mid_lon'] = res[1][0]
        nodes_tmp.append(row)

    nodes = pd.DataFrame(nodes_tmp)

    return nodes

class EclipseData(object):
    def __init__(self,fname,meta={},track_kwargs={}):
        """
        Load a CSV containing eclipse obscuration data.
        """
        df = pd.read_csv(fname,comment='#')

        height = df['height'].unique()
        n_heights = len(height)
        assert n_heights  == 1, f'One height expected, got: {n_heights}'
        height = height[0]

        # Calculate vectors of center lats and lons.
        center_lats = np.sort(df['lat'].unique())
        center_lons = np.sort(df['lon'].unique())

        # Find the lat/lon step size.
        dlat = center_lats[1] - center_lats[0]
        dlon = center_lons[1] - center_lons[0]

        # Calculate vectors of boundary lats and lons.
        lat_0   = center_lats.min() - dlat/2.
        lat_1   = center_lats.max() + dlat/2.
        lats    = np.arange(lat_0,lat_1+dlat,dlat)

        lon_0 = center_lons.min() - dlon/2.
        lon_1 = center_lons.max() + dlon/2.
        lons    = np.arange(lon_0,lon_1+dlon,dlon)
        
        # These if statements are to handle an error that can occur
        # when dlat or dlon are very small and you get the wrong number
        # of elements due to a small numerical error.
        if len(lats) > len(center_lats)+1:
            lats=lats[:len(center_lats)+1]

        if len(lons) > len(center_lons)+1:
            lons=lons[:len(center_lons)+1]


        cshape      = (len(center_lats),len(center_lons))

        meta['fname']    = fname        
        self.meta        = meta
        self.df          = df
        self.center_lats = center_lats
        self.center_lons = center_lons
        self.cshape      = cshape
        self.lats        = lats
        self.lons        = lons
        self.dlat        = dlat
        self.dlon        = dlon
        self.height      = height
    
        # Load track data.
        self.track_kwargs = track_kwargs
        df_track_csv = fname.replace('MAX_OBSCURATION','ECLIPSE_TRACK')
        if os.path.exists(df_track_csv):
            df_track = pd.read_csv(df_track_csv,parse_dates=[0],comment='#')
            df_track = df_track.set_index('date_ut')
            
            self.df_track = df_track
        else:
            print('File not found: {!s}'.format(df_track_csv))
            print('No eclipse track data loaded.')
            self.df_track = None
        
    def get_obsc_arr(self,obsc_min=0,obsc_max=1):
        """
        Convert the Obscuration DataFrame to a 2D numpy array that can easily be plotted with pcolormesh.

        obsc_min: obscuration values less than this number will be converted to np.nan
        obsc_max: obscuration values greater than this number will be converted to np.nan
        """
        df = self.df.copy()
        
        if obsc_min > 0:
            tf = df['obsc'] < obsc_min
            df.loc[tf,'obsc'] = np.nan
            
        if obsc_max < 1:
            tf = df['obsc'] > obsc_max
            df.loc[tf,'obsc'] = np.nan
            
        obsc_arr    = df['obsc'].to_numpy().reshape(self.cshape)
        
        return obsc_arr
    
    def overlay_obscuration(self,ax,obsc_min=0,alpha=0.65,cmap='gray_r',vmin=0,vmax=1,zorder=1):
        """
        Overlay obscuration values on a map.
        
        ax: axis object to overlay obscuration on.
        """
        obsc_min = self.meta.get('obsc_min',obsc_min)
        alpha    = self.meta.get('alpha',alpha)
        cmap     = self.meta.get('cmap',cmap)
        vmin     = self.meta.get('vmin',vmin)
        vmax     = self.meta.get('vmax',vmax)
        
        lats     = self.lats
        lons     = self.lons
        obsc_arr = self.get_obsc_arr(obsc_min=obsc_min)
        pcoll    = ax.pcolormesh(lons,lats,obsc_arr,vmin=vmin,vmax=vmax,cmap=cmap,zorder=zorder,alpha=alpha)
        
        return {'pcoll':pcoll}
    
    def overlay_track(self,ax,annotate=False,**kw_args):
        """
        Overlay eclipse track on map.
        
        ax: axis object to overlay track values on.
        """
        if self.df_track is None:
            print('No eclipse track data loaded.')
            return
        else:
            df_track = self.df_track
        
        # Annotate places specifed in meta['track_annotate'], but make sure
        # the places are actually on the track.

        # Set default annotation style dictionary.
        tp = {}
        tp['bbox']          = dict(boxstyle='round', facecolor='white', alpha=0.75)
        tp['fontweight']    = 'bold'
        tp['fontsize']      = 12
        tp['zorder']        = 975
        tp['va']            = 'top'

        tas_new = []
        tas = self.meta.get('track_annotate',[]).copy()
        for ta in tas:
            ta_se   = ta.get('startEnd','start')

            tr_lats = df_track['lat'].to_numpy()
            tr_lons = df_track['lon'].to_numpy()

            ta_lat = ta['lat']
            ta_lon = ta['lon']

            ta_lats = np.ones_like(tr_lats)*ta_lat
            ta_lons = np.ones_like(tr_lons)*ta_lon

            ta_rngs = harc_plot.geopack.greatCircleDist(ta_lats,ta_lons,tr_lats,tr_lons)*Re
            ta_inx  = np.argmin(ta_rngs)

            ta['lat']   = tr_lats[ta_inx]
            ta['lon']   = tr_lons[ta_inx]

            if ta_se == 'start':
                ta['text']  = df_track.index[ta_inx].strftime('%d %b %Y\n%H%M UT')
            else:
                ta['text']  = df_track.index[ta_inx].strftime('%H%M UT')
            
            tp_tmp  = tp.copy()
            ta_tp   = ta.get('tp',{})
            tp_tmp.update(ta_tp)
            ta['tp']    = tp_tmp

            tas_new.append(ta)
        tas = tas_new

        for ta in tas:
            ta_lat  = ta['lat']
            ta_lon  = ta['lon']
            ta_text = ta['text']
            ta_tp   = ta['tp']
            ax.text(ta_lon,ta_lat,'{!s}'.format(ta_text),**ta_tp)

        for inx,(rinx,row_0) in enumerate(df_track.iterrows()):
            lat_0 = row_0['lat']
            lon_0 = row_0['lon']

            if annotate:
                if inx == 0:
                    ax.text(lon_0,lat_0,'{!s}'.format(rinx.strftime('%H%M UT')),**tp)
                if inx == len(df_track)-1:
                    ax.text(lon_0,lat_0,'{!s}'.format(rinx.strftime('%H%M UT')),**tp)

            if inx == len(df_track)-1:
                continue


            row_1 = df_track.iloc[inx+1]
            lat_1 = row_1['lat']
            lon_1 = row_1['lon']
            ax.annotate('', xy=(lon_1,lat_1), xytext=(lon_0,lat_0),zorder=950,
                    xycoords='data', size=20,
                    arrowprops=dict(facecolor='red', ec = 'none', arrowstyle="simple",
                connectionstyle="arc3,rad=-0.1"))        

        return {}


def overlay_grapes(nodes,ax,style='existing',label_midpoints=True):
    # Plot Ground Locations of Grapes
    lats    = nodes['lat'].values
    lons    = nodes['lon'].values

    if style=='existing':
        label   = 'Existing HamSCI Sites'
        color   = 'k'
        size    = 50
        marker  = '*'
        zorder  = 1000
    elif style == 'proposed':
        label   = 'Additional Proposed Sites'
        color   = 'k'
        size    = 20
        marker  = '^'
        zorder  = 1001

    ax.scatter(lons,lats,label=label,color=color,s=size,marker=marker,zorder=zorder)

    for tx_call,tx in txs.items():
        lats    = nodes[tx_call+'_mid_lat'].values
        lons    = nodes[tx_call+'_mid_lon'].values
        if label_midpoints:
            label   = '{!s}-RX Midpoints'.format(tx_call)
        else:
            label   = None
        color   = tx['color']
        size    = 35
        zorder  = 1002
#        if style=='existing':
#            lats    = nodes[tx_call+'_mid_lat'].values
#            lons    = nodes[tx_call+'_mid_lon'].values
#            label   = '{!s}-RX Existing Midpoints'.format(tx_call)
#            color   = tx['color']
#            size    = 25
#            zorder  = 1002
#
#        elif style == 'proposed':
#            lats    = nodes[tx_call+'_mid_lat'].values
#            lons    = nodes[tx_call+'_mid_lon'].values
#            label   = '{!s}-RX Proposed Midpoints'.format(tx_call)
#            color   = tx['color']
#            size    = 25
#            zorder  = 1003
        ax.scatter(lons,lats,label=label,color=color,s=size,ec='k',zorder=zorder)

if __name__ == '__main__':
    output_dir      = 'output/map'
    region          = 'US'
    plot_eclipse    = True

    harc_plot.gl.clear_dir(output_dir)

    # ## Define Station Locations
    txs = {}
    tx = {}
    tx['lat']       =  40.68
    tx['lon']       = -105.04
    tx['color']     = 'red'
    txs['WWV']      = tx

    tx = {}
    tx['lat']       =  45.291165502
    tx['lon']       = -75.753663652
    tx['color']     = 'yellow'
    txs['CHU']  = tx

    # Load in data about existing Grapes.
    node_csv        = os.path.join('data','grape','nodelist_status.csv')
    existing_nodes  = load_and_filter_existing_nodes(node_csv,region=region)
    existing_nodes  = compute_midpoints(existing_nodes,txs)

    # Load in proposed Grapes
    proposed_nodes_csv  = os.path.join('data','grape','proposed_grapes.kmz')
    proposed_nodes      = load_proposed_grapes(proposed_nodes_csv)
    proposed_nodes      = compute_midpoints(proposed_nodes,txs)
    proposed_out_csv    = os.path.join(output_dir,'proposed_grapes.csv')
    proposed_nodes.to_csv(proposed_out_csv)

    if plot_eclipse:
        # ## Load Eclipse Data
        eclipses = {}

        meta = {}
        meta['obsc_min']    = 0.85
        meta['alpha']       = 0.25
        meta['cmap']        = mpl.cm.Purples
        fname = 'data/eclipse_calc/20231014.1400_20231014.2100_300kmAlt_0.2dlat_0.2dlon/20231014.1400_20231014.2100_300kmAlt_0.2dlat_0.2dlon_MAX_OBSCURATION.csv.bz2'

        tas = []
        ta  = {}
        ta['lat']       = 45.
        ta['lon']       = -128.
        ta['startEnd']  = 'start'
        ta['tp']        = {'va':'bottom'}
        tas.append(ta)

        ta  = {}
        ta['lat']       = 22.
        ta['lon']       = -90. 
        ta['startEnd']  = 'end'
        tas.append(ta)
        meta['track_annotate'] = tas

        ecl = EclipseData(fname,meta=meta)
        eclipses['2023']    = ecl

        meta = {}
        meta['obsc_min']    = 0.975
        fname                   = 'data/eclipse_calc/20240408.1500_20240408.2100_300kmAlt_0.2dlat_0.2dlon/20240408.1500_20240408.2100_300kmAlt_0.2dlat_0.2dlon_MAX_OBSCURATION.csv.bz2'
        tas = []
        ta  = {}
        ta['lat']       = 20
        ta['lon']       = -107.
        ta['startEnd']  = 'start'
        tas.append(ta)

        ta  = {}
        ta['lat']       =  47.
        ta['lon']       = -61. 
        ta['startEnd']  = 'end'
        tas.append(ta)
        meta['track_annotate'] = tas
        ecl                     = EclipseData(fname,meta=meta)
        ecl = EclipseData(fname,meta=meta)
        eclipses['2024']    = ecl

    # Plot on a Map ################################################################ 
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18,14))
    ax  = fig.add_subplot(1,1,1,projection=projection)
    border_color = '0.80'
    ax.coastlines(color=border_color)
    ax.add_feature(cfeature.BORDERS,edgecolor=border_color)
    ax.gridlines(draw_labels=True)

    if plot_eclipse:
        for ecl_year,ecl in eclipses.items():
            ecl.overlay_obscuration(ax)
            ecl.overlay_track(ax)

    alpha = 0.5
    # Plot Transmitter
    for tx_call,tx in txs.items():
        tx_lat      = tx['lat']
        tx_lon      = tx['lon']
        tx_color    = tx['color']

        ax.scatter(tx_lon,tx_lat,marker='*',s=650,color=tx_color,ec='black')

        kws = {}
        kws['fontsize']     = 16
        kws['fontweight']   = 'bold'
        kws['ha']           = 'center'
        kws['x']            = tx_lon
        kws['y']            = tx_lat+1.00
        kws['s']            = tx_call

        ax.text(**kws)

    tf      = proposed_nodes['name'] == 'Proposed Grape 2'
    exst_df = proposed_nodes[~tf]
    overlay_grapes(exst_df,ax,style='existing',label_midpoints=False)

    prop_df = proposed_nodes[tf]
    overlay_grapes(prop_df,ax,style='proposed')

    ax.legend(loc='lower right',fontsize='medium')

    # Set Map Limits
    rgn_dct   = regions[region]
    rgn_lat_0 = rgn_dct['lat_0']
    rgn_lat_1 = rgn_dct['lat_1']
    rgn_lon_0 = rgn_dct['lon_0']
    rgn_lon_1 = rgn_dct['lon_1']
    ax.set_xlim(rgn_lon_0,rgn_lon_1)
    ax.set_ylim(rgn_lat_0,rgn_lat_1)

    ax.set_title('Locations of Proposed Grape 2 Receivers')

    fpath = os.path.join(output_dir,'map.png')
    fig.savefig(fpath,bbox_inches='tight')
    import ipdb; ipdb.set_trace()
