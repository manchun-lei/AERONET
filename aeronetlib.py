# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:21:47 2023

@author: Manchun LEI

Package name:

Module name:
    aeronetlib
    ---------------
    This module is used to read the AERONET data file
    
"""
import numpy as np
import pandas as pd
import re

def is_aeronet(file):
    f = open(file,"r")
    line = f.readline()
    f.close()
    return 'AERONET' in line

def read_metadata(srcfile):
    """
    Reads the first 6 lines of the source file to parse station name, 
    version number, and AOD level.
    """
    try:
        # Using 'with' statement ensures the file is closed automatically
        with open(srcfile, "r", encoding="utf-8") as f:
            # List comprehension to read the first 6 lines efficiently
            lines = [f.readline().strip() for _ in range(6)]
    except FileNotFoundError:
        print(f"Error: The file '{srcfile}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Check if we have enough lines to prevent IndexError
    if len(lines) < 3:
        print("Error: File header is too short to contain metadata.")
        return None

    # Parse station name from the second line
    station_name = lines[1]
    
    # Initialize variables to None to avoid NameError if regex fails
    version_number = None
    level_number = None

    # Parse version and level from the third line (index 2)
    header_info = lines[2]
    
    # Regex for Version: matches 'Version' followed by digits
    v_match = re.search(r'Version\s+(\d+)', header_info)
    if v_match:
        version_number = int(v_match.group(1))
        
    # Regex for AOD Level: matches digits, optionally including a decimal point
    l_match = re.search(r'AOD Level\s+(\d+\.?\d*)', header_info)
    if l_match:
        level_number = float(l_match.group(1))

    df = pd.read_csv(srcfile, skiprows=6)
    latitude = df['Site_Latitude(Degrees)'].values[0]
    longitude = df['Site_Longitude(Degrees)'].values[0]
    elevation = df['Site_Elevation(m)'].values[0]

    # read available wavelength
    aod_cols = [col for col in df.columns if re.match(r'Exact_Wavelengths_of_AOD\(um\)_\d+nm', col)]
    wls_nominal = np.array([int(re.findall(r'\d+', col)[0]) for col in aod_cols])
    wls_effective = np.array(df[aod_cols].values[0]) * 1000

    indices = np.where(wls_effective>0)[0]
    wls_effective = wls_effective[indices]
    wls_nominal = wls_nominal[indices]
    indices = np.argsort(wls_effective)
    wls_effective = wls_effective[indices]
    wls_nominal = wls_nominal[indices]
    
    
    return {
        'station': station_name, 
        'version': version_number, 
        'level': level_number,
        'latitude':latitude,
        'longitude':longitude,
        'elevation':elevation,
        'wl_effective':wls_effective,
        'wl_nominal':wls_nominal
    }

def read_aod_array(srcfile, wls_nominal):

    df = pd.read_csv(srcfile, skiprows=6)
    aod_nominal = ['AOD_'+str(wl)+'nm' for wl in wls_nominal]

    aod_array = df[aod_nominal].values
    # wls = [int(re.findall(r'\d+', col)[0]) for col in aod_cols]
    
    # indices = np.argsort(wls)
    # wls = np.array(wls)[indices]
    # aod_cols = [aod_cols[i] for i in indices]
    # aod_array = aod_array[:,indices]

    # Get the time axis (n) as a separate array
    time_axis = pd.to_datetime(
        df['Date(dd:mm:yyyy)'] + ' ' + df['Time(hh:mm:ss)'], 
                format='%d:%m:%Y %H:%M:%S'
        ).values
    day_of_year_fraction = df['Day_of_Year(Fraction)'].values
    szas = df['Solar_Zenith_Angle(Degrees)'].values

    return {'time':time_axis, 'doy_fraction':day_of_year_fraction, 'sza':szas, 'aod':aod_array}

def find_exact_aod(wls, aods, wl):
    indice = np.where(wls==wl)[0]
    if len(indice)>0:
        return aods[indice[0]]
    else:
        return -999

def compute_aod(wls, aods, wl, method='tli'):
    '''
    'tli' - two band linear interpolation
    'slr' - simple linear regression
    'qdp' - quadratic polynomial
    '''
    out = -999
    mask = aods<=0
    if np.sum(mask)>0:
        print('no available aod')
        return out

    out = find_exact_aod(wls, aods, wl)
    if out>0:
        print('find exact aod value')
        return out

    if len(wls)<2:
        print('not enough avaible aods')
        return out
    else:
        y = np.log(aods)
        x = np.log(wls)
        if method=='tli':
            indices = np.where(wls<wl)[0]
            if len(indices)<1:
                print('left wl not available')
                return out
            x1 = x[indices[-1]]
            y1 = y[indices[-1]]
            indices = np.where(wls>wl)[0]
            if len(indices)<1:
                print('right wl not available')
                return -999
            x2 = x[indices[0]]
            y2 = y[indices[0]]
            p = np.polyfit([x1,x2],[y1,y2],1)
        elif method=='slr':
            p = np.polyfit(x,y,1)
        elif method=='qdp':
            p = np.polyfit(x,y,2)
        else:
            print('wrong method name:',method)
            return out
        out = np.polyval(p, np.log(wl))
        return np.exp(out)

