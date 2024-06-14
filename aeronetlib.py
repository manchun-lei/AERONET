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
import csv
import re

def is_aeronet(file):
    f = open(file,"r")
    line = f.readline()
    f.close()
    return 'AERONET' in line

def read_metadata(file):
    f = open(file,"r")
    lines = []
    for i in range(6):
        lines.append(f.readline().strip())
    f.close()
    # read station_name
    station_name = lines[1]
    # read version
    match = re.search(r'Version\s+(\d+)',lines[2])
    if match:
        version_number = int(match.group(1))
    # read level number
    match = re.search(r'AOD Level\s+(\d+\.\d+|\d+)',lines[2])
    if match:
        level_number = float(match.group(1))
    return station_name,version_number,level_number
    
def string_to_number(s):
    try:
        if s.isdigit():
            a = int(s)
        else:
            a = float(s)
    except ValueError:
        a = s
    return a

def read_csv(csvfile,skiprows=6):
    #load csv to a dict
    #get the header as keys
    with open(csvfile, 'r') as f:
        # skip rows
        for _ in range(skiprows):
            next(f)
        csvreader = csv.reader(f)
        rows = list(csvreader)
    keys = rows[0]
    #load as dict for each row
    with open(csvfile, 'r') as f:
        # skip rows
        for _ in range(skiprows):
            next(f)
        csvreader = csv.DictReader(f)
        rows = list(csvreader)
    # read data from csv into a 2 dimension list
    data = [[] for i in range(len(keys))]
    for row in rows:
        for i in range(len(keys)):
            v = string_to_number(row[keys[i]])
            data[i].append(v)
    #create the dict
    mydict = {}
    for i in range(len(keys)):
        mydict[keys[i]] = data[i]
    return mydict

def extract_from_date(ds,yyyy,mm,dd):
    dates = np.array(ds['Date(dd:mm:yyyy)'])
    date = str(dd).zfill(2)+":"+str(mm).zfill(2)+":"+str(yyyy).zfill(4)
    ind = np.where(dates==date)[0]
    ds1 = {}
    for keyname in ds.keys():
        ds1[keyname] = np.array(dict_in[keyname])[ind]
    return ds1

def extract_from_datetime(ds,yyyy,mm,dd,hh,mt,ss):
    dates = np.array(ds['Date(dd:mm:yyyy)'])
    date = str(dd).zfill(2)+":"+str(mm).zfill(2)+":"+str(yyyy).zfill(4)
    times = np.array(ds['Time(hh:mm:ss)'])
    time = str(hh).zfill(2)+":"+str(mt).zfill(2)+":"+str(ss).zfill(2)

    ind = np.where(np.logical_and(dates==date,times==time))[0]
    ds1 = {}
    for keyname in ds.keys():
        ds1[keyname] = np.array(ds[keyname])[ind]
    return ds1

def get_effective_aods(ds):
    #load the valide aod values and it's exact wavelength in um
    keynames = ds.keys()
    pattern = re.compile(r'AOD_(\d+)nm')
    key_aods = []
    wls_nm = [] # nominative wavelength
    vals = []
    for s in keynames:
        match = pattern.search(s)
        if match:
            key_aods.append(s)
            wls_nm.append(int(match.group(1)))
    for key in key_aods:
        vals.append(ds[key][0])
    wls_nm = np.array(wls_nm)
    vals = np.array(vals)
    ind = np.where(vals>0)
    ef_vals = vals[ind]
    ef_wls_nm = wls_nm[ind]
    ind_sort = np.argsort(ef_wls_nm)
    ef_wls_nm_sort = ef_wls_nm[ind_sort]
    ef_vals_sort = ef_vals[ind_sort]
    # find the exact wavelength
    ef_wls_sort = []
    for wl_nm in ef_wls_nm_sort:
        ef_wls_sort.append(ds['Exact_Wavelengths_of_AOD(um)_{:d}nm'.format(wl_nm)][0])
    return np.array(ef_wls_sort),ef_vals_sort
    
class Aeronet(object):
    def __init__(self,file=None):
        self.ds = {}
        if file is not None:
            if is_aeronet(file):
                self.station_name,self.version_number,self.level_number = read_metadata(file)
                self.ds = read_csv(file)
        
    def extract_from_date(self,yyyy,mm,dd):
        return extract_from_date(self.ds,yyyy,mm,dd,hh,mt,ss)

    def extract_from_datetime(self,yyyy,mm,dd,hh,mt,ss):
        return extract_from_datetime(self.ds,yyyy,mm,dd,hh,mt,ss)
        
    def extract_from_datetime_effective_aods(self,yyyy,mm,dd,hh,mt,ss):
        ds1 = extract_from_datetime(self.ds,yyyy,mm,dd,hh,mt,ss)
        return get_effective_aods(ds1)

        
def angstrom_formula_1(wls,alpha,wl1,aod1):
    return aod1 * (wls/wl1) ** (-alpha)

def angstrom_formula_2(wls,alpha,beta):
    return beta * wls ** (-alpha)
    
def calculate_angstrom(wls,aods):
    # calculate the angstrom coef alpha from two aods 
    # aod1 = aod2*(wl1/wl1)**(-alpha)
    wl1,wl2 = wls
    aod1,aod2 = aods
    return -1*(np.log(aod1/aod2))/(np.log(wl1/wl2))


def linregress_angstrom(wls,aods):
    # estimate the AOD curve using linear regression
    # aod = beta * wls **(-alpha), where beta = aod1/(wl1**(-alpha)), here we do not kown exact value of the wl1 and aod1
    # use this methode, can get lower RMSE with multiple measurements of AOD,
   
    aods_log = np.log(aods)
    wls_log = np.log(wls)
    p = np.polyfit(wls_log,aods_log,1)
    alpha = -p[0]
    beta = np.exp(p[1])

    return alpha,beta

def calculate_aod550(wls,aods,method='linregress'):
    # estimate the tau_aer_550 value
    wl550 = 550
    if np.min(wls)<1:
        wl550 = 0.55
    if method=='linregress':
        alpha,beta = fit_llsq(wls,aods)
        aod550 = beta*wl550**(-alpha)
    elif method=='440-675nm':
        wl440 = 440
        wl675 = 675
        if np.min(wls)<1:
            wl440 = 0.44
            wl675 = 0.675
        difs440 = np.abs(wls-wl440)
        difs675 = np.abs(wls-wl675)
        ind440 = np.where(difs440==np.min(difs440))[0][0]
        ind675 = np.where(difs675==np.min(difs675))[0][0]
        alpha = calculate_angstrom((wls[[ind440,ind675]]),(aods[[ind440,ind675]]))
        aod550 = aods[ind440]*(wl550/wls[ind440])**(-alpha)    
    else:
        aod550 = -1
    
    return aod550

