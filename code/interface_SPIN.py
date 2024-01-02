 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 2020
Last update:
    
Front-end implementation of the SPIN method

@author: Timothé Beaufils
mail: timothe.beaufils@pik-potsdam.de
"""

import os
import numpy as np
import core_SPIN as SPIN
import time
import pandas as pd
import datetime

##### Main

def spin(name,starting_year,ending_year,base_table,gdp="auto",time_steps=1,
         regions="auto",trade_zones="auto",trade_files="auto",gdp_zones="auto",
         dual_mode=False,sectors=26,eps=0.0001,it=10000):
    """
    Launch a run of the Scenario Based Projection for the International Trade Network Algorithm
    Save the projected table as .csv files

    Parameters
    ----------
    name : str
        Name under which projections will be saved.
    starting_year : int
        Base year for the projections
    ending_year : int
        Final year of projection
    base_table : string
        MRIO table to use as a base for the projections
    gdp : string, optional
        Name of the GDP file in 'data/GDP/'
        Can be left empty if file is saved under name paramter.
        GDP file contains gdp rate in absolute number for each gdp gdp zone
        Input file has to be a single csv file
        Each column corresponding to a year and each row to a regional entity.
    time_steps : int, optional
        Duration between two steps, in years. The default is 1.
    regions: string, optional
        Name of the file describing the regions in 'data/regional_aggregation'.
        If left empty, each country corresponds to a region
        If 'world', world is projected as a single region.
    trade_zones: string, optional
        Name of the file describing the trade zones in 'data/regional_aggregation'.
        Trade relations should be defined between regions/trade zones.
        If left empty, trade zones correspond to regions.
        If 'world',trade relations are defined with regard to the rest of the world.
    trade_files: string, optional
        Can be left empty if file is saved under name paramter.
        Name of the imports and exports files in 'data/trade'.
        Else, files should be stored as csv file for each year, under 'name_imports_year.csv' and 'name_exports_year.csv'
        In each file, each row corresponds to a country and each column corresponds to a trade zone
    gdp_zones: string, optional
        Name of the file describing subregional gdp zones in 'data/regional_aggregation'. 
        If left empty, gdp is constrained at the scale of regions
    dual_mode : bool, optional
        If True, trade is constrained for each region
        with regard to intra zone trade and extra zone trade, only.
        If False, trade is constrained for each region
        with regard to every trade zone.
        The default is False.
    sectors : 
        Number of intermediate sectors by country.
        Note that final demand is not included here.
        The default is 26, corresponding to EORA 26.
    eps : float, optional
        Acceptable error for the optimization. The default is 0.0001.
    it: Int, optional
        Maximal number of iterations for the RAS algorithm. The default is 1000.

    Returns
    -------
    None. Saves the projected table and the optimization report as csv files

    """
    print('Launching projection '+name+'...')
    print('Loading input files...')
    years = np.arange(starting_year,ending_year,time_steps)
    start = time.time()
    t,y,v,countries=load_base_IOT(base_table,starting_year,sectors)
    
    #Load inputs
    
    if regions == "auto":  #Load zones for projections
        regions=[[i] for i in range(countries)]
    elif regions== 'world':
        regions = [[i for i in range(countries)]]
    else:
        regions = load_zones(regions)
    
    #Load sets of economic zones
    if trade_zones == "auto": #Load zones for bilateral trade
        trade_zones = regions
    elif trade_zones == 'world':
        trade_zones = [[i for i in range(countries)]]
    else:
        trade_zones = load_zones(trade_zones)
    if len(trade_zones)==1: #Force no dual mode if world is a single trade zone
        dual_mode=False
    
    if gdp_zones =="auto":    #Load zones for GDP
        gdp_zones = regions
    elif gdp_zones == 'countries':
        gdp_zones = [[i] for i in range(countries)]
    else:
        gdp_zones = load_zones(gdp_zones)
        
    #Load parameters
    if gdp != "auto": #Load gdp matrix
        gdp = load_gdp(gdp,len(years))
    else:
        gdp = load_gdp(name,len(years))

    if trade_files != "auto":
        imports,exports = load_trade_files(trade_files,years,regions,trade_zones)
    else:
        imports,exports = load_trade_files(name,years,regions,trade_zones)
          
    assert abs(np.sum(exports)/np.sum(imports)-1)<eps, 'Total imports does not equal total exports'
    report = [[None]*5*len(years)]*(len(regions)+len(trade_zones)**2+2)
    if dual_mode:
        report = [[None]*5*len(years)]*(len(regions)+len(trade_zones)+1+2)
    report=np.array(report)
    mid=time.time()
    print('Input files loaded in '+str(datetime.timedelta(seconds=mid-start)))
    print()
    for step,year in enumerate(years):
        print('Projection of year '+str(year+time_steps))
        print()
        startlap = time.time()
        t1,y1,v1,report_int = SPIN.spin_projection(t, y, v, imports[step], exports[step], gdp[:,step],trade_zones,gdp_zones,regions,eps=eps,it=it,countries=countries,sectors=sectors,dual_mode=dual_mode)
        report[0,5*step:5*(step+1)]=year+time_steps
        outputs = np.sum(t1,axis=1) + np.sum(y1,axis=1)
        inputs = np.sum(t1,axis=0) + v1
        outputs[outputs==0] =1
        inputs[inputs==0] =1
        diff = np.max(np.abs(outputs/inputs-1))
        print('Max desequilibrium observed: '+str(diff))
        print('Saving projection of year '+str(year+time_steps)+'...')
        save_projection(t1,y1,v1,name,year+time_steps)
        report[1:,5*step:5*(step+1)]=np.transpose(report_int)
        t,y,v = t1,y1,v1
        endlap = time.time()
        print("Year "+str(year+1)+" computed in "+str(datetime.timedelta(seconds=endlap-startlap)))
        print()
        print('********')
        print()
    end = time.time()
    print('Projection achieved in '+str(datetime.timedelta(seconds=end-start)))
    np.savetxt(os.path.join('..','output',str(name)+'_'+str(starting_year)+'_Computation_Report.csv'),report,delimiter=',',fmt=('%s,%s,%s,%s,%s,')*len(years))


##### Saving modules
def save_projection(t,y,v,name,step):
    '''
    Save projections as csv files.

    Parameters
    ----------
    t : 2D numpy array
        Interindustry matrix.
    y : 2D numpy array
        Final demand matrix.
    v : 1D numpy array
        Value added array.
    name : string
        Prefix under which projections will be saved.
    step : int
        Projected year.

    Returns
    -------
    None. Files are saved as csv in '/Projections'

    '''
    np.savetxt(os.path.join('..','output',str(name)+'_'+str(step)+'_T.csv'),t,delimiter=',')
    np.savetxt(os.path.join('..','output',str(name)+'_'+str(step)+'_FD.csv'),y,delimiter=',')
    np.savetxt(os.path.join('..','output',str(name)+'_'+str(step)+'_VA.csv'),v,delimiter=',')

##### Loading modules
def load_zones(file):
    '''
    Load and treat a file describing economic zones.
    Input file should have 2 columns, with country names on the first and region name on the second.
    Country names are not important, countries should be ordere by index, corresponding to the reference database index.
    Regions are ordered by names.
    Region names can be arbitrarly defined.
    Description file should be stored in '/Data/Economic_zones'

    Parameters
    ----------
    file : string
        Name of the csv file.

    Returns
    -------
    output : list of list of ints
        Zone composition: each list stores the index of the countries composing a region.

    '''
    output = [[]]
    zones = pd.read_csv(os.path.join('..','data','regional_aggregation',file+'.csv'),delimiter=',',header=0,names=['Countries','Economic zone'])
    zones['id'] = range(len(zones))
    zones.sort_values(by=['Economic zone','id'],inplace=True)
    zonelist = zones.to_numpy()
    ref = zonelist[0,1]
    i=0
    for row in zonelist:
        if row[1]==ref:
            output[i].append(row[2])
        else:
            output.append([row[2]])
            ref = row[1]
            i+=1
    return output

def load_gdp(file,steps):
    """
    Load the file of gdp by country and prepare it
    Parameters
    ----------
    file : string
        Name of the file in the 'Data/GDP' directory
    steps : int
        Number of time steps

    Returns
    -------
    gdp : 2D numpy array
        gdp matrix, with each column corresponding to a time step
    """
    data = np.loadtxt(os.path.join('..','data','GDP',file+'.csv'),delimiter=',')
    if data.ndim==1:
        return data.reshape((len(data),1))
    else:
        return data

def load_base_IOT(version,year,sectors=26):
    """
    Load the MRIO to build the projections on

    Parameters
    ----------
    version : String
        Name of the database to use.
    year : Int
        Starting year of the projection.
    sectors : int
        Number of intermediate sectors per country.

    Returns
    -------
    t : 2D numpy array
        Interindustry matrix
    y : 2D numpy array
        Final demand matrix
    v : 1D numpy array
        Value added array
    countries : Int
        Number of countries
    labels : 2D List
        List of labels for the interindustry matrix.

    """
    t = np.loadtxt(os.path.join('..','data','MRIOT',version+'_'+str(year)+'_T.csv'),delimiter=',')
    y = np.loadtxt(os.path.join('..','data','MRIOT',version+'_'+str(year)+'_FD.csv'),delimiter=',')
    v = np.loadtxt(os.path.join('..','data','MRIOT',version+'_'+str(year)+'_VA.csv'),delimiter=',')
    countries = int(len(t)/sectors)
    return t,y,v,countries

def load_trade_files(trade_files,years,regions,trade_zones):
    '''
    Load imports and exports files for projections zones

    Parameters
    ----------
    trade_files : string
        Name prefix of the imports and exports files.
    years : list of int
        List of years.
    regions : list of list of ints
        Description of the regions.
    trade_zones : list of list of ints
        Description of the trade zones.

    Returns
    -------
    imports : 3D numpy array
        List of imports from trade zones to regions.
        (year x regions x trade_zones).
    exports : 3D numpy array
        List of exports from regions to trade zones.
        (year x regions x trade_zones).

    '''
    imports = np.zeros((len(years),len(regions),len(trade_zones)))
    exports = np.zeros((len(years),len(regions),len(trade_zones)))
    for i,y in enumerate(years):
        imports[i] = np.loadtxt(os.path.join('..','data','trade',trade_files+'_imports_'+str(y+1)+'.csv'),delimiter=',').reshape((len(regions),len(trade_zones)))
        exports[i] = np.loadtxt(os.path.join('..','data','trade',trade_files+'_exports_'+str(y+1)+'.csv'),delimiter=',').reshape((len(regions),len(trade_zones)))
    return imports,exports

