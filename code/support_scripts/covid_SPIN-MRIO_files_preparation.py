# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:12:47 2020

@author: beaufils
"""


import os
import numpy as np
import pandas as pd

source_path = os.path.join('..','..','..','Corona_projections','Publication','Nature_data','Files')
spin_path = os.path.join('..','..')

def load_ifs(year=2015):
    '''
    Load national data from the International Financial Statistics from IMF.
    Data is stored as xlsx files extracted from the IFS database.
    
    Parameters
    ----------
    year : int
        Data year to be extracted.

    Returns
    -------
    labs : list of ints
        List of countries covered in the ifs dataset.
    list of numpy arrays
        list of exchange rates, GDP real value, imports and exports current value per country.

    '''
    labs = np.sort(load_country_label())
    rates = pd.read_excel(os.path.join(source_path,'Sources','exchange_rates.xlsx'),engine='openpyxl')
    euroz = np.loadtxt(os.path.join(source_path,'Sources','eurozone.csv'),dtype=str)
    euro = rates[rates['ISO']=='EMU'][[str(year)]].to_numpy()[0]
    plabs = pd.DataFrame(labs,columns=['ISO'])
    rates = rates.merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    imports = pd.read_excel(os.path.join(source_path,'Sources','imports_lcu.xlsx'),engine='openpyxl').merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    exports = pd.read_excel(os.path.join(source_path,'Sources','exports_lcu.xlsx'),engine='openpyxl').merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    gdp = pd.read_excel(os.path.join(source_path,'Sources','gdp_lcu.xlsx'),engine='openpyxl').merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    imports = imports[[str(year)]].to_numpy()[:,0]
    exports = exports[[str(year)]].to_numpy()[:,0]
    gdp = gdp[[str(year)]].to_numpy()[:,0]
    rates = rates[[str(year)]].to_numpy()[:,0]
    for i,c in enumerate(labs):
        if c in euroz:
            #Countries in eurozone are treated separatly since they have a common exchange rate
            rates[i] = euro
    return labs,[rates,gdp,imports,exports]

def load_weo(edition,years,period='Oct'):
    '''
    Load data from a World Economic Outlook scenario from the IMF

    Parameters
    ----------
    edition : int
        Edition year to load.
    years : list-like of ints
        Forecasted years to be loaded.
    period : string, optional
        Month of the wEO to load.
        WEO dataset are usually released twice a year, in Oct ('Oct') and April ('Apr').
        The default is 'Oct'.
        
    Returns
    -------
    list of 2D numpy arrays
        GDP, imports and exports volume per country and per year.
        (Annual volume rate)

    '''
    labs = np.sort(load_country_label())
    plabs = pd.DataFrame(labs,columns=['ISO'])
    weo = pd.read_excel(os.path.join(source_path,'Sources','WEO_'+period+'_'+str(edition)+'.xlsx'),engine='openpyxl',thousands=r',')
    gdp = weo[weo['WEO Subject Code']=='NGDP_RPCH'][['ISO']+years].merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    imports = weo[weo['WEO Subject Code']=='TM_RPCH'][['ISO']+years].merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    exports = weo[weo['WEO Subject Code']=='TX_RPCH'][['ISO']+years].merge(plabs,how='right',on='ISO').sort_values(by='ISO')
    gdp = gdp[years].to_numpy(dtype=float)/100+1
    imports = imports[years].to_numpy(dtype=float)/100+1
    exports = exports[years].to_numpy(dtype=float)/100+1
    return [gdp,imports,exports]

def prepare_data(base_table=2015,projection_end=2026,scenario_year=2021,counterfactual_year=2019,end_counterfactual_period=2024):
    '''
    Generate GDP and trade files for a projection based on 2 IMF WEO scenarios
    Scenarios have two periods:
        hist is covered using the main scenario before the begin of the counterfactual WEO
        projection period runs from the begin of the counterfactual to the end of the projection scenario
    Data for the base year are extracted from the IFS database.
    Imports are adjusted yearly to exports values.
    Countries not covered are excluded.
    Projections years not covered in the counterfactual scenario are extrapolated
    using the most recent growth rates available

    Parameters
    ----------
    base_table : int, optional
        Year of the base MRIO table. The default is 2015.
    projection_end : int, optional
        End of the projection period. The default is 2026.
    scenario_year : int, optional
        Year of the projection database. The default is 2021.
    counterfactual_year : int, optional
        Year of the counterfactual database. The default is 2019.
        Historical period runs until the beginning of the counterfactual database,
        i.e. data prior to this year are common in both scenarios
    end_counterfactual_period : int, optional
        Final year of the counterfactual database. The default is 2024.
        Posterious years are extrapolated using the most recent rates,
        until the end of the projection period.

    Returns
    -------
    None.
    GDP and trade files are stored in the Data repositery used by the SPIN algorithm

    '''
    blabs = load_country_label()
    labs, ifs = load_ifs(base_table)
    historical_period = np.arange(base_table,counterfactual_year)
    counterfactual_period = np.arange(counterfactual_year,end_counterfactual_period)
    projection_period = np.arange(counterfactual_year,projection_end)
    hist = load_weo(scenario_year,[i+1 for i in historical_period],'Apr')
    counterfact = load_weo(counterfactual_year,[i+1 for i in counterfactual_period])
    projection = load_weo(scenario_year,[i+1 for i in projection_period],'Apr')
    
    #Load and format data from the base year from International Financial Statistics (IFS)
    benchmark = ifs.copy()
    benchmark.extend([i[:,0] for i in hist])
    benchmark.extend([i[:,-1] for i in projection])
    benchmark = np.transpose(benchmark)
    benchmark[benchmark==0] = np.nan
    excl=[]
    for i,c in enumerate(benchmark):
        if any(np.isnan(c)):
            excl.append(i)
    
    #Exclude countries not covered in IFS
    labs = np.delete(labs,excl)
    for i in range(4):
        ifs[i] = np.delete(ifs[i],excl)
        if i!=3:
            hist[i] = np.delete(hist[i],excl,axis=0)
            counterfact[i] = np.delete(counterfact[i],excl,axis=0)
            projection[i] = np.delete(projection[i],excl,axis=0)
    
    #Create historical GDP and trade time series
    gdp_hist,imp_hist,exp_hist = np.zeros((len(labs),len(historical_period))),np.zeros((len(labs),len(historical_period))),np.zeros((len(labs),len(historical_period)))
    gdp0,imp0,exp0 = 1000*ifs[1]/ifs[0],1000*ifs[2]/ifs[0],1000*ifs[3]/ifs[0]   #Convert IFS data in kUS$
    for i in range(len(historical_period)):
        gdp0,imp0,exp0 = gdp0*hist[0][:,i],imp0*hist[1][:,i],exp0*hist[2][:,i]
        print('Difference between exports and imports in '+str(i+base_table+1)+': '+str(np.sum(exp0)/np.sum(imp0)-1))
        imp0 = imp0*np.sum(exp0)/np.sum(imp0)
        gdp_hist[:,i],imp_hist[:,i],exp_hist[:,i] = gdp0,imp0,exp0
    
    gdp1,imp1,exp1 = np.copy(gdp0),np.copy(imp0),np.copy(exp0)
    #Extrapolate missing years in the counterfactual scenario
    for i in range(3):
        for j in range(projection_end-end_counterfactual_period):
            counterfact[i] = np.concatenate((counterfact[i],counterfact[i][:,-1].reshape((len(labs),1))),axis=1)
    
    #Build counterfactual time series
    gdp19,imp19,exp19 = np.zeros((len(labs),len(projection_period))),np.zeros((len(labs),len(projection_period))),np.zeros((len(labs),len(projection_period)))
    for i in range(len(projection_period)):
        gdp0,imp0,exp0 = gdp0*counterfact[0][:,i],imp0*counterfact[1][:,i],exp0*counterfact[2][:,i]
        print('Difference between exports and imports in '+str(i+counterfactual_year+1)+': '+str(np.sum(exp0)/np.sum(imp0)-1))
        imp0 = imp0*np.sum(exp0)/np.sum(imp0)
        gdp19[:,i],imp19[:,i],exp19[:,i] = gdp0,imp0,exp0
       
    #Build projection time series
    gdp20,imp20,exp20 = np.zeros((len(labs),len(projection_period))),np.zeros((len(labs),len(projection_period))),np.zeros((len(labs),len(projection_period)))
    gdp0,imp0,exp0 = np.copy(gdp1),np.copy(imp1),np.copy(exp1)
    for i in range(len(projection_period)):
        gdp0,imp0,exp0 = gdp0*projection[0][:,i],imp0*projection[1][:,i],exp0*projection[2][:,i]
        print('Difference between exports and imports in '+str(i+counterfactual_year+1)+': '+str(np.sum(exp0)/np.sum(imp0)-1))
        imp0 = imp0*np.sum(exp0)/np.sum(imp0)
        gdp20[:,i],imp20[:,i],exp20[:,i] = gdp0,imp0,exp0
    
    #Assert time series consistence
    for a in [gdp_hist,exp_hist,imp_hist,gdp_hist-exp_hist+imp_hist,gdp19,exp19,imp19,gdp19-exp19+imp19,gdp20,imp20,exp20,exp20-exp20+imp20]:
        assert np.sum(a<0)==0
        assert np.sum(np.isnan(a))==0
    
    #Assignate time series to scenarios: historical, counterfactual and projection
    clabs = np.copy(labs)
    gdph,imph,exph = np.copy(gdp_hist),np.copy(imp_hist),np.copy(exp_hist)
    gdp9,imp9,exp9 = np.copy(gdp19),np.copy(imp19),np.copy(exp19)
    gdp0,imp0,exp0 = np.copy(gdp20),np.copy(imp20),np.copy(exp20)
    
    excl = []
    k=0
    for i,c in enumerate(blabs):
        if c in labs:
            l = labs.tolist().index(c)
            clabs[k] = labs[l]
            gdph[k],imph[k],exph[k] = gdp_hist[l],imp_hist[l],exp_hist[l]
            gdp9[k],imp9[k],exp9[k] = gdp19[l],imp19[l],exp19[l]
            gdp0[k],imp0[k],exp0[k] = gdp20[l],imp20[l],exp20[l]
            k+=1
        else:
            excl.append([i,c])
    
    np.savetxt(os.path.join(spin_path,'output','covid_SPIN-MRIO_label_countries.csv'),clabs,delimiter=',',fmt='%s')
    np.savetxt(os.path.join(spin_path,'data','GDP','covid_SPIN-MRIO_hist.csv'),gdph,delimiter=',')
    np.savetxt(os.path.join(spin_path,'data','GDP','covid_SPIN-MRIO_counterfactual.csv'),gdp9,delimiter=',')
    np.savetxt(os.path.join(spin_path,'data','GDP','covid_SPIN-MRIO_baseline.csv'),gdp0,delimiter=',')
    for i,y in enumerate(historical_period):
        np.savetxt(os.path.join(spin_path,'data','trade','covid_SPIN-MRIO_hist_exports_'+str(y+1)+'.csv'),exph[:,i],delimiter=',')
        np.savetxt(os.path.join(spin_path,'data','trade','covid_SPIN-MRIO_hist_imports_'+str(y+1)+'.csv'),imph[:,i],delimiter=',')
    for i,y in enumerate(projection_period):
        np.savetxt(os.path.join(spin_path,'data','trade','covid_SPIN-MRIO_counterfactual_exports_'+str(y+1)+'.csv'),exp9[:,i],delimiter=',')
        np.savetxt(os.path.join(spin_path,'data','trade','covid_SPIN-MRIO_counterfactual_imports_'+str(y+1)+'.csv'),imp9[:,i],delimiter=',')
        np.savetxt(os.path.join(spin_path,'data','trade','covid_SPIN-MRIO_baseline_exports_'+str(y+1)+'.csv'),exp0[:,i],delimiter=',')
        np.savetxt(os.path.join(spin_path,'data','trade','covid_SPIN-MRIO_baseline_imports_'+str(y+1)+'.csv'),imp0[:,i],delimiter=',')
    
    format_base_table(base_table,clabs)

def load_label():
    lab = pd.read_csv(os.path.join(source_path,'Sources','labels_T.txt'),sep='\t',header=None).to_numpy()
    extract = lab[:4914,[1,3]]
    return extract

def load_country_label():
    lab = pd.read_csv(os.path.join(source_path,'Sources','labels_T.txt'),sep='\t',header=None).to_numpy()
    clab=[]
    for i in range(189):
        clab.append(lab[i*26,1])
    return clab

def load_base_EORA(year,countries=189,sectors=26):
    t = np.loadtxt(os.path.join(source_path,'Sources','Eora26_'+str(year)+'_bp_T.txt'),delimiter = '\t')
    y = np.loadtxt(os.path.join(source_path,'Sources','Eora26_'+str(year)+'_bp_FD.txt'),delimiter = '\t')
    v = np.loadtxt(os.path.join(source_path,'Sources','Eora26_'+str(year)+'_bp_VA.txt'),delimiter = '\t')
    return t[:26*countries,:26*countries],y[:26*countries,:],v[:,:26*countries]


def format_base_table(year,country_labels,countries=189,sectors=26,original_va=True,balance_empty_cols=True,balance_empty_rows=True):
    '''
    Saves a basic version of the EORA tables,
    where net changes in stocks and inventories are neglected

    Parameters
    ----------
    year : int
        Year to treat.
    country_labels:
        List of ISO A3 names of countries covered by the ISF dataset
    countries : int, optional
        Number of countries in the initial table. The default is 189.
    sectors : int, optional
        Number of sectors per country. The default is 26.
    exclusion : list of ints, optional
        Countries to exclude from EORA.
        The default is [174],corresponding to USSR.
    original_va : bool, optional. The default is True
        Whether to use value added from the base eora table.
        If False, value added is derived so that the base table is balanced.

    Returns
    -------
    None.

    '''
    t,y,v = load_base_EORA(year)
    
     #Associate formatted labels with EORA source labels
    print('Associating indexes')
    countries = len(country_labels)
    label = load_label()
    elab=[]
    for i in range(189):
        elab.append(label[i*26,0])
    
    exclusion = [i for i,c in enumerate(elab) if c not in country_labels]
   
    
    #Suppress non included countries
    print('Deleting exclusions')
    fulllist = []
    ylist = []
    for i in exclusion:
        fulllist += [j for j in range(i*26,(i+1)*26)]
        ylist += [j for j in range(i*6,(i+1)*6)]
    t = np.delete(t,fulllist,0)
    t = np.delete(t,fulllist,1)
    y = np.delete(y,fulllist,0)
    y = np.delete(y,ylist,1)
    v = np.delete(v,fulllist,1)
    clab = np.delete(elab,exclusion)
    label = np.delete(label,fulllist,0)
    
    print('Formatting value added and final demand')
    y[y<0] = 0
    t[t<0] = 0
    y1 = np.zeros((countries*sectors,countries))
    
    for c in range(countries):
       y1[:,c] = np.sum(y[:,c*6:(c+1)*6],axis=1)
    x = np.sum(t,axis=1) + np.sum(y1,axis=1)
    if original_va:
        v = np.sum(v,axis=0)
    else:
        v = x - np.sum(t,axis=0)              
    
    #If input of a sector is null and output non null,
    #Fill input to avoid consistency issues
    if balance_empty_cols:
        xo = np.sum(t,axis=1) + np.sum(y,axis=1)
        xi = np.sum(t,axis=0) + v
        for i,a in enumerate(xo):
            if a!=0 and xi[i] ==0:
                t[:,i],v[i] = t[i,:],np.sum(y[i,:])
                print('Empty inputs at index '+ str(i))
            if a==0 and xi[i]==0:
                t[i,i],y1[i,i//26],v[i] = 1,1,1
                print('Empty sector at index '+str(i))
    #If output of a sector is null and input non null,
    #Fill output to avoid consistency issues
    if balance_empty_rows:
        for i,a in enumerate(xi):
            if a!=0 and xo[i] ==0:
                t[i,:],y1[i,i//26] = t[:,i],v[i]
                print('Empty outputs at index '+ str(i))
    
    print('Saving the balanced tables')
    np.savetxt(os.path.join(spin_path,'data','MRIOT','covid_SPIN-MRIO_base_'+str(year)+'_T.csv'),t,delimiter=',')
    np.savetxt(os.path.join(spin_path,'data','MRIOT','covid_SPIN-MRIO_base_'+str(year)+'_FD.csv'),y1,delimiter=',')
    np.savetxt(os.path.join(spin_path,'data','MRIOT','covid_SPIN-MRIO_base_'+str(year)+'_VA.csv'),v,delimiter=',')
    np.savetxt(os.path.join(spin_path,'output','covid_SPIN-MRIO_country_labels.csv'),clab,delimiter=',',fmt='%s')
    np.savetxt(os.path.join(spin_path,'output','covid_SPIN-MRIO_full_labels.csv'),label,delimiter=',',fmt='%s')
    
prepare_data()