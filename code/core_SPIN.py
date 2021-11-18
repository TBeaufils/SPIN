# -*- coding: utf-8 -*-
"""
Last updated on Mon Dec 07 2020
    
Back-end implementation of the SPIN method

@author: Timothé Beaufils
mail: timothe.beaufils@pik-potsdam.de
"""

import time
import datetime
import numpy as np
import GRAS as ras
  
##### Main
def spin_projection(t,y,v,imports,exports,gdp,trade_zones,gdp_zones,regions,eps=0.0001,it=1000,countries=189,sectors=26,dual_mode=False):
    """
    Main function. Runs the two steps of the SPIN method

    Parameters
    ----------
    t : 2D numpy array
        Original MRIO table.
    y : 1D numpy array
        Original final demand array.
    v : 1D numpy array
        Original value added array.
    trade : 2D numpy array
        Matrix of bilateral trade flows between trade zones.
    imports : 2D numpy array
        Matrix of imports from trade zones to regions.
    exports : 2D numpy array
        Matrix of exports from regions to trade zones.
    growth : 1D numpy Array
        Growth array by GDP zone.
    trade_zones : list of list of ints
        Composition of trade zones.
    gdp_zones : list of list of ints
        Composition of common growth zones.
    regions : list of list of ints
        Composition of economic zones for the projection step.
    method : String, optional
        Balancing method. The default is 'MRAS'.
        'TRAS' for an alternative
    eps : float, optional
        Convergence threshold. The default is 0.0001.
    it : int, optional
        Max number of iterations. The default is 1000.
    countries : int, optional
        Number of countries. The default is 189.
    sectors : int, optional
        Number of sectors. The default is 26.
    dual_mode : bool, optional
        If True, trade is constrained for each region
        with regard to intra zone trade and extra zone trade, only.
        If False, trade is constrained for each region
        with regard to every trade zone.
        The default is False.

    Returns
    -------
    tz : 2D array
        Projected MRIO table.
    y1 : 1D array
        Projected final demand.
    v1 : 1D array
        Projected value added.
    report : list of lists
        Convergence report.

    """
    detailed_regions,detailed_zones,loc_gdp,x1,fd = projection_step(t,y,v,imports,exports,gdp,regions,gdp_zones,trade_zones,countries,sectors,dual_mode)
    return ms_ras(t,y,v,x1,fd,loc_gdp,detailed_regions,detailed_zones,regions,trade_zones,imports,exports,countries,sectors,it,eps,dual_mode)

##### Steps
def projection_step(t,y,v,imports,exports,gdp,regions,gdp_zones,trade_zones,countries,sectors,dual_mode):
    """
    Run the projection step of the SPIN method.
    Sccales final demand and exports using gdp and trade projections.
    For each region, applies Leontief inverse using local technology matrix with projected exports and final demand.

    Parameters
    ----------
    t : 2D matrix
        Original MRIO table.
    y : 2D Array
        Original final demand matrix.
    v : 1D Array
        Original value added array.
    imports : 2D numpy array
        Matrix of imports from trade zones to regions.
    exports : 2D numpy array
        Matrix of exports from regions to trade zones.
    growth : 1D numpy array
        Growth array by GDP zone.
    regions: list of list of ints
        Composition of regions.
    gdp_zones : list of list of ints
        Composition of the common growth zones.
    trade_zones : list of list of ints
        Composition of trade zones.
    countries : int
        Number of countries.
    sectors : int
        Number of sectors.
    

    Returns
    -------
    x1 : 1D Array
        Gross output by country and by sector.
    zy1 : 1D array
        Final demand constraint for each region and growth zone.
    zgdp : 2D matrix
        Value added constraints for each region and growth zone.
    trade : 2D matrix
        International trade matrix.

    """
    print('_____ Projection step _____')
    start = time.time()
    
    x0 = np.sum(t,axis=1) + np.sum(y,axis=1)
    
    #Allocate GDP between subregional GDP zones
    #Create regional arrays for GDP constraints
    detailed_regions,local_gdp = allocate_regional_gdp(gdp,regions,gdp_zones,countries,sectors)
    #Detail regional composition of trade zones.
    detailed_zones = detail_trade_zones(trade_zones,regions)
    
    #Allocate trade within regions and build regionally aggregated demand array
    fd = [None]*len(regions)  #Domestic consumption constraints
    x1 = [None]*len(regions)  #Gross sectors output constraints
    for i,r in enumerate(regions):
        #Verify consistence of input data
        assert np.sum(local_gdp[i]) + np.sum(imports[i]) > np.sum(exports[i]), 'Inconsistent regional constraints'
        #Aggregate trade partners if dual mode is activated
        trade_partners = get_trade_partners(r,trade_zones,dual_mode)
        #Allocate exports within the region (proportionally)
        loc_exp = split_exports(t,y,r,trade_partners,exports[i],sectors)
        #Allocate consumption within the region
        fd[i],exp_dfd = allocate_consumption(t,y,v,r,detailed_regions[i],local_gdp[i],trade_partners,imports[i],exports[i],countries,sectors)
        #Build aggregated demand array
        aggregated_demand = loc_exp+exp_dfd
        assert all(aggregated_demand>=0)    #Verify postivity
        
        table=get_domestic_block(t,r,sectors)    #Build domestic matrix
        #Build Leontief inverse
        x = get_from_array(x0,r,sectors)
        x[x==0] = 1
        leon = np.linalg.inv(np.identity(sectors*len(r))-table.dot(np.diag(1/x)))
        #Determine sectoral output with Leontief inverse
        x1[i] = leon.dot(aggregated_demand) 
        for j,z in enumerate(detailed_regions[i]):
            lindex=[]
            for c in z:
                lindex+=[i for i in range(c*sectors,(c+1)*sectors)]
            assert local_gdp[i][j] < np.sum(x1[i][lindex]), 'GDP rates too heterogeneous within region ' + str(i)
        assert np.sum(x1[i])+np.sum(imports[i])>np.sum(fd[i]) #Check consistence
    end = time.time()
    print('    Projection step completed. Computation time: '+str(datetime.timedelta(seconds=end-start)))
    print()
    return detailed_regions,detailed_zones,local_gdp,x1,fd

def ms_ras(t,y,v,x1,fd,gdp,detailed_regions,detailed_zones,regions,trade_zones,imports,exports,countries,sectors,it,eps,dual_mode):
    """
    Implementation of the Multi-Scale RAS algorithm

    Parameters
    ----------
    t : 2D array
        Original MRIO table.
    y : 2D array
        Original final demand array.
    v : 1D array
        Original value added array.
    x1 : list of lists of floats
        Gross output by region and by sector.
    fd : list of lists of floats
        Final demand constraints for each region.
    gdp : list of lists of floats
        Value added constraints for each region.
    detailed_regions: list of list of ints
        Subregional gdp zones for each region.
    detailed_regions: list of list of ints
        Regional composition of trade zones.
    regions : list of list of ints
        Composition of regions
    imports : 2D numpy array
        Matrix of imports from trade zones to regions.
    exports : 2D numpy array
        Matrix of exports from regions to trade zones.
    countries : int
        Number of countries.
    sectors : int
        Number of sectors.
    it : int, optional
        Max number of iterations. The default is 5000.
    eps : float, optional
        Convergence threshold. The default is 0.0001.

    Returns
    -------
    tz : 2D array
        Projected MRIO table.
    y1 : 1D array
        Projected final demand.
    v1 : 1D array
        Projected value added.
    report : list of lists
        Convergence report.

    """
    start_all = time.time()
    print('_____ Balancing step _____')
    exarrays = [None]*len(regions)
    imarrays = [None]*len(regions)
    t1 = np.zeros([countries*sectors,countries*sectors])
    y1 = np.zeros((countries*sectors,countries))
    v1 = np.zeros(countries*sectors)
    report = [['From Region'],['To Region'],['Relative error in constraints'],['Iterations'],['Relative Error']]
    
    #Sub-step 1: intra-zone optimisation
    print('    Optmizing regional matrices')
    for i,r in enumerate(regions):
        report[0].append(i)
        report[1].append(i)
        trade_partners = get_trade_partners(r, trade_zones,dual_mode)
        tz = aggregate_domestic_table(t,y,v,r,trade_partners,detailed_regions[i],sectors)
        #Constraint preparation
        rowsum = np.zeros(len(r)*(sectors)+len(trade_partners)+len(detailed_regions[i]))
        colsum = np.zeros(len(r)*(sectors+1)+len(trade_partners))
        rowsum[:len(r)*sectors],colsum[:len(r)*sectors] = x1[i],x1[i]
        rowsum,colsum = x1[i],x1[i]
        for j in range(len(trade_partners)):
            rowsum=np.append(rowsum,imports[i,j])
            colsum = np.append(colsum,exports[i,j])
        rowsum = np.append(rowsum,gdp[i])
        colsum = np.append(colsum,fd[i])
        report[2].append(np.sum(rowsum)/np.sum(colsum)-1)
        assert abs(np.sum(rowsum)/np.sum(colsum)-1)<eps
        assert all(colsum>=0)
        #RAS algorithm
        balanced,iterations,error = ras.gras_method(tz,rowsum,colsum,eps=eps,it=it)
        report[3].append(iterations)
        report[4].append(error)
        #Disaggregation of the balanced table
        t1,y1,v1,exarrays[i],imarrays[i] = disaggregate_domestic_table(t1,y1,v1,balanced,r,len(trade_partners),len(detailed_regions[i]),sectors)
    end_1 = time.time()
    print('    Economic zones optimized in '+str(datetime.timedelta(seconds=end_1-start_all)))
    
    #Sub-step 2: international trade blocks optimisation
    print('    Balancing international trade')
    start2 = time.time()
    if dual_mode:
        world_block=[]
        world_exp = np.array([])
        world_imp = np.array([])
    for id_zone0,zone0 in enumerate(detailed_zones):
        if dual_mode:
            world_block.append([])
            for c0 in zone0:
                world_exp = np.append(world_exp,exarrays[c0][:,1])
                world_imp = np.append(world_imp,imarrays[c0][1,:])
        for id_zone1,zone1 in enumerate(detailed_zones):
            if dual_mode:
                if zone1 == zone0:
                    world_block[-1].append(np.zeros((len(zone0)*sectors,len(zone0)*(sectors+1))))
                else:
                    world_block[-1].append(aggregate_tradeblock(t,y,zone0,zone1,regions,sectors))
            if not dual_mode or zone1==zone0:
                if dual_mode:
                    id_zone0 = 0
                    id_zone1 = 0
                report[0].append(id_zone0)
                report[1].append(id_zone1)
                tradeblock = aggregate_tradeblock(t,y,zone0,zone1,regions,sectors)
                ex = np.array([])
                im = np.array([])
                for c0 in zone0:
                    ex = np.append(ex,exarrays[c0][:,id_zone1])
                for c1 in zone1:
                    im = np.append(im,imarrays[c1][id_zone0,:])
                if np.sum(ex)==0:
                    report[2].append(0)
                else:
                    report[2].append(np.sum(im)/np.sum(ex)-1)
                    assert abs(np.sum(im)/np.sum(ex)-1)<eps
                #Note: RAS on transposed trade block to align convergence on column-wise constraints
                balanced,iterations,error = ras.ras_method(np.transpose(tradeblock),im,ex,eps=eps,it=it)
                report[3].append(iterations)
                report[4].append(error)
                t1,y1 = disaggregate_tradeblock(t1,y1,np.transpose(balanced),zone0,zone1,regions,sectors)
    if dual_mode:
        world_block = np.block(world_block)
        report[0].append('RoW')
        report[1].append('RoW')
        if np.sum(world_exp)==0:
            report[2].append(0)
        else:
            report[2].append(np.sum(world_imp)/np.sum(world_exp)-1)
            assert abs(np.sum(world_imp)/np.sum(world_exp)-1)<eps
        balanced,iterations,error = ras.ras_method(np.transpose(world_block),world_imp,world_exp,eps=eps,it=it)
        report[3].append(iterations)
        report[4].append(error)
        t1,y1 = disaggregate_world_block(t1,y1,np.transpose(balanced),detailed_zones,regions,sectors)
    end_2 = time.time()
    print('    International trade matrices optimized in '+str(datetime.timedelta(seconds=end_2-start2)))
    print('    Overall balancing time: '+str(datetime.timedelta(seconds=end_2-start_all)))
    print()
    return t1,y1,v1,report

##### Getters
# Standard function to extract elements from the base MRIO table

def get_domestic_block(t,region,sectors):
    '''
    Extract the domestic block from the base inter-industry matrix

    Parameters
    ----------
    t : 2D array
        Inter-industry matrix.
    region : list of ints
        Composition of the region.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    table : 2D array
        Domestic inter-industry block.

    '''
    table = np.zeros((len(region)*sectors,len(region)*sectors))
    for i,c in enumerate(region):
        for j,p in enumerate(region):
            table[i*sectors:(i+1)*sectors,j*sectors:(j+1)*sectors]=t[c*sectors:(c+1)*sectors,p*sectors:(p+1)*sectors]
    return table

def get_exports(t,y,origin,destination,sectors,excl = [None],block=False):
    '''
    Basic function to extract exports from a region to another

    Parameters
    ----------
    t : 2D array
        Inter-industry table.
    y : 2D array
        Final demand matrix.
    origin : int or list of ints
        Exporting entities.
    destination : int or list of ints
        Composition of the destination zone.
    sectors : int
        Number of sectors per country.
    excl : list of ints, optional
        List of countries not to be considered as trade partners. 
    block : bool, optional
        If True, the full export block is returned.
        If False, exports are aggregated with regard to the destination zone

    Returns
    -------
    exp : 1D - 2D array
        Array or matrix of exports.

    '''
    if isinstance(origin,int):
        origin = [origin]
    if isinstance(destination,int):
        destination = [destination]
    if isinstance(excl,int):
        excl=[excl]
    exp = np.zeros((len(origin)*sectors,len(destination)*(sectors+1)))
    for i,c in enumerate(origin):
        for j,p in enumerate(destination):
            if p not in excl:
                exp[i*sectors:(i+1)*sectors,j*(sectors+1):(j+1)*(sectors+1)-1] = t[c*sectors:(c+1)*sectors,p*sectors:(p+1)*sectors]
                exp[i*sectors:(i+1)*sectors,(j+1)*(sectors+1)-1] = y[c*sectors:(c+1)*sectors,p]
    if not block:
        exp = np.sum(exp,axis=1)
    return exp

def get_imports(t,y,destination,origin,sectors,excl=[None],block=False):
    '''
    Generic function to return intermediate imports from a block to another

    Parameters
    ----------
    t : 2D array
        Inter-industry table.
    y : 2D array
        Final demand matrix.
    destination : int or list of ints
        Composition of the destination zone.
    origin : int or list of ints
        Exporting entities.
    sectors : int
        Number of sectors per country.
    excl : list of ints, optional
        List of countries not to be considered as trade partners. 
        The exporting country is ignored by default.
    block : bool, optional
        If True, the full export block is returned.
        If False, exports are aggregated with regard to the destination zone

    Returns
    -------
    exp : 1D - 2D array
        Array or matrix of exports.

    '''
    if isinstance(origin,int):
        origin = [origin]
    if isinstance(destination,int):
        destination = [destination]
    if isinstance(excl,int):
        excl=[excl]
    inter_imp = np.zeros((len(origin)*sectors,len(destination)*sectors))
    final_imp = np.zeros((len(origin)*sectors,len(destination)))
    for j,p in enumerate(destination):
        for i,c in enumerate(origin):
            if c not in excl:
                inter_imp[i*sectors:(i+1)*sectors,j*sectors:(j+1)*sectors] = t[c*sectors:(c+1)*sectors,p*sectors:(p+1)*sectors]
                final_imp[i*sectors:(i+1)*sectors,j] = y[c*sectors:(c+1)*sectors,p]
    if not block:
        inter_imp = np.sum(inter_imp,axis=0)
        final_imp = np.sum(final_imp,axis=0)
    return inter_imp,final_imp

def get_from_array(array,region,sectors):
    '''
    Extract the array associated with a set of countries from a global array.

    Parameters
    ----------
    array : 1D array
        Gobal array.
    region : list of ints
        Composition of the region to extract.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    extract: 1D array
        Values related to the selection.

    '''
    if isinstance(region,int):
        region = [region]
    extract = np.zeros(len(region)*sectors)
    for i,c in enumerate(region):
        extract[i*sectors:(i+1)*sectors] = array[c*sectors:(c+1)*sectors]
    return extract

def get_dfd(y,region,sectors):
    '''
    Extract the shares of domestic final demand from the final demand matrix

    Parameters
    ----------
    y : 2D array
        Final demand array.
    region : list of ints
        Composition of the domestic region.
    sectors : int
        number of sectors per country.

    Returns
    -------
    dfd: 2D array
        Shares of domestic final demand per sector.

    '''
    dfd = np.zeros((len(region)*sectors,len(region)))
    for i,c in enumerate(region):
        for j,p in enumerate(region):
            dfd[i*sectors:(i+1)*sectors,j]=y[c*sectors:(c+1)*sectors,p]
    return dfd

def get_trade_partners(region,trade_zones,dual = False):
    '''
    Returns the list of possible trade partners for a region,
    distinguishing between intra-zone trade and extra-zone trade only.
    If dual mode is not active, simply returns the list of trade zones
    
    Parameters
    ----------
    region : list of ints
        Composition of the region.
    trade_zones : list of list of ints
        Composition of trade zones.
    dual : boolean, optional
        Discriminate between intra-zone and extra-zone trade. 
        The default is False.

    Returns
    -------
    trade_partners : list of list of ints
        Composition of possible trade partners.

    '''
    if dual:
        trade_partners =[[],[]]
        for i in trade_zones:
            if region[0] in i:
                trade_partners[0] += i
            if region[0] not in i:
                trade_partners[1] += i
    else:
        trade_partners=trade_zones
    return trade_partners

def get_alpha(t,y,region,trade_partners,sectors):
    '''
    Compute the share of final commodities in imports to a region
    and the maximal share of final commodities to ensure a positive
    consumption of commodities domestically produced (FD>Final_imports)

    Parameters
    ----------
    t : 2D array
        Inter-industry matrix.
    y : 2D array
        final demand matrix.
    region : list of ints
        Composition of the destination region.
    trade_partners : list of list of ints
        Composition of the trade zones.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    alpha : float
        Shares of final commodities in imports from partners to region.
    lim_alpha : 
        Share of final imports that would exceed final demand.

    '''
    imports = 0
    final_imports = 0
    exports = 0
    for partner in trade_partners:
        add_inter,add_final = get_imports(t,y,region,partner,sectors,region)
        imports += np.sum(add_inter) + np.sum(add_final)
        final_imports += np.sum(add_final)
        exports += np.sum(get_exports(t,y,region,partner,sectors,region))
    alpha = final_imports/imports
    lim_alpha = np.sum(y[:,region])/imports
    assert alpha > 0 and alpha < 1 and lim_alpha > 0 and alpha < lim_alpha
    return alpha,lim_alpha

#### Splitters & allocators

#Function to allocate gdp, exports, imports and domestic final demand
#between countries within a region

def split_gdp(v,zone,gdp,sectors):
    '''
    Allocate GDP between the countries of a GDP zone,
    at the pro-rata to their contribution to current GDP

    Parameters
    ----------
    v : 1D array
        Value added array.
    zone : list of ints
        Composition of the current GDP zone.
    gdp : float
        Total GDP of the zone.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    1D array
        GDP per country.

    '''
    gdp0 = np.zeros(len(zone))
    for i,c in enumerate(zone):
        gdp0[i] = np.sum(v[c*sectors:(c+1)*sectors])
    s_gdp = gdp0/np.sum(gdp0)
    assert np.isclose(np.sum(s_gdp*gdp),gdp)
    return s_gdp*gdp

def split_exports(t,y,zone,trade_zones,exports,sectors):
    '''
    Allocate exports between countries of a region
    proportionally to their contribution at the previous time step

    Parameters
    ----------
    t : 2D array
        Inter-industry MRIO table.
    y : 2D array
        Final demand matrix.
    zone : list of ints
        Composition of the region.
    trade_zones : list of list of ints
        Composition of all trade zones.
    exports : 1D array
        Exports from the region to other trade zones.
    sectors : int
        Sectors per country.

    Returns
    -------
    1D array
        Allocation of exports per sector for each country in the region.

    '''
    current_exp = np.zeros((len(zone)*sectors,len(trade_zones)))
    for i,c in enumerate(zone):
        for j,p in enumerate(trade_zones):
            #Total exports are computed for each country
            #With regard to every trade zone
            #Excluding within the region
            current_exp[i*sectors:(i+1)*sectors,j] = get_exports(t,y,c,p,sectors,excl=zone)
    exp_by_zone=np.sum(current_exp,axis=0)
    exp_by_zone[exp_by_zone==0] = 1
    new_exp = current_exp.dot(np.diag(exports/exp_by_zone))
    assert all(np.isclose(np.sum(new_exp,axis=0),exports))
    return np.sum(new_exp,axis=1)

def split_imports(t,y,region,trade_zones,imports,sectors,locked_imp):
    '''
    Allocate imports between countries of the same region.
    Ensures that IMP + GDP > EXP for each country.
    Ensures that GDP - EXP > IMP for each region

    Parameters
    ----------
    t : 2D array
        Inter-industry matrix.
    y : 2D array
        Final demand matrix.
    region : list of ints
        Composition of the region.
    trade_zones : list of list of ints
        Composition of the international trade zones.
    imports :  1D array
        Regional imports constraints.
    sectors : int
        Number of sectors per country.
    locked_imp : 1D array
        Imports already allocated per country to ensure inputs>exports.
    
    Returns
    -------
    2D array
        Allocation of imports per partner for each country in the region.

    '''
    current_imp = np.zeros((len(trade_zones),len(region)))
    for i,c in enumerate(region):
        for j,p in enumerate(trade_zones):
            inter_imp,final_imp = get_imports(t,y,c,p,sectors,excl=region)
            current_imp[j,i] = np.sum(inter_imp) + np.sum(final_imp)
    imp_per_partner = np.sum(current_imp,axis=1)
    imp_per_partner[imp_per_partner==0] = 1
    cur_shares_imp = np.diag(1/imp_per_partner).dot(current_imp)
    shares_imp_by_partner = (imports/np.sum(imports)).reshape((1,len(trade_zones)))
    imp_per_country = np.sum(current_imp,axis=0)
    imp_per_country[imp_per_country==0] = 1
    shares_imp_per_country = current_imp.dot(np.diag(1/imp_per_country))
    min_imp = shares_imp_per_country.dot(np.diag(locked_imp))
    rest_imp_by_partner = (shares_imp_by_partner*(np.sum(imports)-np.sum(locked_imp)))
    new_imp = rest_imp_by_partner.dot(cur_shares_imp) + np.sum(min_imp,axis=0)
    assert np.isclose(np.sum(new_imp),np.sum(imports))
    return np.reshape(new_imp,(len(region)))

def allocate_consumption(t,y,v,region,detailed_region,gdp,trade_partners,imports,exports,countries,sectors):
    '''
    Split consumption between countries of a region.

    Parameters
    ----------
    t : 2D array
        Inter-industry table.
    y : 2D array
        Final demand matrix.
    v : 1D array
        Value added array.
    region : list of ints
        Composition of the region.
    detailed_region : list of list of ints
        Composition of the gdp zones within the region
    gdp : list of floats
        Regional GDP.
    trade_partners : list of list of ints
        Available trade partners.
    imports : 1D array
        Regional imports from the partners.
    exports : 1D array
        Regional exports to the partners.
    countries : int
        Number of countries.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    dfd : 1D array
        Domestic consumption of regionally produced commodities.
    loc_fd : 1D array
        Final demand constraint per country

    '''  
    loc_fd = np.sum(y[:,region],axis=0) #Regional consumption array
    reg_fd = np.sum(gdp)-np.sum(exports) + np.sum(imports)
    
    #Determine regional repartition between final and intermediate imports
    alpha,alpha_lim_0 = get_alpha(t,y,region,trade_partners,sectors)
    alpha_lim = reg_fd/np.sum(imports)
    alpha = alpha*(min(alpha_lim,1)/min(alpha_lim_0,1))
    
    reg_dfd = reg_fd - alpha*np.sum(imports)
    dfd = get_dfd(y,region,sectors)
    for i,z in enumerate(detailed_region):
        gdp0 = 0
        for c in z:
            gdp0 += np.sum(v[region[c]*sectors:(region[c]+1)*sectors])
        dfd[:,z] = dfd[:,z]*gdp[i]/gdp0
        loc_fd[z] = loc_fd[z]*gdp[i]/gdp0
    loc_dfd = reg_dfd*np.sum(dfd,axis=1)/np.sum(dfd)
    loc_fd = reg_fd*loc_fd/np.sum(loc_fd)
    
    assert np.isclose(np.sum(loc_fd),reg_fd)
    assert np.isclose(np.sum(loc_dfd),reg_dfd)
    
    return loc_fd,loc_dfd

def allocate_regional_gdp(gdp,regions,gdp_zones,countries,sectors):
    '''
    Extrapolates GDP constraints at subregional scale.
    Creates list of relatives indices

    Parameters
    ----------
    gdp : 1D array
        List of subregional GDP constraints (ordinated as gdp_zones).
    regions : list of list of ints
        Composition of regions.
    gdp_zones : list of list of ints
        Composition of GDP zones.
    countries : int
        Total number of countries.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    detailed_regions : list of list of list of ints
        List of countries (rank 3) in gdp zones (rank 2) in regions (rank 1).
        Countries are numbered in reference to their position in their region.
        Ex: int 2 in list 1 in list 3 
        is country (2) in region (3)
        country (2) belongs to gdp_zone (1) of region (3)
    local_gdp : list of list of list of floats
        List of national GDP constraints (r3) in gdp zones (r2) in regions (r1).
        Same structure as detailed_regions but contains expected GDP 
        instead of regional indice.
    '''
    
    detailed_regions = [None]*len(regions)    #Composition of subregional zones
    local_gdp = [None]*len(regions)  #GDP of subregional zones
    for igz,gz in enumerate(gdp_zones):
        #Loop to find the region the GDP zone belongs to
        found = False
        ir=0
        while not found:   
            if gz[0] in regions[ir]:
                found=True
            else:
                ir +=1
        
        #Create list of relative indices for GDP zones within regions
        loc_indices=[]
        for i,c in enumerate(gz):
            #Save relative position of country in region
            loc_indices.append(regions[ir].index(c))
        if detailed_regions[ir]:
            #Save regional indices of countries of GDP zone 
            detailed_regions[ir].append(loc_indices)
        else:   #If the region has currently no other GDP zone, initialize list
            detailed_regions[ir]=[loc_indices]
        if local_gdp[ir]:   #Save national GDP list by subregion
            local_gdp[ir].append(gdp[igz])
        else:   #If the region has currently no other GDP zone, initialize list
            local_gdp[ir] = [gdp[igz]]
        
    return detailed_regions,local_gdp

def detail_trade_zones(trade_zones,regions):
    '''
    Creates an index of regional composition of trade zones.

    Parameters
    ----------
    trade_zones : list of list of ints
        Composition of trade zones.
    regions : list of list of ints
        Composition of trade zones.

    Returns
    -------
    detailed_zones : list of list of ints
        Regional composition of trade zones.
        Index of lists correspond to trade zone
        Index in nested list correspond to regions.

    '''
    detailed_zones = [None]*len(trade_zones) #Regional composition of trade zones
    #Loop between trade zones
    for i,zone in enumerate(trade_zones):
        #Loop between countries in trade zone
        for j,c in enumerate(zone):
            #Loop betwen regions to find the region the country belongs to
            found = False
            ir=0
            while not found:
                if c in regions[ir]:
                    found = True
                else:
                    ir +=1
            if not detailed_zones[i]:   #Initialize zone composition
                detailed_zones[i] = [ir]
            elif ir not in detailed_zones[i]:
                #Add region to the nested list
                detailed_zones[i].append(ir)
    return detailed_zones

##### Aggregators & Disaggregators

#Functions to create and dislocate regional and international blocks

def aggregate_domestic_table(t,y,v,region,trade_zones,gdp_zones,sectors):
    '''
    Creates a regional block for the first step of the MS RAS procedure

    Parameters
    ----------
    t : 2D array
        Inter-industry matrix.
    y : 2D array
        Final demand matrix.
    v : 1D array
        Value added array.
    region : list of ints
        Composition of the regional block.
    trade_zones : list of list of ints
        Composition of the trade zones.
    gdp_zones : list of list of ints
        Composition of the zones with common GDP constraints in the region.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    table : 2D array
        Donestic table.

    '''
    #Creates an empty block of coherent dimensions
    table = np.zeros((len(region)*sectors+len(trade_zones)+len(gdp_zones),len(region)*(sectors+1)+len(trade_zones)))
    #Insert the domestic inter-industry block
    table[:len(region)*sectors,:len(region)*sectors]=get_domestic_block(t,region,sectors)
    #Insert the domestic final demand block
    table[:len(region)*sectors,-len(region):] = get_dfd(y,region,sectors)
    for i,z in enumerate(trade_zones):  #Insert trade blocks
        #Extract intermediate and final imports
        inter_imp,final_imp = get_imports(t,y,region,z,sectors,excl=region)
        #Insert intermediate imports below the domestic table
        table[len(region)*sectors+i,:len(region*sectors)] = inter_imp
        #Insert final imports below domestic consumption
        table[len(region)*sectors+i,-len(region):] = final_imp
        #Insert exports on the right side of the domestic block
        table[:len(region)*sectors,len(region)*sectors+i] = get_exports(t,y,region,z,sectors,excl=region)
    for i,g in enumerate(gdp_zones):    #Insert gdp arrays
        for c in g:
            table[-len(gdp_zones)+i,c*sectors:(c+1)*sectors] = get_from_array(v,region[c],sectors)
    return table

def disaggregate_domestic_table(t,y,v,table,region,trade_zones,gdp_zones,sectors):
    """
    Reallocate a balanced regional table into the main sections of the MRIO table
    Extract constraints for the international trade balancing

    Parameters
    ----------
    t : 2D array
        Projected inter-industry table.
    y : 2D array
        Projected final demand matrix.
    v : 1D array
        Projected value added array.
    table : 2D array
        Balanced regional table.
    region : list of ints
        Composition of the local region.
    trade_zones : int
        Number of trade zones.
    gdp_zones : int
        Number of subregional gdp_zones.
    sectors : int
        Number of sectors per country.

    Returns
    -------
    t : 2D array
        Updated inter-industry table.
    y : 2D array
        Updated final demand array.
    v : 1D array
        Updated value added array.
    exports : 3D array
        Updated export constraints.
    imports : 3D array
        Updated import constraints.

    """
    inter_imp = table[len(region)*sectors:len(region)*sectors+trade_zones,:len(region)*sectors]
    final_imp = table[len(region)*sectors:len(region)*sectors+trade_zones,-len(region):]
    exports = table[:len(region)*sectors,len(region)*sectors:len(region)*sectors+trade_zones]
    imports = np.zeros((trade_zones,len(region)*(sectors+1)))
    v1 = np.sum(table[-gdp_zones:,:sectors*len(region)],axis=0)
    for i,c in enumerate(region):
        imports[:,i*(sectors+1):(i+1)*(sectors+1)-1] = inter_imp[:,i*sectors:(i+1)*sectors]
        imports[:,(i+1)*(sectors+1)-1] = final_imp[:,i]
        for j,d in enumerate(region):
            t[c*sectors:(c+1)*sectors,d*sectors:(d+1)*sectors] = table[i*sectors:(i+1)*sectors,j*sectors:(j+1)*sectors]
            y[c*sectors:(c+1)*sectors,d] = table[i*sectors:(i+1)*sectors,-len(region)+j]
        v[c*sectors:(c+1)*sectors] = v1[i*sectors:(i+1)*sectors]
    return t,y,v,exports,imports

def aggregate_tradeblock(t,y,origin,destination,regions,sectors):
    """
    Returns the tradeblock between two economic zones
    Intra-regional blocks are left empty

    Parameters
    ----------
    t : 2D array
        Global inter-industry matrix
    y : 2D array
        Final demand matrix
    origin : list of lists of ints
        Regional composition of the exporting trade zone
    destination : List of lists of ints
        Composition of the importing trade zone
    regions : list of list of ints
        Composition of economic regions
    sectors : int
        Number of economic sectors

    Returns
    -------
    t1 : 2D array
        Tradeblock between the two economic zones

    """
    blocks=[]
    for ro in origin:
        blocks.append([])
        for rd in destination:
            blocks[-1].append(get_exports(t,y,regions[ro],regions[rd],sectors,excl=regions[ro],block=True))
    return np.block(blocks)

def disaggregate_tradeblock(t,y,block,origin,destination,regions,sectors):
    """
    Reverse operations from aggregate_tradeblock. 
    Allocate the updated international trade block in the projected MRIO table

    Parameters
    ----------
    t : 2D Array
        Projected inter-industry matrix
    y : 2D array
        projected final demand matrix
    block : 2D Array
        Trade block
    origin : List of ints
        Composition of the exporting zone
    destination : List of ints 
        Composition of the importing zone
    regions : list of list of ints
        Composition of the region
        Intraregional blocks are left untouched
    sectors : int
        Number of  sectors per country

    Returns
    -------
    t : 2D array
        Updated inter-industry matrix
    y : 2D array
        Updated final demand matrix

    """
    row=0
    for ro in origin:
        col = 0
        for rd in destination:
            if ro!=rd:
                for i,co in enumerate(regions[ro]):
                    for j,cd in enumerate(regions[rd]):
                        t[co*sectors:(co+1)*sectors,cd*sectors:(cd+1)*sectors] = block[(row+i)*sectors:(row+i+1)*sectors,(col+j)*(sectors+1):(col+j+1)*(sectors+1)-1]
                        y[co*sectors:(co+1)*sectors,cd] = block[(row+i)*sectors:(row+i+1)*sectors,(col+j+1)*(sectors+1)-1]
            col += len(regions[rd])
        row += len(regions[ro])
    return t,y

def disaggregate_world_block(t,y,block,trade_zones,regions,sectors):
    """
    Reverse operations from aggregate_tradeblock specific for the dual mode. 
    Allocate the updated interzone trade block in the projected MRIO table

    Parameters
    ----------
    t : 2D Array
        Projected inter-industry matrix
    y : 2D array
        projected final demand matrix
    block : 2D Array
        Trade block
    trade_zones : List of list ints
        Composition of the trade zones
    regions : list of list of ints
        Composition of the region
        Intraregional blocks are left untouched
    sectors : int
        Number of  sectors per country

    Returns
    -------
    t : 2D array
        Updated inter-industry matrix
    y : 2D array
        Updated final demand matrix

    """
    row=0
    for i0,z0 in enumerate(trade_zones):
        for ro in z0:
            col =0
            for i1,z1 in enumerate(trade_zones):
                if i0 == i1:
                    col += np.sum([len(regions[r]) for r in z0])
                else:
                    for rd in z1:
                        for i,co in enumerate(regions[ro]):
                            for j,cd in enumerate(regions[rd]):
                                t[co*sectors:(co+1)*sectors,cd*sectors:(cd+1)*sectors] = block[(row+i)*sectors:(row+i+1)*sectors,(col+j)*(sectors+1):(col+j+1)*(sectors+1)-1]
                                y[co*sectors:(co+1)*sectors,cd] = block[(row+i)*sectors:(row+i+1)*sectors,(col+j+1)*(sectors+1)-1]
                        col += len(regions[rd])
            row += len(regions[ro])
    return t,y