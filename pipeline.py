from __future__ import print_function, division
from six.moves import range
import dynesty
import os
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py
import brutus
import sys
import glob
from dl import queryClient as qc
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astroquery.xmatch import XMatch
from astroquery.gaia import Gaia
from astropy.table import Table, join,vstack
from astropy.io import fits

from brutus import filters
from brutus.utils import inv_magnitude
from brutus import fitting
from brutus.los import LOS_clouds_priortransform as ptform
from brutus.los import LOS_clouds_loglike_samples as loglike
from zero_point import zpt
from dynesty import plotting as dyplot


def process_cg_region(region_name, CG_table, tail_ext=0.2, head_ext = 0.1, width=0.3 ,save_df=True):
    """
    Generate the four corners of a polygon region centered on the CG head-tail axis.

    Parameters
    ----------
    region_name : str
        The name of the CG region to process.
    CG_table : pd.DataFrame
        DataFrame containing 'Name', 'ra_deg', 'dec_deg', 'tail_ra_deg', 'tail_dec_deg'.
    tail_ext : float, optional
        Extension beyond the tail in degrees. Default is 0.2 deg.
    head_ext : float, optional
        Extension beyond the head in degrees. Default is 0.1 deg.
    save_df : bool, optional
        Whether to save the polygon as a CSV file. Default is True.

    Returns
    -------
    tuple of np.ndarray
        The four corner coordinates (p1, p2, p3, p4), each as (RA, Dec).
    """

    obj_head = SkyCoord(CG_table[CG_table['Name']==region_name]['ra_deg'].values*u.deg, CG_table[CG_table['Name']==region_name]['dec_deg'].values*u.deg)[0]
    obj_tail = SkyCoord(CG_table[CG_table['Name']==region_name]['tail_ra_deg'].values*u.deg, CG_table[CG_table['Name']==region_name]['tail_dec_deg'].values*u.deg)[0]
    
    folder_name = f"/Users/anniegao/Documents/CG_mapping_files/0-queried_region/{region_name}"
    # os.makedirs(folder_name, exist_ok=True)

    # --- Define region corners ---
    del_dec = obj_tail.dec.value - obj_head.dec.value
    del_ra = (obj_tail.ra.value - obj_head.ra.value)
    u_vec = np.array([del_ra, del_dec]) / np.sqrt(del_ra ** 2 + del_dec ** 2)
    u_perp = np.array([u_vec[1], -u_vec[0]])
    hw = width/2 * u_perp
    p1 = np.array([obj_tail.ra.value, obj_tail.dec.value]) + hw + tail_ext * u_vec
    p2 = np.array([obj_tail.ra.value, obj_tail.dec.value]) - hw + tail_ext * u_vec
    p3 = np.array([obj_head.ra.value, obj_head.dec.value]) - hw - head_ext * u_vec
    p4 = np.array([obj_head.ra.value, obj_head.dec.value]) + hw - head_ext * u_vec
    
    if save_df:
        corner_pd = pd.DataFrame(data=[p1, p2, p3, p4], columns=['ra', 'dec'])
        corner_pd.to_csv(f'{folder_name}_corner.csv', index=False)
        print(f'data frame saved to {folder_name}corner.csv')
    return p1, p2, p3, p4

def query_region(p1, p2, p3, p4, region_name):
    folder_name = f"/Users/anniegao/Documents/CG_mapping_files/1-queried_stars/{region_name}"
    # os.makedirs(folder_name, exist_ok=True)
    four_corner = np.concatenate([p1, p2, p3, p4]).tolist()
    query1 = f"""
        SELECT *
        FROM decaps_dr2.object
        WHERE q3c_poly_query(ra, dec, ARRAY {four_corner} ) """
    query3 = f"""SELECT * 
                FROM twomass.psc
                WHERE q3c_poly_query(ra, dec, ARRAY {four_corner} ) """
    df1 = qc.query(sql=query1,fmt='table') 
    df3 = qc.query(sql=query3,fmt='table')
    gaia_xmatched = XMatch.query(cat1 = df1, cat2='vizier:I/355/gaiadr3', max_distance= 0.5*u.arcsec, colRA1='ra', colDec1='dec')
    df1_pd = df1.to_pandas()
    xmatch_pd = gaia_xmatched.to_pandas()
    xmatch_pd_tomerge = xmatch_pd.iloc[:, 210:]
    xmatch_pd_tomerge['obj_id'] = xmatch_pd['obj_id'].values
    xmatch_pd_tomerge['angDist'] = xmatch_pd['angDist'].values
    merge_xmatch = df1_pd.merge(xmatch_pd_tomerge, how='left', left_on='obj_id', right_on='obj_id')
    merged_pd = merge_xmatch.drop_duplicates(subset='obj_id', keep='first') # because there are 13 stars that are duplicated -- two Gaia sources are within 0.5 arcsec from Decaps2. Only keep the closer one. 
    merged_table = Table.from_pandas(merged_pd) # convert back to astropy table # merged_table.write("/Users/anniegao/Documents/CG_mapping_files/CG31/CG31_Decaps2Gaia_crossmatched.csv",format='csv')
    print('Finished crossmatching with Gaia')
    ## Cross Match with 2MASS
    twomass_xmatched = XMatch.query(cat1 = df1, cat2 = 'vizier:II/246/out', max_distance= 0.5*u.arcsec, colRA1='ra', colDec1='dec')
    twomass_xmatch_pd = twomass_xmatched.to_pandas()
    twomass_xmatch_id = twomass_xmatch_pd.iloc[:, 210:211]
    twomass_xmatch_id['obj_id'] = twomass_xmatch_pd['obj_id'].values
    twomass_xmatch_id['angDist'] = twomass_xmatch_pd['angDist'].values
    merge1 = df1_pd.merge(twomass_xmatch_id, how='left', left_on='obj_id', right_on='obj_id')
    merge2 = merge1.merge(df3.to_pandas().iloc[:, 2:], how='left',left_on='2MASS', right_on = 'designation') # Table.from_pandas(merge2).write("/Users/anniegao/Documents/CG_mapping_files/CG31/CG31_Decaps2TwoMASS_crossmatched.csv",format='csv', overwrite=True)
    print('Finished crossmatching with 2MASS')
    ## Merge three tables
    a = merge2.iloc[:, 209:]
    a['obj_id'] = merge2['obj_id']
    decaps_2mass_gaia = merged_table.to_pandas().merge(a, on = 'obj_id') # merge Decaps + 2MASS with Decaps + Gaia 
    output_path = f'{folder_name}_decaps_2mass_gaia.csv'
    Table.from_pandas(decaps_2mass_gaia).write(output_path,format='csv', overwrite=True)
    print(f"[{region_name}] All catalogs merged and saved to:\n→ {output_path}")
    return decaps_2mass_gaia

def clean_and_fit(pd_merged, base, region_name, cut_val1 = 0.78, cut_val2 = 22): 
    """
    Return data frame, parallax, and parallax error that were used for fitting into Brutus; save the fitted file.
    """
    filename = f'{base}2-star_modeling/output/'
    ## Fit data 
    Ndraws = 100 #draws to save to generate the 2D posterior
    thin = 20 # factor to thin samples by for saving to disk
    filt = filters.decam[1:] + filters.tmass[:] #+ filters.vista[2:] #+filters.gaia[:]
    # zero points
    zp_mist = brutus.utils.load_offsets('/Users/anniegao/Documents/CG_mapping_files/2-star_modeling/offsets_mist_v9.txt',filters=filt)
    # import MIST model grid
    gridfile = '/Users/anniegao/Documents/CG_mapping_files/2-star_modeling/grid_mist_v10.h5'
    (models_mist, labels_mist, lmask_mist) = brutus.utils.load_models(gridfile, filters=filt)
    BF_mist = fitting.BruteForce(models_mist, labels_mist, lmask_mist)
    #load tables for parallax zeropoint correction
    zpt.load_tables()

    dist_select = pd_merged[~((pd_merged['Plx']< 1000/600)|(pd_merged['Plx']>1000/200))]
    merged_table = Table.from_pandas(dist_select)
    Ag, Ar, Ai = 1.272392, 0.876292, 0.678924 #3.384, 2.483, 1.838
    mDwarf1 = merged_table['mean_mag_g']- (Ag/(Ag-Ar))* (merged_table['mean_mag_g'] - merged_table['mean_mag_r']-1.4)
    mDwarf2 = merged_table['mean_mag_r'] - merged_table['mean_mag_i'] - (Ar-Ai)/(Ag-Ar) *(merged_table['mean_mag_g']-merged_table['mean_mag_r']-1.4)
    m_dwarf_table = merged_table[(mDwarf1<cut_val2) & (mDwarf2>cut_val1)]

    flux_decam = np.c_[m_dwarf_table['mean_g'].value, m_dwarf_table['mean_r'].value, m_dwarf_table['mean_i'].value, m_dwarf_table['mean_z'].value, m_dwarf_table['mean_y'].value]
    flux_decam_err = np.c_[m_dwarf_table['err_g'].value, m_dwarf_table['err_r'].value, m_dwarf_table['err_i'].value,m_dwarf_table['err_z'].value, m_dwarf_table['err_y'].value]
    mag_2mass =  np.c_[m_dwarf_table['j_m'].value, m_dwarf_table['h_m'].value, m_dwarf_table['k_m'].value ] #merged_table['Gmag'], merged_table['BPmag'], merged_table['RPmag'],
    magerr_2mass = np.c_[m_dwarf_table['j_msigcom'].value, m_dwarf_table['h_msigcom'].value, m_dwarf_table['k_msigcom'].value]  #merged_table['e_Gmag'], merged_table['e_BPmag'], merged_table['e_RPmag'],
    mag_decam = -2.5*np.log10(flux_decam)
    magerr_decam = 1.086*flux_decam_err/flux_decam
    mag = np.c_[mag_decam[:], mag_2mass[:]]
    mag_err = np.c_[magerr_decam[:], magerr_2mass[:]]

    #add 0.02 mag uncertainty in quadrature to decaps
    mag_err[:,0:5] = np.sqrt(mag_err[:,0:5]**2 + 0.02**2)
    #add 0.03 mag uncertainty in quadrature to vvv/2mass
    mag_err[:,5:] = np.sqrt(mag_err[:,5:]**2 + 0.03**2)
    # mag=0 means that there is no detection
    mag = np.where(np.isclose(mag, 0.), np.nan, mag)
    mag_err = np.where(np.isclose(mag, 0.), np.nan, mag_err)

    # convert back to flux
    flux, flux_err = inv_magnitude(mag, mag_err)

    #unified quality mask
    cc_flag_ok = (m_dwarf_table['cc_flg']=='000')
    gal_contam_ok = (m_dwarf_table['gal_contam'] ==0)

    decam_nmag_cflux_ok = np.c_[m_dwarf_table['nmag_cflux_ok_g'], m_dwarf_table['nmag_cflux_ok_r'], 
                            m_dwarf_table['nmag_cflux_ok_i'],m_dwarf_table['nmag_cflux_ok_z'],
                            m_dwarf_table['nmag_cflux_ok_y']]
    decam_fracflux_avg_ok = np.c_[m_dwarf_table['fracflux_avg_g'], m_dwarf_table['fracflux_avg_r'],
                                m_dwarf_table['fracflux_avg_i'], m_dwarf_table['fracflux_avg_z'],
                                m_dwarf_table['fracflux_avg_y']]
    valid_decam = (mag[:, :5]>0) & (decam_nmag_cflux_ok>0) & (decam_fracflux_avg_ok>0.75)
    valid_2mass =(mag[:, 5:8]>0) & (cc_flag_ok[:, None]) & (gal_contam_ok[:, None])
    clean = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.)
    clean[:, :5] *= valid_decam
    clean[:, 5:8] *= valid_2mass

    # threshold on number of good bands
    flux_mask = (np.sum(clean, axis=1) >= 4) & (np.sum(clean[:, :5], axis=1) >= 1)
    merged_table_selected = m_dwarf_table[flux_mask]
    parallax, parallax_err = merged_table_selected['Plx'], merged_table_selected['e_Plx']
    correct_parallax_mask = (np.isfinite(parallax)) & (np.isin(merged_table_selected['Solved'], [31, 95]))
    flux_new = flux[flux_mask, :]
    flux_err_new = flux_err[flux_mask, :]
    mask = np.isfinite(mag_err)[flux_mask, :]  # create boolean band mask
    # parallax, parallax_err = m_dwarf_table['Plx'][flux_mask], m_dwarf_table['e_Plx'][flux_mask]
    correct_parallax_mask = (np.isfinite(parallax)) & (np.isin(m_dwarf_table['Solved'][flux_mask], [31, 95]))
    #apply parallax correction
    parallax_correction = zpt.get_zpt(m_dwarf_table['Gmag'][flux_mask][correct_parallax_mask], 
                                    m_dwarf_table['nueff'][flux_mask][correct_parallax_mask], 
                                    m_dwarf_table['pscol'][flux_mask][correct_parallax_mask], 
                                    m_dwarf_table['elat'][flux_mask][correct_parallax_mask], 
                                    m_dwarf_table['Solved'][flux_mask][correct_parallax_mask],
                                    _warnings=False)
    parallax_correction[~np.isfinite(parallax_correction)] = 0 
    parallax[correct_parallax_mask] = parallax[correct_parallax_mask]-parallax_correction
    coords = SkyCoord(ra = m_dwarf_table['ra'].value*u.deg, dec = m_dwarf_table['dec'].value*u.deg ).transform_to('galactic')
    coords = coords[flux_mask]

    BF_mist.fit(flux_new, flux_err_new, mask, 
                m_dwarf_table['SolID'][flux_mask], 
                filename+f'{region_name}_mist_MDwarf',
                data_coords = np.c_[coords.l.value, coords.b.value], 
                parallax=parallax, 
                parallax_err=parallax_err,
                phot_offsets = zp_mist, 
                # dustfile = dustfile, 
                Ndraws = 250, 
                Nmc_prior = 50, 
                logl_dim_prior=True,
                save_dar_draws = True, 
                running_io = True, 
                verbose= True
                )
    print(f'Fitted result saved to {filename}{region_name}_mist_MDwarf.h5')
    merged_table_selected.write(f'{base}2-star_modeling/M_dwarf_fit_results/{region_name}_Mdwarf.csv', format = 'csv')    
    print(f'Selected Mdwarf table saved to {base}2-star_modeling/M_dwarf_fit_results/{region_name}_Mdwarf.csv')
    np.savez(f"{base}2-star_modeling/M_dwarf_fit_results/{region_name}_parallax.npz", parallax=parallax.value,parallax_err=parallax_err.value)
    print(f'Saved modified parallax and parallax_err too.')

    return merged_table_selected, parallax, parallax_err


def run_nested_sampling(base, region_name): 
    filename = f'{base}2-star_modeling/output/'

    # selected_table = pd.read_csv(f"{base}2-star_modeling/M_dwarf_fit_results/{region_name}_Mdwarf.csv")

    f = h5py.File(filename+region_name+ '_mist_MDwarf'+ '.h5', 'r')
    chi2_mist =f['obj_chi2min'] #best fit chi2
    nbands_mist = f['obj_Nbands']# number of bands in fit
    dists_mist = f['samps_dist'] # distance samples
    reds_mist = f['samps_red'] # A(V) samples
    dreds_mist = f['samps_dred']# R(V) samples
    good=(stats.chi2.sf(chi2_mist, nbands_mist) > 0.01)
    dists_mist = np.array(dists_mist)[good, :]
    reds_mist = np.array(reds_mist)[good, :]
    dreds_mist = np.array(dreds_mist)[good, :]

    plx_data = np.load(f"{base}2-star_modeling/M_dwarf_fit_results/{region_name}_parallax.npz")
    parallax, parallax_err = plx_data['parallax'], plx_data['parallax_err']
    pdfbin, xedges, yedges = brutus.pdf.bin_pdfs_distred((dists_mist, reds_mist, dreds_mist), 
                                                        parallaxes = parallax,  # limits smoothing
                                                        parallax_errors=parallax_err,  # if parallax SNR is high
                                                        #  avlim=(0., 4.5),
                                                        )
    N_stars = pdfbin.shape[0]
    max_positions = np.argmax(pdfbin.reshape(N_stars, -1), axis=1)
    x_idx, y_idx = np.unravel_index(max_positions, (750, 300))  # shape: (N_stars,)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])  # shape: (750,)
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])  # shape: (300,)
    x_max = x_centers[x_idx] # want to choose x_max<10.5 to avoid the second distance reference
    y_max = y_centers[y_idx]
    plot_path = '/Users/anniegao/Documents/CG_mapping_files/3-nested_sampling/plots'
    np.savez(f"{plot_path}/{region_name}_individual_star_samples.npz",
            good=good,
            x_max=x_max,
            y_max=y_max,
            xedges=xedges,
            yedges=yedges,
            pdfbin = pdfbin)

    # export stars with maximum likelihood.
    merged_table = pd.read_csv(f'{base}2-star_modeling/M_dwarf_fit_results/{region_name}_Mdwarf.csv')[good]
    merged_table['derived_dist'] = 10**( x_max/5 +1)
    merged_table['derived Av'] = y_max
    merged_table.to_csv(f'{base}2-star_modeling/M_dwarf_fit_results/{region_name}_Mdwarf_res.csv', index= False)

    
    # convert from kpc to distance modulus, add limit of 10.5 distance modulus to avoid confusion
    dist_mist_new = dists_mist[x_max<10.5]
    red_mist_new = reds_mist[x_max<10.5]
    dms_mist = 5. * np.log10(dist_mist_new) + 10
    nclouds = 1  # number of clouds
    ndim = 2 * nclouds + 4  # number of parameters

    # adjust default distance modulus limits in ptform
    ptform_kwargs = {'dlims': (6., 10.)}

    # distances and extinctions to be passed to loglike
    logl_args = [dms_mist, red_mist_new]
    logl_kwargs = {'monotonic': True}  # extinctions must increase

    # fit dust along the LOS with dynesty
    sampler = dynesty.NestedSampler(loglike, ptform, ndim,
                                    ptform_kwargs=ptform_kwargs,
                                    logl_args=logl_args,
                                    # logl_kwargs=logl_kwargs
                                    )
    sampler.run_nested(dlogz=0.01)
    # save results
    res = sampler.results

    with open(f'{base}3-nested_sampling/results/{region_name}_nested_sampling_res.pkl', 'wb') as f:
        pickle.dump(res, f)
    print(f'Nested Sampling result saved to {base}3-nested_sampling/results/{region_name}_nested_sampling_res.pkl')

    return good, res

def make_plots(base, region_name):
    plot_path = os.path.join(base, '3-nested_sampling', 'plots')
    star_sample = np.load(f"{plot_path}/{region_name}_individual_star_samples.npz")
    x_max= star_sample['x_max'],
    y_max= star_sample['y_max'],
    pdfbin = star_sample['pdfbin']
    with open(f'{base}3-nested_sampling/results/{region_name}_nested_sampling_res.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    mu_samples = loaded_dict.samples[:, 4]

    # --- Plot 1: Av vs Mu with posterior ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))# Prepare the plot
    # Set axis labels
    ax1.set_xlabel('Distance Modulus $\mu$')
    ax1.set_ylabel('Extinction (Av)')
    im = ax1.imshow(np.sum(pdfbin, axis=0).T, aspect='auto', cmap='Blues', interpolation=None, origin='lower', 
                    extent=[star_sample['xedges'][0], star_sample['xedges'][-1], star_sample['yedges'][0], star_sample['yedges'][-1]],
                    # vmin=0, vmax=0.08
                    )
    ax1.scatter(x_max, y_max, color='red', s=3, alpha=0.6, label='Star max PDF location')
    ax1.axvline(7.4, c='y', alpha=0.5, ls='-.', label='IVS distance')
    ax1.axvspan(6.98, 8.7, facecolor='y', alpha=0.1)
    # fitted distance modulus:
    ax1.axvspan(np.quantile(mu_samples, 0.16),  np.quantile(mu_samples, 0.84), facecolor='C2', alpha=0.1)
    ax1.axvline(np.quantile(mu_samples, 0.5), c='C2', alpha=0.5, ls='-.', label = f'Posterior {region_name} $\mu$')

    ax1.set_title(f'Star pdf and {region_name} Distance Posterior')
    cbar = fig1.colorbar(im, ax=ax1)
    cbar.set_label(f'PDF for each star in region {region_name}')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{plot_path}/{region_name}_av_vs_mu_star_posteriors.png", dpi=300)
    plt.close(fig1)

   # --- Plot 2: Corner plot ---
    fig2, axes = dyplot.cornerplot(loaded_dict, 
                                labels=['P_b', 's_0', 's', 'av_0', 'mu_1', 'av_1'], # p_b: portion of outliers
                                show_titles=True,
                                fig=plt.subplots(6, 6, figsize=(30, 30)))
    fig2.tight_layout()
    fig2.savefig(f"{plot_path}/{region_name}_corner_plot.png", dpi=300)
    plt.close(fig2)
    
    return fig1, fig2