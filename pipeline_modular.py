from __future__ import print_function, division
import dynesty
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py
import brutus
# from dl import queryClient as qc
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
from pathlib import Path
import logging 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PATH = Path('/Users/anniegao/Documents/CG_mapping_files')
# All other paths derive from it
QUERIED_STARS_DIR = BASE_PATH / '1-queried_stars'
FIT_RESULTS_DIR = BASE_PATH / '2-star_modeling/M_dwarf_fit_results'
PLOTS_DIR = BASE_PATH / '3-nested_sampling/plots'
CLOUD_FIT_DIR = BASE_PATH/'3-nested_sampling/results'
GRID_FILE = BASE_PATH / '2-star_modeling/grid_mist_v10.h5'
OFFSET_FILE = BASE_PATH / '2-star_modeling/offsets_mist_v9.txt'

config = {
    "mdwarf": {"cut1": 0.78, "cut2": 22.5},
    "phot_err": {"decam": 0.02, "tmass": 0.03, 'unwise': 0.04},
    "chi2_pval": 0.01,
}

PLOT_RC = {
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
}
# Apply plot settings
plt.rcParams.update(PLOT_RC)

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
    xmatch_pd_clean = xmatch_pd.sort_values('angDist').drop_duplicates(subset = 'obj_id', keep='first')
    xmatch_pd_tomerge = xmatch_pd_clean.iloc[:, 210:].copy()
    xmatch_pd_tomerge['obj_id'] = xmatch_pd_clean['obj_id'].values
    xmatch_pd_tomerge['angDist'] = xmatch_pd_clean['angDist'].values
    merge_xmatch = df1_pd.merge(xmatch_pd_tomerge, how='left', left_on='obj_id', right_on='obj_id')
    # merged_pd = merge_xmatch.drop_duplicates(subset='obj_id', keep='first') # because there are 13 stars that are duplicated -- two Gaia sources are within 0.5 arcsec from Decaps2. Only keep the closer one. 
    pd_merged = Table.from_pandas(merge_xmatch) # convert back to astropy table # pd_merged.write("/Users/anniegao/Documents/CG_mapping_files/CG31/CG31_Decaps2Gaia_crossmatched.csv",format='csv')
    print('Finished crossmatching with Gaia')
    ## Cross Match with 2MASS
    twomass_xmatched = XMatch.query(cat1 = df1, cat2 = 'vizier:II/246/out', max_distance= 0.5*u.arcsec, colRA1='ra', colDec1='dec')
    twomass_xmatch_pd = twomass_xmatched.to_pandas()
    twomass_xmatch_pd = twomass_xmatch_pd.sort_values('angDist').drop_duplicates(subset='obj_id', keep='first')
    twomass_xmatch_id = twomass_xmatch_pd.iloc[:, 210:211]
    twomass_xmatch_id['obj_id'] = twomass_xmatch_pd['obj_id'].values
    twomass_xmatch_id['angDist'] = twomass_xmatch_pd['angDist'].values
    merge1 = df1_pd.merge(twomass_xmatch_id, how='left', left_on='obj_id', right_on='obj_id')
    merge2 = merge1.merge(df3.to_pandas().iloc[:, 2:], how='left',left_on='2MASS', right_on = 'designation') # Table.from_pandas(merge2).write("/Users/anniegao/Documents/CG_mapping_files/CG31/CG31_Decaps2TwoMASS_crossmatched.csv",format='csv', overwrite=True)
    print('Finished crossmatching with 2MASS')
    ## Merge three tables
    a = merge2.iloc[:, 209:].copy()
    a['obj_id'] = merge2['obj_id'].values
    decaps_2mass_gaia = pd_merged.to_pandas().merge(a, on = 'obj_id', how = 'left') # merge Decaps + 2MASS with Decaps + Gaia 
    output_path = f'{folder_name}_decaps_2mass_gaia.csv'
    Table.from_pandas(decaps_2mass_gaia).write(output_path,format='csv', overwrite=True)
    print(f"[{region_name}] All catalogs merged and saved to:\n→ {output_path}")
    return decaps_2mass_gaia

def xmatch_unwise(region_name):
    pd_current = pd.read_csv(f'/Users/anniegao/Documents/CG_mapping_files/1-queried_stars/{region_name}_decaps_2mass_gaia.csv')
    unwise_xmatched = XMatch.query(cat1 = Table.from_pandas(pd_current), cat2='vizier:II/363/unwise', max_distance= 0.5*u.arcsec, colRA1='ra', colDec1='dec').to_pandas()
    unwise_clean = (
        unwise_xmatched
        .sort_values('angDist')
        .drop_duplicates(subset='obj_id', keep='first')
    )
    unwise_pd_tomerge = unwise_clean.iloc[:, 414:].copy()
    unwise_pd_tomerge['obj_id'] = unwise_clean['obj_id'].values
    unwise_pd_tomerge['angDist_unwise'] = unwise_clean['angDist'].values
    merge_xmatch = pd_current.merge(unwise_pd_tomerge, how='left', on='obj_id')
    merge_xmatch.to_csv(f'/Users/anniegao/Documents/CG_mapping_files/1-queried_stars/{region_name}_decaps_2mass_gaia_unwise.csv', index=False)
    return merge_xmatch

def flux_to_mag(flux_vals: np.ndarray, flux_err_vals: np.ndarray):
    """Convert flux to magnitude"""
    mag = -2.5 * np.log10(flux_vals)
    mag_err = 1.086 * flux_err_vals / flux_vals
    return mag, mag_err

# def mag_to_flux(mag: np.ndarray, mag_err: np.ndarray):
#     """Convert magnitude to flux"""
#     flux = 10 ** (-0.4 * mag)
#     flux_err = flux * mag_err / 1.086
#     return flux, flux_err

def mdwarf_cut(pd_merged: Table, cut_val1: float=config["mdwarf"]["cut1"], cut_val2: float=config["mdwarf"]["cut2"]):
    """
        Apply M-dwarf color-magnitude selection.
        Returns: Boolean mask for M-dwarf candidates
    """
    Ag, Ar, Ai = 1.272392, 0.876292, 0.678924
    # Dereddening using configured extinction coefficients
    mDwarf1 = pd_merged['mean_mag_g']- (Ag/(Ag-Ar))* (pd_merged['mean_mag_g'] - pd_merged['mean_mag_r']-1.4)
    mDwarf2 = pd_merged['mean_mag_r'] - pd_merged['mean_mag_i'] - (Ar-Ai)/(Ag-Ar) *(pd_merged['mean_mag_g']-pd_merged['mean_mag_r']-1.4)
    return (mDwarf1 < cut_val2) & (mDwarf2 > cut_val1)

def prepare_photometry(pd_merged: Table, max_mag_err = None):
    """Extract and prepare photometry from merged catalog. 
        Returns dictionary with flux, flux_err, mag, mag_err arrays"""
    if max_mag_err is None:
        max_mag_err = 0.25
    flux_decam = np.c_[pd_merged['mean_g'].value, pd_merged['mean_r'].value, pd_merged['mean_i'].value, pd_merged['mean_z'].value, pd_merged['mean_y'].value]
    flux_decam_err = np.c_[pd_merged['err_g'].value, pd_merged['err_r'].value, pd_merged['err_i'].value,pd_merged['err_z'].value, pd_merged['err_y'].value]
    mag_2mass =  np.c_[pd_merged['j_m'].value, pd_merged['h_m'].value, pd_merged['k_m'].value ] #pd_merged['Gmag'], pd_merged['BPmag'], pd_merged['RPmag'],
    magerr_2mass = np.c_[pd_merged['j_msigcom'].value, pd_merged['h_msigcom'].value, pd_merged['k_msigcom'].value]  #pd_merged['e_Gmag'], pd_merged['e_BPmag'], pd_merged['e_RPmag'],
    flux_unwise = np.c_[pd_merged['FW1'], pd_merged['FW2']]
    flux_unwise_err = np.c_[pd_merged['e_FW1'], pd_merged['e_FW2']]
    mag_unwise = -2.5* np.log10(flux_unwise) + 22.5
    magerr_unwise = 1.086 * (flux_unwise_err/flux_unwise)
    mag_decam, magerr_decam  = flux_to_mag(flux_decam, flux_decam_err) #-2.5*np.log10(flux_decam)
    mag = np.c_[mag_decam[:], mag_2mass[:], mag_unwise[:]]
    mag_err = np.c_[magerr_decam[:], magerr_2mass[:], magerr_unwise[:]]
    #add 0.02 mag uncertainty in quadrature to decaps
    mag_err[:,0:5] = np.sqrt(mag_err[:,0:5]**2 + config["phot_err"]["decam"]**2)
    #add 0.03 mag uncertainty in quadrature to vvv/2mass
    mag_err[:,5:8] = np.sqrt(mag_err[:,5:8]**2 + config["phot_err"]["tmass"]**2)
    #add 0.04 mag uncertainty in quadrature for unWISE
    mag_err[:,8:10] = np.sqrt(mag_err[:,8:10]**2 + config["phot_err"]["unwise"]**2)
    # mag=0 means that there is no detection
    mag = np.where(np.isinf(mag), np.nan, mag)
    mag_err = np.where(mag_err>max_mag_err, np.nan, mag_err)
    # # convert back to flux
    # flux, flux_err = inv_magnitude(mag, mag_err)
    return {
            'mag': mag,
            'mag_err': mag_err,
            # 'flux': flux,
            # 'flux_err': flux_err
        }
 
def create_quality_mask(pd_table: Table, mag: np.ndarray, mag_err: np.ndarray):
    """Create unified quality mask for photometry
       Returns Boolean mask array indicating good measurements"""

    min_bands = 4
    min_decam_bands = 1
    # Convert to flux
    flux, flux_err = inv_magnitude(mag, mag_err)
    # 2MASS quality flags
    cc_flag_ok = (pd_table['cc_flg']=='000')
    gal_contam_ok = (pd_table['gal_contam'] ==0)
    # DECam quality flags
    decam_nmag_cflux_ok = np.c_[pd_table['nmag_cflux_ok_g'], pd_table['nmag_cflux_ok_r'], 
                            pd_table['nmag_cflux_ok_i'],pd_table['nmag_cflux_ok_z'],
                            pd_table['nmag_cflux_ok_y']]
    decam_fracflux_avg_ok = np.c_[pd_table['fracflux_avg_g'], pd_table['fracflux_avg_r'],
                                pd_table['fracflux_avg_i'], pd_table['fracflux_avg_z'],
                                pd_table['fracflux_avg_y']]
    # unWISE quality flags
    unwise_flags = (pd_table['FlagsW1'] == 0) & (pd_table['FlagsW2']==0)
    unwise_err_ok = (pd_table['e_FW1'] <0.05) & (pd_table['e_FW2']< 0.05)
    unwise_fraclux_ok = (pd_table['fFW1']> 0.85) & (pd_table['fFW2']> 0.85)
    # Valid measurements
    valid_decam = (flux[:, :5]>0) & (decam_nmag_cflux_ok>0) & (decam_fracflux_avg_ok>0.75)
    valid_2mass =(flux[:, 5:8]>0) & (cc_flag_ok[:, None]) & (gal_contam_ok[:, None])
    valid_unwise = (flux[:, 8:10]>0) & (unwise_flags[:, None]) & (unwise_err_ok[:, None]) & (unwise_fraclux_ok[:, None])# Convert to flux for final mask

    # Clean mask
    clean = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.)
    clean[:, :5] *= valid_decam
    clean[:, 5:8] *= valid_2mass
    clean[:, 8:10] *= valid_unwise
    # threshold on number of good bands
    final_mask = (np.sum(clean, axis=1) >= min_bands) & (np.sum(clean[:, :5], axis=1) >= min_decam_bands)
    return final_mask

def correct_parallax(pd_merged: Table, parallax: np.ndarray, parallax_err: np.ndarray):
    """
        Apply parallax zero-point correction.
        Returns: Corrected parallax and parallax_err
    """
    from zero_point import zpt
    # Load correction tables
    zpt.load_tables()
    # Identify sources with valid parallax solutions
    correct_parallax_mask = (np.isfinite(parallax)) & (np.isin(pd_merged['Solved'], [31, 95]))
    
    #apply zero-point correction
    parallax_correction = zpt.get_zpt(pd_merged['Gmag'][correct_parallax_mask], 
                                    pd_merged['nueff'][correct_parallax_mask], 
                                    pd_merged['pscol'][correct_parallax_mask], 
                                    pd_merged['elat'][correct_parallax_mask], 
                                    pd_merged['Solved'][correct_parallax_mask],
                                    _warnings=False)
    # handle invalid corrections
    parallax_correction[~np.isfinite(parallax_correction)] = 0 
    # apply all corrections
    parallax_corrected = parallax.copy()
    parallax_corrected[correct_parallax_mask] -= parallax_correction
    
    return parallax_corrected, parallax_err

def clean_data(pd_merged, region_name, apply_mdwarf_cut = False, md2_min = config["mdwarf"]["cut1"], md1_max = config["mdwarf"]["cut2"], save_selected_w_plx = True):
    logger.info(f"Cleaning data for {region_name} (M-dwarf cut: {apply_mdwarf_cut})")
    if not isinstance(pd_merged, Table):
        pd_merged = Table.from_pandas(pd_merged)

    tag = "_mdwarf" if apply_mdwarf_cut else "_allstar"

    if apply_mdwarf_cut:
        mdwarf_mask = mdwarf_cut(pd_merged, md2_min, md1_max)
        pd_merged = pd_merged[mdwarf_mask]
        logger.info(f"  M-dwarf cut: {len(pd_merged)} stars selected")
    # prepare photometry
    phot_data = prepare_photometry(pd_merged)
    mag_preped = phot_data['mag']
    mag_err_preped = phot_data['mag_err']
    final_mask = create_quality_mask(pd_merged, mag_preped, mag_err_preped)
    pd_merged_selected = pd_merged[final_mask]

    parallax, parallax_err = correct_parallax(pd_merged_selected, pd_merged_selected['Plx'].value.copy(), pd_merged_selected['e_Plx'].value.copy())
    pd_merged_selected['parallax_fit'] = parallax
    pd_merged_selected['parallax_err_fit'] = parallax_err
    
    if save_selected_w_plx:
        outpath = FIT_RESULTS_DIR / f"{region_name}{tag}_selected.csv"
        pd_merged_selected.write(str(outpath), format='csv', overwrite=True)
        logger.info(f"  Selected table saved to {outpath}")

    return {
    "table": pd_merged_selected,
    "mag": mag_preped[final_mask],
    "mag_err": mag_err_preped[final_mask],
    "mask": final_mask,
    "parallax": parallax,
    "parallax_err": parallax_err,
    "tag": tag
}

def fit_brutus(cleaned_data, region_name, tag):
    # Extract data
    table_sel = cleaned_data['table']
    mag_sel = cleaned_data['mag']
    mag_err_sel = cleaned_data['mag_err']
    parallax = cleaned_data['parallax']
    parallax_err = cleaned_data['parallax_err']
    filt = filters.decam[1:] + filters.tmass[:] + filters.wise[:2] #+filters.gaia[:]
    # zero points
    zp_mist = brutus.utils.load_offsets(OFFSET_FILE,filters=filt)
    # import MIST model grid
    gridfile = str(GRID_FILE)
    (models_mist, labels_mist, lmask_mist) = brutus.utils.load_models(gridfile, filters=filt)
    BF_mist = fitting.BruteForce(models_mist, labels_mist, lmask_mist)
    #load tables for parallax zeropoint correction
    zpt.load_tables()
    flux, flux_err = inv_magnitude(mag_sel, mag_err_sel)
    coords = SkyCoord(ra = table_sel['ra'].value*u.deg, 
                      dec = table_sel['dec'].value*u.deg ).transform_to('galactic')
    output_prefix = FIT_RESULTS_DIR / f"{region_name}{tag}_brutus"
    logger.info(f"  Fitting {len(table_sel)} stars...")
    BF_mist.fit(flux, flux_err, 
                np.isfinite(mag_err_sel),
                table_sel['SolID'], 
                str(output_prefix), #f'{base}2-star_modeling/output/'+'Akari_CG22_emission30_cut22'
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
    logger.info(f"  Fit complete: {output_prefix}.h5")
    return output_prefix

def from_h5_file(region_name, tag="_mdwarf", chi2_threshold=None, 
                save_pdf_results=True):
    """
    Load fit results from HDF5 file and create PDF bins.
    
    Args:
        region_name: Region name
        tag: File naming tag
        chi2_threshold: Minimum chi2 probability (uses config if None)
        save_pdf_results: Whether to save PDF results
    
    Returns:
        Dictionary with fit results and PDF data
    """
    import brutus
    
    if chi2_threshold is None:
        chi2_threshold = config["chi2_pval"]
    
    
    # FIX: Consistent path construction
    h5file_path = FIT_RESULTS_DIR / f"{region_name}{tag}_brutus.h5"
    
    if not h5file_path.exists():
        raise FileNotFoundError(f"Fit results not found: {h5file_path}")
    
    # Load fit results
    with h5py.File(h5file_path, 'r') as f:
        chi2_mist = np.array(f['obj_chi2min'])
        nbands_mist = np.array(f['obj_Nbands'])
        dists_mist = np.array(f['samps_dist'])
        reds_mist = np.array(f['samps_red'])
        dreds_mist = np.array(f['samps_dred'])
    
    # Apply chi2 quality cut
    good = stats.chi2.sf(chi2_mist, nbands_mist) > chi2_threshold
    if np.sum(good) == 0:
        raise ValueError(f"No stars passed chi2 threshold for {region_name}")
    
    # Select good stars
    dists_good = dists_mist[good, :]
    reds_good = reds_mist[good, :]
    dreds_good = dreds_mist[good, :]
    
    # Load parallax for good stars (FIX: Critical bug fix!)
    selected_table_path = FIT_RESULTS_DIR / f"{region_name}{tag}_selected.csv"
    if not selected_table_path.exists():
        raise FileNotFoundError(f"Selected table not found: {selected_table_path}")
    
    selected_table = Table.read(selected_table_path)
    parallax = selected_table['parallax_fit'].value[good]  # FIX: Apply good mask
    parallax_err = selected_table['parallax_err_fit'].value[good]  # FIX: Apply good mask
    
    # Create PDF bins
    pdfbin, xedges, yedges = brutus.pdf.bin_pdfs_distred(
        (dists_good, reds_good, dreds_good), 
        parallaxes=parallax,
        parallax_errors=parallax_err
    )
    
    # Find maximum likelihood for each star
    N_stars = pdfbin.shape[0]
    max_positions = np.argmax(pdfbin.reshape(N_stars, -1), axis=1)
    x_idx, y_idx = np.unravel_index(max_positions, pdfbin.shape[1:])
    
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    
    x_max = x_centers[x_idx]
    y_max = y_centers[y_idx]
    
    # Save PDF results if requested
    if save_pdf_results:
        pdf_path = PLOTS_DIR / f"{region_name}_individual_star_samples.npz"
        np.savez(
            pdf_path,
            good=good,
            x_max=x_max,
            y_max=y_max,
            xedges=xedges,
            yedges=yedges,
            pdfbin=pdfbin
        )
    
    return {
        "dists_good": dists_good,
        "reds_good": reds_good,
        "dreds_good": dreds_good,
        "good_mask": good,
        "pdfbin": pdfbin,
        "xedges": xedges,
        "yedges": yedges,
        "x_max": x_max,
        "y_max": y_max
    }

def save_final_catalog(region_name, pdf_results, tag, save_file=False):
    good_mask = pdf_results["good_mask"]
    file_path = FIT_RESULTS_DIR / f"{region_name}{tag}_selected.csv"
    table = pd.read_csv(file_path)[good_mask]
    table['derived_dist'] = 10**(pdf_results["x_max"]/5 +1)
    table['derived_dist_modulus'] = pdf_results["x_max"]
    table['derived_Av'] = pdf_results["y_max"]
    if save_file:
        table.to_csv(FIT_RESULTS_DIR / f"{region_name}{tag}_res.csv", index= False)
        logger.info(f"  Final catalog saved: {FIT_RESULTS_DIR} /{region_name}{tag}_res.csv ")
        logger.info(f"  Total stars: {len(table)}")
    return table

def run_nested_sampling(dists_mist, reds_mist, region_name, save_pkl= False):
    dms_mist = 5. * np.log10(dists_mist) + 10
    nclouds = 1  # number of clouds
    ndim = 2 * nclouds + 4  # number of parameters

    # adjust default distance modulus limits in ptform
    ptform_kwargs = {'dlims': (6., 10.)}

    # distances and extinctions to be passed to loglike
    logl_args = [dms_mist, reds_mist]
    # logl_kwargs = {'monotonic': True}  # extinctions must increase

    # fit dust along the LOS with dynesty
    sampler = dynesty.NestedSampler(loglike, ptform, ndim,
                                    ptform_kwargs=ptform_kwargs,
                                    logl_args=logl_args,
                                    # logl_kwargs=logl_kwargs
                                    )
    sampler.run_nested(dlogz=0.01)
    # save results
    res = sampler.results
    if save_pkl:
        file_path = f"{CLOUD_FIT_DIR}/{region_name}_nested_sampling_res.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(res, f)
        logger.info(f"  Results saved to {file_path}")
    return res

def plot_av_vs_mu(pdf_results, nested_res, region_name, save=True):
    """Av vs distance modulus plot with star PDF and posterior"""
    weights = np.exp(nested_res.logwt - nested_res.logz[-1])
    samples_equal = dynesty.utils.resample_equal(nested_res.samples, weights)
    av0 = samples_equal[:, 3]
    mu_samples = samples_equal[:, 4]
    av1 = samples_equal[:, 5]
    fig, ax = plt.subplots(figsize=(10, 8))# Prepare the plot
    # Av vs Mu with posterior 
    # Set axis labels
    ax.set_xlabel('Distance Modulus $\mu$')
    ax.set_ylabel('Extinction $A_V$ (mag)')
    pdfbin = pdf_results['pdfbin']
    xedges, yedges = pdf_results['xedges'], pdf_results['yedges']
    x_max, y_max = pdf_results['x_max'], pdf_results['y_max']
    im = ax.imshow(np.sum(pdfbin, axis=0).T, aspect='auto', cmap='Blues', interpolation=None, origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    # vmin=0, vmax=0.08
                    )
    ax.scatter(x_max, y_max, color='red', s=3, alpha=0.6, label='Star max PDF location')
    ax.axvline(7.4, c='y', alpha=0.5, ls='-.', label='IVS distance')
    ax.axvspan(6.98, 8.7, facecolor='y', alpha=0.1)
    ax.axvline(5*np.log10(359)-5, c='C4' , alpha = 0.5, ls = '-.', label = 'YSO distance')
    # fitted distance modulus:
    mu50 = np.quantile(mu_samples, 0.5)
    mu16, mu84 = np.quantile(mu_samples, 0.16), np.quantile(mu_samples, 0.84)
    ax.axvspan(mu16, mu84, facecolor='C2', alpha=0.1)
    ax.axvline(mu50, c='C2', alpha=0.5, ls='-.', label = f'Posterior {region_name} (μ={mu50:.2f})')
    av0_50, av1_50 = np.quantile(av0, 0.5), np.quantile(av1, 0.5)
    ax.hlines(y = av0_50, xmin=4, xmax = mu50, label = f'$A_V0$ = {av0_50:.2f}', colors= 'C2', linewidth=2)
    ax.hlines(y = av1_50, xmin=mu50, xmax = 19, label=f'$A_V1$ = {av1_50:.2f}', colors= 'C2', linewidth=2)
    # dist_tick_locations = sorted(np.concatenate((np.arange(4,20, 2), [np.quantile(mu_samples, 0.16), np.quantile(mu_samples, 0.5), np.quantile(mu_samples, 0.84)])))
    dist_tick_locations = sorted(np.concatenate((
        np.arange(np.floor(xedges[0]), np.ceil(xedges[-1]), 2),
        [mu50]
    )))

    ax.set_title(f'{region_name} Distance Posterior')
    ax.set_xticks(dist_tick_locations)
    ax.tick_params(axis='x', labelrotation=90)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'PDF for each star')
    ax.legend(loc='upper right', framealpha=0.8, fontsize='x-small')

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # upside-down histogram in the main axes 
    ax_hist = inset_axes(ax, width="100%", height="100%",    # keep this at 100%
        loc="upper left", bbox_to_anchor=(0, 0.56, 1, 0.44),  # taller inset (44% of ax1)
        bbox_transform=ax.transAxes, borderpad=0 )

    ax_hist.set_facecolor('none')
    ax_hist.patch.set_alpha(0)
    ax_hist.hist(mu_samples, bins=50, histtype='step', density=True, color='C2')
    # Flip histogram upside-down
    ax_hist.invert_yaxis()
    # EXACT same x-axis
    ax_hist.set_xlim(ax.get_xlim())
    # Clean look
    ax_hist.set_xticks([])
    ax_hist.set_yticks([])
    for spine in ['right', 'top', 'left', 'bottom']:
        ax_hist.spines[spine].set_visible(False)

    fig.tight_layout()
    if save:
        save_path = f"{PLOTS_DIR}/{region_name}_av_vs_mu_star_posteriors.png"
        fig.savefig(save_path, dpi=300, bbox_inches = 'tight')
    return fig

def plot_corner(nested_res, region_name, save=True):
    """Corner plot from nested sampling"""
    fig, axes = dyplot.cornerplot(nested_res, 
                                labels=['P_b', 's_0', 's', 'av_0', 'mu_1', 'av_1'], # p_b: portion of outliers
                                show_titles=True,
                                fig=plt.subplots(6, 6, figsize=(30, 30)))
    fig.suptitle(f'{region_name} Nested Sampling Posterior', fontsize=20, y=0.995)
    fig.tight_layout()
    if save:
        save_path = f"{PLOTS_DIR}/{region_name}_corner_plot.png"
        fig.savefig(save_path, dpi=300, bbox_inches = 'tight')
    return fig