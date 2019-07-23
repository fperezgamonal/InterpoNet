import numpy as np
import os
from skimage.util.shape import view_as_blocks
from skimage.transform import rescale

UNKNOWN_FLOW_THRESH = 1e9
SMALLFLOW = 0.0
LARGEFLOW = 1e8
DEBUG = False  # flag to print out verbose information like: range of optical flow, dimensions of matrix, etc.


def downscale_all(img, mask, edges, downscale):
    """
    Downscales all the inputs by a given scale.
    :param img: A sparse  flow map - h x w x 2
    :param mask: A binary mask  - h x w x 1
    :param edges:An edges map  - h x w x 1
    :param downscale: Downscaling factor.
    :return: the downscaled versions of the inputs
    """
    img[:, :, 0][mask == -1] = np.nan
    img[:, :, 1][mask == -1] = np.nan

    img = img[:(img.shape[0] - (img.shape[0] % downscale)), :(img.shape[1] - (img.shape[1] % downscale)), :]

    blocks = view_as_blocks(img, (downscale, downscale, 2))
    img = np.nanmean(blocks, axis=(-2, -3, -4))

    mask = np.ones_like(img)
    mask[np.isnan(img)] = -1
    mask = mask[:, :, 0]
    img[np.isnan(img)] = 0

    if edges is not None:
        edges = edges[:(edges.shape[0] - (edges.shape[0] % downscale)), :(edges.shape[1] - (edges.shape[1] % downscale))]
        edges = rescale(edges, 1 / float(downscale), preserve_range=True)

    return img, mask, edges


def inverse_of_map(of_map, of_mask, scale):
    """
    Inverse the input flow map turns (x1,y1) -> (dx, dy) to (x2,y2) -> (-dx, -dy),
    :param of_map: h x w x 2 optical flow map
    :param of_mask: h x w binary mask for the existing pixels in the flow map
    :param scale: The scale used to downsample the flow map before feeding it into the function
    :return: A flow map with the inverse input optical flow map.
    """
    map_count = np.zeros((of_map.shape[0], of_map.shape[1]))
    rev_of_map = np.zeros_like(of_map, dtype=np.float32)
    for y in range(of_map.shape[0]):
        for x in range(of_map.shape[1]):
            if of_mask[y, x] == 1:
                rev_x = (x * scale + of_map[y, x, 0]) / scale
                rev_y = (y * scale + of_map[y, x, 1]) / scale
                rev_x = int(np.floor(rev_x + 0.5))
                rev_y = int(np.floor(rev_y + 0.5))
                if 0 < rev_x < rev_of_map.shape[1] and 0 < rev_y < rev_of_map.shape[0]:
                    map_count[rev_y, rev_x] += 1
                    rev_of_map[rev_y, rev_x] += (-rev_of_map[rev_y, rev_x] - of_map[y, x]) / map_count[rev_y, rev_x]
    return rev_of_map, (np.asarray(map_count > 0, dtype=np.float32) * 2) - 1


def mean_map_of_and_rev_ba(of_map, of_mask, rev_of_map_ba, rev_of_mask_ba):
    """
    Calculates the mean flow map between the optical flow map from A to B and the inverse flow map from B to A
    :param of_map: flow map from A to B
    :param of_mask: binary mask of AB flow map
    :param rev_of_map_ba: inverse flow map from B to A
    :param rev_of_mask_ba: binary mask of BA inverse flow map
    :return: the mean flow map
    """
    of_map_nan = of_map.copy()
    of_map_nan[np.stack((of_mask, of_mask), axis=2) == -1] = np.nan
    rev_of_map_ba_nan = rev_of_map_ba.copy()
    rev_of_map_ba_nan[np.stack((rev_of_mask_ba, rev_of_mask_ba), axis=2) == -1] = np.nan

    mean_map = np.nanmean(np.stack((of_map_nan, rev_of_map_ba_nan), axis=3), axis=3, dtype=np.float32)
    mean_map_mask = -(np.asarray(np.isnan(mean_map), np.float32) * 2 - 1)[:, :, 0]
    mean_map[np.isnan(mean_map)] = 0

    return mean_map, mean_map_mask


def create_mean_map_ab_ba(img, mask, img_ba, mask_ba, scale):
    """
    A wrapper for the functions: inverse_of_map and mean_map_of_and_rev_ba
    :param img: flow map from A to B
    :param mask: binary mask of AB flow map
    :param img_ba: flow map from B to A
    :param mask_ba: binary mask of BA flow map
    :param scale: The scale used to downsample the flow map before feeding it into the function
    :return: the mean flow map
    """
    rev_of_map_ba, rev_of_mask_ba = inverse_of_map(img_ba, mask_ba, scale=scale)
    mean_map, mean_map_mask = mean_map_of_and_rev_ba(img, mask, rev_of_map_ba, rev_of_mask_ba)

    return mean_map, mean_map_mask


def calc_variational_inference_map(imgA_filename, imgB_filename, flo_filename, out_filename, dataset):
    """
    Run the post processing variation energy minimization.
    :param imgA_filename: filename of RGB image A of the image pair.
    :param imgB_filename: filename of RGB image B of the image pair.
    :param flo_filename: filename of flow map to set as initialization.
    :param out_filename: filename for the output flow map.
    :param dataset: sintel / kitti
    """
    shell_command = './SrcVariational/variational_main ' + imgA_filename + ' ' + imgB_filename + ' ' + flo_filename +\
                    ' ' + out_filename + ' -' + dataset
    exit_code = os.system(shell_command)

# Added some utils from https://github.com/fperezgamonal/flownet2-tf to compute metrics
def get_metrics(metrics, average=False, flow_fname=None):
    dash = "-" * 50
    line = '_' * 50
    if average:
        title_str = "{:^50}".format('MPI-Sintel Flow Error Metrics (AVERAGE)')
        flow_fname_str = "{:^50}".format('For all files above')
    else:
        title_str = "{:^50}".format('MPI-Sintel Flow Error Metrics')
        if flow_fname is not None:
            flow_fname_str = "{:^50}".format(flow_fname)
        else:
            flow_fname_str = "{:^50}".format('Unknown filename')

    headers = '{:<5s}{:^15s}{:^15s}{:^15s}'.format('Mask', 'MANG', 'STDANG', 'MEPE')
    all_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(all)', metrics['mangall'], metrics['stdangall'],
                                                            metrics['EPEall'])
    mat_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(mat)', metrics['mangmat'], metrics['stdangmat'],
                                                            metrics['EPEmat'])
    umat_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(umt)', metrics['mangumat'], metrics['stdangumat'],
                                                             metrics['EPEumat'])
    dis_headers = '{:<5s}{:^15s}{:^15s}{:^15s}'.format('', 'S0-10', 'S10-40', 'S40+')

    dis_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(dis)', metrics['S0-10'], metrics['S10-40'],
                                                            metrics['S40plus'])

    final_string_formatted = "{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n{8}\n{9}\n{10}\n{11}\n{12}\n".format(
        line, title_str, flow_fname_str, line, headers, dash, all_string, mat_string, umat_string, line, dis_headers,
        dash, dis_string)

    return final_string_formatted


# TODO: add other metrics available in the MATLAB script (EPEmat, EPEumat, S0-10, S10-40, S40+)
def compute_all_metrics(est_flow, gt_flow, occ_mask=None, inv_mask=None):
    """
    Computes the metrics (if enough masks are provided) of MPI-Sintel (EPEall, EPEmat, EPEumat, S0-10, S10-40, S40+)
        -NOTE: flow_error_mask discards the pixels with the passed value (i.e.: uses the 'complementary' mask).
    Based on the original MATLAB code from Stefan Roth. Changed by Ferran Pérez Gamonal to compute MPI-Sintel metrics.
    Ported here from MATLABs implementation (the original code can be found in the supplemental material of the
    following publication: http://www.ipol.im/pub/art/2019/238/)
    :param est_flow: estimated optical flow with shape: (height, width, 2)
    :param gt_flow: ground truth optical flow with shape: (height, width, 2)
    :param occ_mask: (optional) occlusions mask (1s specify that a pixel is occluded, 0 otherwise)
    :param inv_mask: (optional) invalid mask that specifies which pixels have invalid flow (not considered for error)
    :return: dictionary with computed metrics (0 if it cannot be computed, no masks)
    """
    metrics = dict([])
    bord = 0
    height, width, _ = gt_flow.shape
    # Separate gt flow fields (horizontal + vertical)
    of_gt_x = gt_flow[:, :, 0]
    of_gt_y = gt_flow[:, :, 1]

    # Separate est flow fields (horizontal + vertical)
    of_est_x = est_flow[:, :, 0]
    of_est_y = est_flow[:, :, 1]

    if occ_mask is not None:
        occ_mask = occ_mask == 255  # check that once read the value is 255
    else:
        occ_mask = np.full((height, width), False)

    if inv_mask is not None:
        inv_mask = inv_mask == 255  # check that once read the value is 255
    else:
        inv_mask = np.full((height, width), False)  # e.g.: every pixel has a valid flow

    # EPE all
    mang, stdang, mepe = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, inv_mask, True, bord)
    metrics['EPEall'] = mepe
    metrics['mangall'] = mang
    metrics['stdangall'] = stdang
    # Check if there are any occluded pixels
    if occ_mask.size and np.unique(occ_mask).shape[0] > 1:  # array is not empty and contains at least 2 diff. values
        # EPE-matched (pixels that are not occluded)
        # Always mask out invalid pixels (inv_mask == 1)
        # For matched we want to avoid the 1's
        mat_occ_msk = occ_mask | inv_mask  # 0's are valid and non-occluded ==> gt_value=1 (rejected value)
        mat_mang, mat_stdang, mat_mepe = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, mat_occ_msk, True, bord)

        # EPE-unmatched (pixels that are occluded)
        # " " " invalid pixels
        # For unmatched we want to avoid the 0's
        un_occ_msk = occ_mask & ~inv_mask  # 1's are valid and occluded
        umat_mang, umat_stdang, umat_mepe = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, un_occ_msk, False,
                                                            bord)
        not_occluded = 0
    else:
        # No occluded pixels (umat = 0, mat = all)
        mat_mepe = mepe
        umat_mepe = 0
        mat_mang = mang
        umat_mang = 0
        mat_stdang = stdang
        umat_stdang= 0

        # We need to count the number of occluded instances to properly compute averages of several images
        not_occluded = 1

    metrics['EPEmat'] = mat_mepe
    metrics['mangmat'] = mat_mang
    metrics['stdangmat'] = mat_stdang
    metrics['EPEumat'] = umat_mepe
    metrics['mangumat'] = umat_mang
    metrics['stdangumat'] = umat_stdang

    # Masks for S0 - 10, S10 - 40 and S40 +)
    l1_of = np.sqrt(of_gt_x ** 2 + of_gt_y ** 2)
    disp_mask = l1_of
    disp_mask[np.asarray(disp_mask < 10).nonzero()] = 0
    disp_mask[(disp_mask >= 10) & (disp_mask <= 40)] = 1
    # careful & takes precedence to <=/>=/== (use parenthesis)
    disp_mask[disp_mask > 40] = 2

    # Actually compute S0 - 10, S10 - 40 and S40 +
    # Note: not correct (ambiguous truth evaluation) in Python (used "number in array") instead
    # pixels_disp_1 = sum(disp_mask[:] == 0)  # S0-10
    # pixels_disp_2 = sum(disp_mask[:] == 1)  # S10-40
    # pixels_disp_3 = sum(disp_mask[:] == 2)  # S40+

    # Remember that flow_error_mask ignores the values equal to gt_value in the mask
    # So, for S0-10, we want to pass only the pixels with a velocity within the 0-10 range
    # We pass 1 in this position, -1 elsewhere (number different than the labels 0 through 2)
    # ======= S0-10 =======
    if 0 in disp_mask[:]:
        # Compute  S0 - 10 nominally
        msk_s010 = disp_mask
        # msk_s010[np.asarray(msk_s010 != 0).nonzero()] = -1
        # msk_s010[(msk_s010 == 0)] = 1
        # msk_s010[np.asarray(msk_s010 == -1).nonzero()] = 0
        # msk_s010 = msk_s010 == 1  # convert to bool! (True/False in python)
        # We want 1's only where 0's (pixels with velocity in range 0-10) in disp_mask, 0 elsewhere
        # Numpy has np.where(condition, value_where_cond_is_met, value_elsewhere)
        # And accepts bools
        msk_s010 = np.where(msk_s010 == 0, True, False)
        # Mask out invalid pixels(defined in the 'invalid' folder)
        # % We want to take into account only the valid and values = 1 in msk_s010
        msk_s010 = (msk_s010) & (~inv_mask)
        _, _, s0_10 = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, msk_s010, False, bord)
        s0_10_is_zero = 0
    else:
        s0_10 = 0
        # Count instances with no pixels in this range of movement to average over all images
        s0_10_is_zero = 1

    metrics['S0-10'] = s0_10

    # ======= S10-40 =======
    if 1 in disp_mask[:]:
        # Compute S10 - 40 nominally
        msk_s1040 = disp_mask  # have value 1
        # msk_s1040[np.asarray(msk_s1040 != 1).nonzero()] = -1
        # msk_s1040[np.asarray(msk_s1040 == -1).nonzero()] = 0
        # msk_s1040 = msk_s1040 == -1
        # np.where() to the rescue
        msk_s1040 = np.where(msk_s1040 == 1, True, False)

        # Mask out the invalid pixels
        # Same reasoning as s0 - 10 mask
        msk_s1040 = (msk_s1040) & (~inv_mask)
        # The desired pixels have already value 1, we are done.
        _, _, s10_40 = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, msk_s1040, False, bord)
        s10_40_is_zero = 0
    else:
        s10_40 = 0
        # Count instances with no pixels in this range of movement to average over all images
        s10_40_is_zero = 1

    metrics['S10-40'] = s10_40

    # ======= S40+ =======
    if 2 in disp_mask[:]:
        # Compute S40+ nominally
        msk_s40plus = disp_mask
        # msk_s40plus[np.asarray(msk_s40plus != 2).nonzero()] = -1
        # msk_s40plus[np.asarray(msk_s40plus == 2).nonzero()] = 1
        # msk_s40plus[np.asarray(msk_s40plus == -1).nonzero()] = 0
        # msk_s40plus = msk_s40plus == 1
        msk_s40plus = np.where(msk_s40plus == 2, True, False)

        # Mask out the invalid pixels
        # Same reasoning as s0 - 10 and s10 - 40 masks
        msk_s40plus = (msk_s40plus) & (~inv_mask)
        _, _, s40plus = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, msk_s40plus, False, bord)
        s40plus_is_zero = 0
    else:
        s40plus = 0
        # Count instances with no pixels in this range of movement to average over all images
        s40plus_is_zero = 1

    metrics['S40plus'] = s40plus

    return metrics, not_occluded, s0_10_is_zero, s10_40_is_zero, s40plus_is_zero

def flow_error_mask(tu, tv, u, v, mask=None, gt_value=False, bord=0):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param mask: binary mask that specifies a region of interest
    :param gt_value: specifies if we ignore False's (0's) or True's (0's) in the computation of a certain metric
    :return: End point error of the estimated flow
    """
    smallflow = 0.0

    # stu = tu[bord+1:end-bord,bord+1:end-bord]
    # stv = tv[bord+1:end-bord,bord+1:end-bord]
    # su = u[bord+1:end-bord,bord+1:end-bord]
    # sv = v[bord+1:end-bord,bord+1:end-bord]

    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknown = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH) | (mask == gt_value)
    # stu[idxUnknown] = 0
    # stv[idxUnknown] = 0
    # su[idxUnknown] = 0
    # sv[idxUnknown] = 0

    # ind2 = [(np.absolute(stu[:]) >= smallflow) | (np.absolute(stv[:]) >= smallflow)]
    ind2 = (abs(stu) >= smallflow) | (abs(stv) >= smallflow)
    ind2 = (idxUnknown < 1) & ind2
    index_su = su[ind2]  # should be updated to A[tuple(idx_list)]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    angle = un * tun + vn * tvn + (an * tn)
    index = [angle >= 1.0]  # due to some precision errors we may have 1.0001 instead of 1 an have a nan at the end!
    angle[index] = 0.999
    ang = np.arccos(angle)  # un * tun + vn * tvn + (an * tn))
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    stdang = np.std(ang * 180 / np.pi)

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mang, stdang, mepe
