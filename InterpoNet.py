import numpy as np
import tensorflow as tf
import skimage as sk
import utils
import io_utils
import model
import argparse
import os
import datetime
import shutil


# TODO: maybe for a more fair comparison, use padding with zeros like we do with flownetS
# auxiliar function to compute the new image size (for test only) for input images which are not divisble by divisor
# Important note: the authors slightly crop the content by a few pixels if the size is not divisible by the downscale
# factor. Padding with zeros and then cropping back to the original content may be better to avoid interpolation
# artifacts at the borders
def get_downsampled_image_size(og_height, og_width, downscale_factor=8):
    new_height = (og_height - (og_height % downscale_factor)) / downscale_factor
    new_width = (og_width - (og_width % downscale_factor)) / downscale_factor
    return new_height, new_width


def test_one_image(args):
    # Load edges file
    print("Loading files...")
    edges = io_utils.load_edges_file(args.edges_filename, width=args.img_width, height=args.img_height)

    # Load matching file
    sparse_flow, mask_matches = io_utils.load_matching_file(args.matches_filename, width=args.img_width,
                                                            height=args.img_height)
    sparse_flow_img = utils.flow_to_image(sparse_flow)
    sk.io.imsave('tmp_sparse_flow_ogscale.png', sparse_flow_img)

    # downscale
    print("Downscaling...")
    sparse_flow, mask_matches, edges = utils.downscale_all(sparse_flow, mask_matches, edges, args.downscale)

    if args.ba_matches_filename is not None:
        sparse_flow_ba, mask_matches_ba = io_utils.load_matching_file(args.ba_matches_filename, width=args.img_width,
                                                                      height=args.img_height)
        sparse_flow_img = utils.flow_to_image(sparse_flow_ba)
        sk.io.imsave('tmp_sparse_flow_ogscale_bwd.png', sparse_flow_img)

        # downscale ba
        sparse_flow_ba, mask_matches_ba, _ = utils.downscale_all(sparse_flow_ba, mask_matches_ba, None, args.downscale)
        sparse_flow, mask_matches = utils.create_mean_map_ab_ba(sparse_flow, mask_matches, sparse_flow_ba,
                                                                mask_matches_ba, args.downscale)

    sparse_flow_img = utils.flow_to_image(sparse_flow)
    sk.io.imsave('tmp_sparse_flow_downscaled.png', sparse_flow_img)

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():

            sparse_flow_ph = tf.placeholder(tf.float32, shape=(None, sparse_flow.shape[0], sparse_flow.shape[1], 2),
                                            name='sparse_flow_ph')
            matches_mask_ph = tf.placeholder(tf.float32, shape=(None, sparse_flow.shape[0], sparse_flow.shape[1], 1),
                                             name='matches_mask_ph')
            edges_ph = tf.placeholder(tf.float32, shape=(None, sparse_flow.shape[0], sparse_flow.shape[1], 1),
                                      name='edges_ph')

            forward_model = model.getNetwork(sparse_flow_ph, matches_mask_ph, edges_ph, reuse=False)

            saver_keep = tf.train.Saver(tf.all_variables(), max_to_keep=0)

            sess = tf.Session()

            saver_keep.restore(sess, args.model_filename)

            print("Performing inference...")
            prediction = sess.run(forward_model, feed_dict={
                sparse_flow_ph: np.expand_dims(sparse_flow, axis=0),
                matches_mask_ph: np.reshape(mask_matches, [1, mask_matches.shape[0], mask_matches.shape[1], 1]),
                edges_ph: np.expand_dims(np.expand_dims(edges, axis=0), axis=3),
            })
            print("Upscaling...")
            upscaled_pred = sk.transform.resize(prediction[0], [args.img_height, args.img_width, 2],
                                                preserve_range=True, order=3)

            if not os.path.isdir('tmp_interponet'):
                os.makedirs('tmp_interponet')
            io_utils.save_flow2file(upscaled_pred, filename='tmp_interponet/out_no_var.flo')

            # But if out_dir is provided as a full path to output flow, keep it
            unique_name = os.path.basename(args.img1_filename)[:-4]
            if os.path.basename(args.out_filename) is not None:
                parent_folder_name = os.path.dirname(args.out_filename)
            else:
                parent_folder_name = 'interponet_one_inference_{}'.format(os.path.basename(args.img1_filename)[:-4]) \
                    if args.new_par_folder is None else args.new_par_folder
                print("parent_folder_name: '{}".format(parent_folder_name))
                if not os.path.isdir(parent_folder_name):
                    os.makedirs(parent_folder_name)
            out_flo_path = os.path.join(parent_folder_name, unique_name + '_flow.flo')
            print("out_flo_path: '{}'".format(out_flo_path))
            print("Variational post Processing...")
            utils.calc_variational_inference_map(args.img1_filename, args.img2_filename,
                                                 'tmp_interponet/out_no_var.flo', out_flo_path, 'sintel')

            # Read outputted flow to compute metrics
            pred_flow = io_utils.read_flow(out_flo_path)

            # Save image visualization of predicted flow (Middlebury colour coding)
            if args.save_image:
                flow_img = utils.flow_to_image(pred_flow)
                out_path_full = os.path.join(parent_folder_name, unique_name + '_viz.png')
                sk.io.imsave(out_path_full, flow_img)

            # Compute metrics
            if args.compute_metrics and args.gt_flow is not None:
                metrics = utils.compute_all_metrics(pred_flow, args.gt_flow, occ_mask=args.occ_mask,
                                                    inv_mask=args.inv_mask)
                metrics_str = utils.get_metrics(metrics, flow_fname=unique_name)
                print(metrics_str)  # print to stdout (use test_batch to log several images' error metrics to file)

            # Remove temporal directory and everything in it
            shutil.rmtree('tmp_interponet')


def test_batch(args):
    # Height, width after downsampling (original scheme slightly crops sparse flow, edges + matches mask)
    # See 'downscale_all' for more details
    height_downsample, width_downsample = get_downsampled_image_size(og_height=args.img_height, og_width=args.img_width,
                                                                     downscale_factor=args.downscale)

    # Allocate network w. placeholders (once)
    sparse_flow_ph = tf.placeholder(tf.float32, shape=(None, height_downsample, width_downsample, 2),
                                    name='sparse_flow_ph')
    mask_matches_ph = tf.placeholder(tf.float32, shape=(None, height_downsample, width_downsample, 1),
                                     name='mask_matches_ph')
    edges_ph = tf.placeholder(tf.float32, shape=(None, height_downsample, width_downsample, 1), name='edges_ph')

    forward_model = model.getNetwork(sparse_flow_ph, mask_matches_ph, edges_ph, reuse=False)

    saver_keep = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    with tf.Session() as sess:
        saver_keep.restore(sess, args.model_filename)

        # Read and process the resulting list, one element at a time
        with open(args.img1_filename, 'r') as input_file:
            path_list = input_file.readlines()

            # Initialise some auxiliar variables to track metrics
            add_metrics = np.array([])
            # Auxiliar counters for metrics
            not_occluded_count = 0
            not_disp_S0_10_count = 0
            not_disp_S10_40_count = 0
            not_disp_S40plus_count = 0
            if args.log_metrics2file:
                basefile = os.path.basename(args.img1_filename)
                logfile = basefile.replace('.txt', '_metrics.log')
                logfile_full = os.path.join(args.out_filename, logfile) if args.new_par_folder is None else \
                    os.path.join(args.out_filename, args.new_par_folder, logfile)
                if not os.path.isdir(os.path.dirname(logfile_full)):
                    os.makedirs(os.path.dirname(logfile_full))
                # Open file (once)
                logfile = open(logfile_full, 'w')
                if args.new_par_folder is not None:
                    now = datetime.datetime.now()
                    date_now = now.strftime('%d-%m-%y_%H-%M-%S')
                    # Record header for the file detailing 'experiment' string (new_par_folder)
                    header_str = "Today is {}\nOpening and logging experiment '{}'\n Written to file: '{}'\n".format(
                        date_now, args.new_par_folder, logfile_full)
                    logfile.write(header_str)

            for img_idx in range(len(path_list)):
                # Read + pre-process files
                # Each line is split into a list with N elements (separator: blank space (" "))
                path_inputs = path_list[img_idx][:-1].split(' ')  # remove \n at the end of the line!
                assert 4 <= len(path_inputs) <= 6, (
                    'More paths than expected. Expected: I1+I2+edges+matches(4), I1+I2+edges+matches+gtflow(5),'
                    ' I1+I2+edges+matches+backward_matches(5) or I1+I2+edges+matches+backward_matches+gtflow(6)')
                # Common operations to all input sizes
                # Read input frames (for variational)
                img1_filename = path_inputs[0]
                img2_filename = path_inputs[1]
                edges_fname = path_inputs[2]
                matches_fname = path_inputs[3]
                # Important: read and THEN pad if needed
                # Load edges
                edges = io_utils.load_edges_file(edges_fname, width=args.img_width, height=args.img_height)

                # Load matching file (+initialise sparse flow)
                sparse_flow, mask_matches = io_utils.load_matching_file(matches_fname, width=args.img_width,
                                                                        height=args.img_height)

                # tmp to get more info
                print("FF semi-dense initial flow has matches in {:.2%} of pixels".format(np.sum(mask_matches != -1) /
                                                                                          np.prod(mask_matches.shape)))
                # downscale
                sparse_flow, mask_matches, edges = utils.downscale_all(sparse_flow, mask_matches, edges, args.downscale)
                # I1 + I2 + edges + matches
                if len(path_inputs) == 4:
                    print("Input type is: I1+I2+edges+matches, so no error metrics can be computed")
                    args.compute_metrics = False
                elif len(path_inputs) == 5:
                    # I1+I2+edges+matches+ba_matches
                    if args.ba_matches_filename is not None:
                        ba_matches_filename = path_inputs[4]
                        sparse_flow_ba, mask_matches_ba = io_utils.load_matching_file(ba_matches_filename,
                                                                                      width=args.img_width,
                                                                                      height=args.img_height)
                        # downscale ba
                        sparse_flow_ba, mask_matches_ba, _ = utils.downscale_all(sparse_flow_ba, mask_matches_ba, None,
                                                                                 args.downscale)
                        sparse_flow, mask_matches = utils.create_mean_map_ab_ba(sparse_flow, mask_matches,
                                                                                sparse_flow_ba, mask_matches_ba,
                                                                                args.downscale)

                    # I1+I2+edges+matches+gt_flow
                    else:
                        if args.compute_metrics:
                            gt_flow = io_utils.read_flow(path_inputs[4])
                            occ_mask = None
                            inv_mask = None

                elif len(path_inputs) == 6:
                    # I1+I2+edges+matches+ba_matches+gt_flow
                    if args.ba_matches_filename is not None:
                        ba_matches_filename = path_inputs[4]
                        sparse_flow_ba, mask_matches_ba = io_utils.load_matching_file(ba_matches_filename,
                                                                                      width=args.img_width,
                                                                                      height=args.img_height)
                        # downscale ba
                        sparse_flow_ba, mask_matches_ba, _ = utils.downscale_all(sparse_flow_ba, mask_matches_ba, None,
                                                                                 args.downscale)
                        sparse_flow, mask_matches = utils.create_mean_map_ab_ba(sparse_flow, mask_matches,
                                                                                sparse_flow_ba, mask_matches_ba,
                                                                                args.downscale)
                        if args.compute_metrics:
                            gt_flow = io_utils.read_flow(path_inputs[5])
                            occ_mask = None
                            inv_mask = None
                        else:
                            print("Warning: gt_flow provided but compute_metrics=False"
                                  "Won't compute error, please change the flag's value")
                            gt_flow = None
                            occ_mask = None
                            inv_mask = None

                    # I1+I2+edges+matches+gt_flow+occ
                    else:
                        if args.compute_metrics:
                            gt_flow = io_utils.read_flow(path_inputs[4])
                            occ_mask = sk.io.imread(path_inputs[5])
                            inv_mask = None
                        else:
                            print("Warning: inputted gt_flow and occlusion mask but compute_metrics=False!"
                                  "Won't compute error, please change the flag's value")
                            gt_flow = None
                            occ_mask = None
                            inv_mask = None

                elif len(path_inputs) == 7:
                    # I1+I2+edges+matches+ba_matches+gt_flow
                    if args.ba_matches_filename is not None:
                        ba_matches_filename = path_inputs[4]
                        sparse_flow_ba, mask_matches_ba = io_utils.load_matching_file(ba_matches_filename,
                                                                                      width=args.img_width,
                                                                                      height=args.img_height)
                        # downscale ba
                        sparse_flow_ba, mask_matches_ba, _ = utils.downscale_all(sparse_flow_ba, mask_matches_ba, None,
                                                                                 args.downscale)
                        sparse_flow, mask_matches = utils.create_mean_map_ab_ba(sparse_flow, mask_matches,
                                                                                sparse_flow_ba, mask_matches_ba,
                                                                                args.downscale)
                        if args.compute_metrics:
                            gt_flow = io_utils.read_flow(path_inputs[5])
                            occ_mask = sk.io.imread(path_inputs[6])
                            inv_mask = None
                        else:
                            print("Warning: gt_flow and occ_mask provided but compute_metrics=False"
                                  "Won't compute error, please change the flag's value")
                            gt_flow = None
                            occ_mask = None
                            inv_mask = None

                    # I1 + I2 + edges + matches + gt_flow + occ + inv
                    else:
                        if args.compute_metrics:
                            gt_flow = io_utils.read_flow(path_inputs[4])
                            occ_mask = sk.io.imread(path_inputs[5])
                            inv_mask = sk.io.imread(path_inputs[6])
                        else:
                            print("Warning: inputted gt_flow, occlusions mask and invalid mask but compute_metrics"
                                  "=False! Won't compute error, please change the flag's value")
                            gt_flow = None
                            occ_mask = None
                            inv_mask = None

                elif len(path_inputs) == 8:
                    # I1+I2+edges+matches+ba_matches+gt_flow+occ+inv
                    if args.ba_matches_filename is not None:
                        ba_matches_filename = path_inputs[4]
                        sparse_flow_ba, mask_matches_ba = io_utils.load_matching_file(ba_matches_filename,
                                                                                      width=args.img_width,
                                                                                      height=args.img_height)
                        # downscale ba
                        sparse_flow_ba, mask_matches_ba, _ = utils.downscale_all(sparse_flow_ba, mask_matches_ba, None,
                                                                                 args.downscale)
                        sparse_flow, mask_matches = utils.create_mean_map_ab_ba(sparse_flow, mask_matches,
                                                                                sparse_flow_ba, mask_matches_ba,
                                                                                args.downscale)
                        if args.compute_metrics:
                            gt_flow = io_utils.read_flow(path_inputs[5])
                            occ_mask = sk.io.imread(path_inputs[6])
                            inv_mask = sk.io.imread(path_inputs[7])
                        else:
                            print("Warning: gt_flow and occ_mask provided but compute_metrics=False"
                                  "Won't compute error, please change the flag's value")
                            gt_flow = None
                            occ_mask = None
                            inv_mask = None

                else:
                    raise ValueError("Unexpected number of inputs, options are: (4) I1+I2+edges+matches,"
                                     "(5) I1+I2+edges+matches+gt_flow, (5) I1+I2+edges+matches+ba_matches,"
                                     "(6) I1+I2+edges+matches+ba_matches+gt_flow, (6) I1+I2+edges+matches+gt_flow+occ, "
                                     "(7) I1+I2+edges+matches+ba_matches+gt_flow+occ, "
                                     "(7) I1+I2+edges+matches+gt_flow+occ+inv or "
                                     "(8) I1+I2+edges+matches+ba_matches+gt_flow+occ+inv")
                # If at some moment we want to pad with zeros instead of cropping, use here utils.adapt_x

                # Compute OF and run variational post-processing
                with tf.device('/gpu:0'):
                    with tf.Graph().as_default():
                        prediction = sess.run(forward_model, feed_dict={
                            sparse_flow_ph: np.expand_dims(sparse_flow, axis=0),
                            mask_matches_ph: np.reshape(mask_matches, [1, mask_matches.shape[0],
                                                                       mask_matches.shape[1], 1]),
                            edges_ph: np.expand_dims(np.expand_dims(edges, axis=0), axis=3),
                        })
                        # Upscale prediction
                        upscaled_pred = sk.transform.resize(prediction[0], [args.img_height, args.img_width, 2],
                                                            preserve_range=True, order=3)

                        if not os.path.isdir('tmp_interponet'):
                            os.makedirs('tmp_interponet')
                        io_utils.save_flow2file(upscaled_pred, filename='tmp_interponet/out_no_var.flo')

                        parent_folder_name = path_inputs[0].split('/')[-2] if args.new_par_folder is None \
                            else args.new_par_folder
                        unique_name = os.path.basename(path_inputs[0])[:-4]
                        out_path_complete = os.path.join(args.out_filename, parent_folder_name)
                        if not os.path.isdir(out_path_complete):
                            os.makedirs(out_path_complete)

                        out_flo_path = os.path.join(out_path_complete, unique_name + '_flow.flo')

                        # Variational post Processing
                        utils.calc_variational_inference_map(img1_filename, img2_filename,
                                                             'tmp_interponet/out_no_var.flo', out_flo_path, 'sintel')

                        # Read outputted flow to compute metrics
                        pred_flow = io_utils.read_flow(out_flo_path)

                        # Save image visualization of predicted flow (Middlebury colour coding)
                        if args.save_image:
                            flow_img = utils.flow_to_image(pred_flow)
                            out_path_full = os.path.join(out_path_complete, unique_name + '_viz.png')
                            sk.io.imsave(out_path_full, flow_img)

                        if args.compute_metrics and gt_flow is not None:
                            # Compute all metrics
                            metrics, not_occluded, not_disp_s010, not_disp_s1040, not_disp_s40plus = \
                                utils.compute_all_metrics(pred_flow, gt_flow, occ_mask=occ_mask,
                                                          inv_mask=inv_mask)
                            final_str_formated = utils.get_metrics(metrics, flow_fname=unique_name)
                            if args.accumulate_metrics:
                                not_occluded_count += not_occluded
                                not_disp_S0_10_count += not_disp_s010
                                not_disp_S10_40_count += not_disp_s1040
                                not_disp_S40plus_count += not_disp_s40plus
                                # Update metrics array
                                current_metrics = np.hstack(
                                    (metrics['mangall'], metrics['stdangall'], metrics['EPEall'],
                                     metrics['mangmat'], metrics['stdangmat'], metrics['EPEmat'],
                                     metrics['mangumat'], metrics['stdangumat'], metrics['EPEumat'],
                                     metrics['S0-10'], metrics['S10-40'], metrics['S40plus']))
                                # Concatenate in one new row (if empty just initialises to current_metrics)
                                add_metrics = np.vstack(
                                    [add_metrics, current_metrics]) if add_metrics.size else current_metrics

                            if args.log_metrics2file:
                                logfile.write(final_str_formated)
                            else:  # print to stdout
                                print(final_str_formated)

            # Remove temporal directory and everything in it
            shutil.rmtree('tmp_interponet')


if __name__ == '__main__':
    # Parsing the parameters
    parser = argparse.ArgumentParser(description='Interponet inference (image or filelist)')
    parser.add_argument('--img1_filename', type=str, help='First image filename in the image pair (or path to txt file'
                                                          ' with all the paths',
                        default=None)
    parser.add_argument('--img2_filename', type=str, help='Second image filename in the image pair',
                        default=None)
    parser.add_argument('--edges_filename', type=str, help='Edges filename', default='example/frame_0001.dat')
    parser.add_argument('--matches_filename', type=str, help='Sparse matches filename',
                        default=None)
    parser.add_argument('--out_filename', type=str, help='Flow output filename (or path if .txt file is passed)',
                        default=None)

    parser.add_argument('--model_filename', type=str, help='Saved model parameters filename')
    parser.add_argument('--ba_matches_filename', type=str,
                        help='Sparse matches filename from Second image to first image')

    parser.add_argument('--img_width', type=int, help='Saved model parameters filename')
    parser.add_argument('--img_height', type=int, help='Saved model parameters filename')
    parser.add_argument('--downscale', type=int, help='Saved model parameters filename')

    parser.add_argument('--sintel', action='store_true', help='Use default parameters for sintel')

    parser.add_argument('--compute_metrics', type=io_utils.str2bool, required=False,
                        help='whether to compute error metrics or not (if True all available metrics are computed,'
                             ' check utils.py)', default=True)
    parser.add_argument('--accumulate_metrics', type=io_utils.str2bool, required=False,
                        help='for batch: whether to accumulate metrics to compute averages (excluding outliers: Inf,'
                             ' Nan) or not)', default=True)
    parser.add_argument('--log_metrics2file', type=io_utils.str2bool, required=False,
                        help='whether to log the metrics to a file instead of printing them to stdout', default=False,)
    parser.add_argument('--new_par_folder', type=str, required=False,
                        help='for batch inference, instead of creating a subfolder with the first file parent name,'
                             ' we assign a custom', default=None)
    parser.add_argument('--save_image', type=bool, required=False, help='whether to save an colour-coded image of the'
                                                                        ' predicted flow or not', default=True,)
    parser.add_argument('--gt_flow', type=str, required=False,
                        help='Path to ground truth flow so we can compute error metrics',
                        default=None,)
    parser.add_argument('--occ_mask', type=str, required=False,
                        help='Path to occlusions mask (1s indicate pixel is occluded, 0 otherwise)',
                        default=None,)
    parser.add_argument('--inv_mask', type=str, required=False,
                        help='Path to invalid mask with pixels that should not be considered when computing metrics = '
                             '1(invalid flow)',
                        default=None,)
    arguments = parser.parse_args()

    if arguments.sintel:
        if arguments.img_width is None:
            arguments.img_width = 1024
        if arguments.img_height is None:
            arguments.img_height = 436
        if arguments.downscale is None:
            arguments.downscale = 8
        if arguments.model_filename is None:
            arguments.model_filename = 'models/ff_sintel.ckpt'
    if arguments.img1_filename or arguments.img2_filename or arguments.edges_filename or arguments.matches_filename:
        # Use default
        print("Missing some required argument (img1, img2, edges or matches)")
        print("Will be using default testing image 'examples/frame_0001.png'")
        arguments.img1_filename = 'examples/frame_0001.png'
        arguments.img2_filename = 'examples/frame_0002.png'
        arguments.edges_filename = 'examples/frame_0001.dat'
        arguments.matches_filename = 'examples/frame_0001.txt'
        arguments.ba_matches_filename = 'examples_frame_0001_BA.txt'
        arguments.out_dirname = 'results'

    if os.path.basename(arguments.img1_filename).lower()[-3:] == 'txt':
        print("Testing all images from user-provided text file: '{}'".format(arguments.img1_filename))
        test_batch(arguments)
    else:
        print("Testing one single image with name: '{}'".format(arguments.img1_filename))
        test_one_image(arguments)
