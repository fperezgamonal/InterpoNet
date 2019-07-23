import numpy as np
import tensorflow as tf
import skimage as sk
import utils
import io_utils
import model
import argparse
import os
import datetime


def test_one_image(args):
    # Load edges file
    print("Loading files...")
    edges = io_utils.load_edges_file(args.edges_filename, width=args.img_width, height=args.img_height)

    # Load matching file
    img, mask = io_utils.load_matching_file(args.matches_filename, width=args.img_width, height=args.img_height)

    # downscale
    print("Downscaling...")
    img, mask, edges = utils.downscale_all(img, mask, edges, args.downscale)

    if args.ba_matches_filename is not None:
        img_ba, mask_ba = io_utils.load_matching_file(args.ba_matches_filename, width=args.img_width,
                                                      height=args.img_height)

        # downscale ba
        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, args.downscale)
        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, args.downscale)

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():

            image_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 2), name='image_ph')
            mask_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 1), name='mask_ph')
            edges_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 1), name='edges_ph')

            forward_model = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)

            saver_keep = tf.train.Saver(tf.all_variables(), max_to_keep=0)

            sess = tf.Session()

            saver_keep.restore(sess, args.model_filename)

            print("Performing inference...")
            prediction = sess.run(forward_model,
                                  feed_dict={image_ph: np.expand_dims(img, axis=0),
                                             mask_ph: np.reshape(mask, [1, mask.shape[0], mask.shape[1], 1]),
                                             edges_ph: np.expand_dims(np.expand_dims(edges, axis=0), axis=3),
                                             })
            print("Upscaling...")
            upscaled_pred = sk.transform.resize(prediction[0], [args.img_height, args.img_width, 2],
                                                preserve_range=True, order=3)

            # io_utils.save_flow_file(upscaled_pred, filename='out_no_var.flo')
            # save_flow_file uses deprecated code
            io_utils.write_flow(upscaled_pred, filename='out_no_var.flo')

            print("Variational post Processing...")
            utils.calc_variational_inference_map(args.img1_filename, args.img2_filename, 'out_no_var.flo',
                                                 args.out_filename, 'sintel')


def test_batch(args):
    # Read first line to get image width and height from first image (assuming all have the same dimensions)
    with open(args.img1_filename, 'r') as input_file:
        first_line = input_file.readline()
    tmp_filename = first_line.split()[0]
    tmp_image = sk.io.imread(tmp_filename)
    height, width, _ = tmp_image.shape

    # Allocate network w. placeholders (once)
    image_ph = tf.placeholder(tf.float32, shape=(None, height, width, 2), name='image_ph')
    mask_ph = tf.placeholder(tf.float32, shape=(None, height, width, 1), name='mask_ph')
    edges_ph = tf.placeholder(tf.float32, shape=(None, height, width, 1), name='edges_ph')

    forward_model = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)

    saver_keep = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    with tf.Session() as sess:
        saver_keep.restore(sess, args.model_filename)

        # Read and process the resulting list, one element at a time
        with open(args.img1_filename, 'r') as input_file:
            path_list = input_file.readlines()

            if args.compute_metrics and args.accumulate_metrics:
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
                img1_fname = path_inputs[0]
                img2_fname = path_inputs[1]
                edges_fname = path_inputs[2]
                matches_fname = path_inputs[3]
                edges = io_utils.load_edges_file(edges_fname, width=width, height=height)

                # Load matching file
                img, mask = io_utils.load_matching_file(matches_fname, width=width, height=height)

                # downscale
                img, mask, edges = utils.downscale_all(img, mask, edges, args.downscale)
                # I1 + I2 + edges + matches
                if len(path_inputs) == 4:
                    print("Input type is: I1+I2+edges+matches, so no error metrics can be computed")
                    args.compute_metrics = False
                elif len(path_inputs) == 5:
                    # I1+I2+edges+matches+ba_matches
                    if args.ba_matches_filename is not None:
                        ba_matches_filename = path_inputs[4]
                        img_ba, mask_ba = io_utils.load_matching_file(ba_matches_filename, width=args.img_width,
                                                                      height=args.img_height)
                        # downscale ba
                        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, args.downscale)
                        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, args.downscale)

                    # I1+I2+edges+matches+gt_flow
                    else:
                        if args.compute_metrics:
                            print("Last input file is a path to a ground truth flow")
                            gt_flow = io_utils.read_flow(path_inputs[4])
                            occ_mask = None
                            inv_mask = None

                elif len(path_inputs) == 6:
                    # I1+I2+edges+matches+ba_matches+gt_flow
                    if args.ba_matches_filename is not None:
                        ba_matches_filename = path_inputs[4]
                        img_ba, mask_ba = io_utils.load_matching_file(ba_matches_filename, width=args.img_width,
                                                                      height=args.img_height)
                        # downscale ba
                        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, args.downscale)
                        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, args.downscale)
                        if args.compute_metrics:
                            print("Last input file is a path to a ground truth flow")
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
                        img_ba, mask_ba = io_utils.load_matching_file(ba_matches_filename, width=args.img_width,
                                                                      height=args.img_height)
                        # downscale ba
                        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, args.downscale)
                        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, args.downscale)
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
                        img_ba, mask_ba = io_utils.load_matching_file(ba_matches_filename, width=args.img_width,
                                                                      height=args.img_height)
                        # downscale ba
                        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, args.downscale)
                        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, args.downscale)
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

                # Compute OF and run variational post-processing
                with tf.device('/gpu:0'):
                    with tf.Graph().as_default():
                        prediction = sess.run(forward_model, feed_dict={image_ph: np.expand_dims(img, axis=0),
                                                                        mask_ph: np.reshape(mask, [1, mask.shape[0],
                                                                                                   mask.shape[1], 1]),
                                                                        edges_ph: np.expand_dims(
                                                                            np.expand_dims(edges, axis=0), axis=3),
                                                                        })
                        # Upscale prediction
                        upscaled_pred = sk.transform.resize(prediction[0], [args.img_height, args.img_width, 2],
                                                            preserve_range=True, order=3)

                        # io_utils.save_flow_file(upscaled_pred, filename='out_no_var.flo')
                        # save_flow_file uses deprecated code
                        io_utils.write_flow(upscaled_pred, filename='out_no_var.flo')

                        parent_folder_name = path_inputs[0].split('/')[-2] if args.new_par_folder is None \
                            else args.new_par_folder
                        unique_name = path_inputs[0].split('/')[-1][:-4]
                        out_path_complete = os.path.join(args.out_filename, parent_folder_name)
                        if not os.path.isdir(out_path_complete):
                            os.makedirs(out_path_complete)

                        out_flo_path = os.path.join(out_path_complete, unique_name, '_flow.flo')

                        # Variational post Processing
                        utils.calc_variational_inference_map(img1_fname, img2_fname, 'out_no_var.flo', out_flo_path,
                                                             'sintel')

                        # Read outputted flow to compute metrics
                        pred_flow = io_utils.read_flow(out_path_full)
                        # Save image visualization of predicted flow (Middlebury colour coding)
                        if args.save_image:
                            flow_img = utils.flow_to_image(pred_flow)
                            out_path_full = os.path.join(out_path_complete, unique_name + '_viz.png')
                            sk.io.imsave(out_path_full, flow_img)

                        if args.compute_metrics and args.gt_flow is not None:
                            # Compute all metrics
                            metrics, not_occluded, not_disp_s010, not_disp_s1040, not_disp_s40plus = \
                                utils.compute_all_metrics(pred_flow, gt_flow, occ_mask=occ_mask, inv_mask=inv_mask)
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


if __name__ == '__main__':
    # Parsing the parameters
    parser = argparse.ArgumentParser(description='Interponet inference (image or filelist)')
    parser.add_argument('--img1_filename', type=str, help='First image filename in the image pair (or path to txt file'
                                                          ' with all the paths',
                        default='example/frame_0001.png')
    parser.add_argument('--img2_filename', type=str, help='Second image filename in the image pair',
                        default='example/frame_0002.png')
    parser.add_argument('--edges_filename', type=str, help='Edges filename', default='example/frame_0001.dat')
    parser.add_argument('--matches_filename', type=str, help='Sparse matches filename',
                        default='example/frame_0001.txt')
    parser.add_argument('--out_filename', type=str, help='Flow output filename (or path if .txt file is passed)',
                        default='example/')

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
                        default='data/samples/sintel/frame_00186.flo',)
    parser.add_argument('--occ_mask', type=str, required=False,
                        help='Path to occlusions mask (1s indicate pixel is occluded, 0 otherwise)',
                        default='data/samples/sintel/frame_00186_occ_mask.png',)
    parser.add_argument('--inv_mask', type=str, required=False,
                        help='Path to invalid mask with pixels that should not be considered when computing metrics = '
                             '1(invalid flow)',
                        default='data/samples/sintel/frame_00186_inv_mask.png',)
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

    if os.path.basename(arguments.img1_filename).lower()[-3:] == 'txt':
        print("Testing all images from user-provided text file: '{}'".format(arguments.img1_filename))
        test_batch(arguments)
    else:
        print("Testing one single image with name: '{}'".format(arguments.img1_filename))
        test_one_image(arguments)
