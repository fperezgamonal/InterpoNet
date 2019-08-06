import tensorflow as tf
# TODO: should use TF 2.0 compat version to interpolate (but it is not available in the cluster...)
# import tensorflow.compat.v2 as tf_v2
# older tf.resize_* had a pretty nasty bug ==> https://github.com/tensorflow/tensorflow/issues/6720


def conv_layer(conv_input, name, shape, reuse=None, activation=tf.nn.elu):
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable('weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv_input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[shape[3]], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        return activation(bias, name=scope.name)


# TODO: do bilinear interpolation with tensorflow (like FlowNet2-tf) to remove skimage dependencies (problems in server)
def getNetwork(input_img, mask, edges, og_height, og_width, reuse=tf.AUTO_REUSE):
    images = tf.concat([input_img, mask, edges], axis=3)

    conv1 = conv_layer(conv_input=images, name='conv1', shape=[7, 7, 4, 32])
    conv2 = conv_layer(conv_input=conv1, name='conv2', shape=[7, 7, 32, 64])
    conv3 = conv_layer(conv_input=conv2, name='conv3', shape=[7, 7, 64, 128])
    conv4 = conv_layer(conv_input=conv3, name='conv4', shape=[7, 7, 128, 128])
    conv5 = conv_layer(conv_input=conv4, name='conv5', shape=[7, 7, 128, 128])
    conv6 = conv_layer(conv_input=conv5, name='conv6', shape=[7, 7, 128, 128])
    conv7 = conv_layer(conv_input=conv6, name='conv7', shape=[7, 7, 128, 256])
    conv8 = conv_layer(conv_input=conv7, name='conv8', shape=[7, 7, 256, 256])
    conv9 = conv_layer(conv_input=conv8, name='conv9', shape=[7, 7, 256, 256])
    conv10 = conv_layer(conv_input=conv9, name='conv10', shape=[7, 7, 256, 256])

    detour1 = conv_layer(conv_input=conv1, name='conv_out_1', shape=[7, 7, 32, 2], activation=tf.identity)
    detour2 = conv_layer(conv_input=conv2, name='conv_out_2', shape=[7, 7, 64, 2], activation=tf.identity)
    detour3 = conv_layer(conv_input=conv3, name='conv_out_3', shape=[7, 7, 128, 2], activation=tf.identity)
    detour4 = conv_layer(conv_input=conv4, name='conv_out_4', shape=[7, 7, 128, 2], activation=tf.identity)
    detour5 = conv_layer(conv_input=conv5, name='conv_out_5', shape=[7, 7, 128, 2], activation=tf.identity)
    detour6 = conv_layer(conv_input=conv6, name='conv_out_6', shape=[7, 7, 128, 2], activation=tf.identity)
    detour7 = conv_layer(conv_input=conv7, name='conv_out_7', shape=[7, 7, 256, 2], activation=tf.identity)
    detour8 = conv_layer(conv_input=conv8, name='conv_out_8', shape=[7, 7, 256, 2], activation=tf.identity)
    detour9 = conv_layer(conv_input=conv9, name='conv_out_9', shape=[7, 7, 256, 2], activation=tf.identity)
    detour10 = conv_layer(conv_input=conv10, name='conv_out_10', shape=[7, 7, 256, 2], activation=tf.identity)

    # NOTE: we wanted to do the resizing internally so it could be faster (on the GPU) but we found several problems:
    #   * Functions of the family tf.image.resize_... have a nasty bug ==>
    #   * This bug is fixed on TF 2.0 but we cannot install it on the server or locally ATM
    #   * This implies that we revert to resizing externally, probably on the CPU and hence slower
    #   * We cannot use skimage.transform.resize due to mismatch library versions
    #   * We finally use opencv's resize but need to add an extra transpose as it swaps the channels (width <==> height)
    # Bilinear interpolation to original size (BUG in V1)
    # flow = tf.image.resize_bicubic(detour10, tf.stack([og_height, og_width]), align_corners=True, half
    # Desireable (BUG FREE - V2)
    # flow = tf_v2.image.resize(detour10, tf.stack([og_height, og_width]), method=ResizeMethod.BILINEAR,
    #                          preserve_aspect_ratio=True)
    # argument 'half_pixel_centers' reduces the bug impact but is not available on TF 1.12 (cluster version)
    # flow = tf.image.resize_bicubic(detour10, tf.stack([og_height, og_width]), align_corners=True,)
    # half_pixel_centers=True) not available on 1.12 (cluster version) :(

    return detour10  # flow

