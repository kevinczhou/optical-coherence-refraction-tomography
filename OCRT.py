from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import cv2
from scipy.fftpack import fft, fftshift, ifft, ifftshift
import scipy.io
import scipy.interpolate
import os
from time import time


class OCRT2D:

    def __init__(self, sample_id, save_directory):
        # sample_id: a string identifying the sample (basename of the files);
        # save_directory: where to save the tf graph after optimization;
        # set hyperparameters here or after running the constructor, and then run load_data_and_resolve_constants, which
        # may create new variables that depend on the instance variables; finally, run build_graph, which creates the tf
        # variables and operations

        self.sample_id = sample_id
        self.save_directory = save_directory + sample_id
        self.tube_diameter = 1.108516  # mm
        self.Ascan_numpix = 2048  # how many pixels in one A-scan originally?
        self.depth = 2.219226700101120  # maximum imaging depth in air in mm
        self.n_back = 1.342  # immersion index
        self.wall_thickness = None  # used in parametric optimization
        self.ths = None  # angles in degrees
        self.step = None  # rk4 step size
        self.numz = None  # number of rk4 steps to take
        self.numy = 1  # number of B-scans per vol (out of plane dimension) used for recon; set to 1, but in the future
        # for 3D this would be >1
        self.numx_downsamp = None  # number of rays after downsampling
        self.B_upsample = 2  # number of times to laterally upsample Bscan
        self.downsample = 8  # integer factor by which to downsample in z
        self.downsample_x = 10  # int factor to downsample in x
        self.valid_Ascans = None  # some A-scans invalid due to downsampling
        self.x_max = None  # num of A-scans after upsampling post downsampling
        self.z_max = None  # len of A-scans after upsampling post downsampling
        self.use_Bweights_for_bp = None  # eg, to remove the central reflection
        self.dy = None  # spacing in xy in physical distance (not pixels)
        self.angle_step = None  # step in degrees between each angle
        self.use_multires = None  # partway through optimization, change pixel count for reconstruction
        self.PSFconv_format = 'gaussian'  # 'arbitrary', 'gaussian', or both ('arbitrarygaussian')
        self.reduced_shape = None  # shape of filtered/unfiltered B-scans prior to flattening
        self.sx = 23  # pixel width of convolution kernel (full width)
        self.sz = 1  # pixel height of convolution kernel (full width)
        self.outer_radius = None
        self.Bscan_dims = None
        self.outer_index = 0.11047276  # relative index values determined previously empirically
        self.inner_index = -0.11068416
        self.numiter = 200  # number of optimization iterations
        self.size_factor_ = 8  # how many times more pixels on a side for the reconstruction compared to the index map;
        # the underscore version is the static version, necessary for defining variables; the non-underscore version is
        # a placeholder, for multires optimization
        self.TV_reg_coeff = None  # the regularization coefficient for the total variation regularization of the
        # nonparametric portion of the index map (this one will be a tf.placeholder)
        self.TV_reg_coeff_ = 8e-5  # this is the initializing value
        self.shift_reg_coeff = 10.  # coeff for penalty on spatial shifts
        self.tube_radius_change_reg_coeff = 0  # penalty for change in radius
        self.imsize = 256  # size of RI map images to be plotted during training; irrelevant to the actual optimization
        self.num_y_pixels = 10  # for the reconstruction; this would be of consequence in the future when we implement
        # 3D reconstruction
        self.batchsize = None  # by default, same as len(ths)
        self.rotdir = -1  # -1 or 1, depending on direction of rotation
        self.numneigh_s = 7  # the size of the neighborhood that contributes to the index approximation (odd number)
        self.angle_membership = None  # angle membership of each point in Bscan
        self.Bweights = None  # same size as Bscans
        self.external_TV_mask = None
        self.internal_mask = None
        self.freq_filter = 'sqrt'  # which filter to use prior to backprojection?

        # binary settings:
        self.infer_backprojection_filter = False  # after registration, optimize for filter, holding all else fixed?
        self.include_attenuation = True  # whether to include attenuation in the forward model
        self.use_reflectance_model = True  # optimize general backreflectance model (ops will be in place regardless)
        self.discret_levels = 30  # if using reflectance model angle modulation (e.g., lambertian model),
        # how many discretization levels for the angles?
        self.use_spatial_shifts = True  # optimize Bscan spatial shifts in xz
        self.use_parametric_optimization = False  # i.e., tube parameters
        self.switch_to_nonparametric = False  # if you want to switch from tube parameters to general parameterization
        self.switch_iter_nonparametric = 200  # switch at what iteration?
        self.use_multires = False  # switch recon resolution part way through
        self.switch_iter = 250  # at what iteration to change recon res
        self.final_size_factor = 8  # final recon size (as a multiple of index map size; e.g., if the index map is of
        # size (128, 128), and final_size_factor is 4, then the recon size is (512, 512));
        # note that with these "switch" options must be implemented by the script that uses this class;
        self.use_Bweights = False  # Bweights is a tensor the same size as the Bscan data that gives a weight for each
        # pixel
        self.use_Bweights_for_bp = False  # specifically whether to use Bweights when backprojecting
        self.use_gpu = False
        self.stop_gradient_projection = True  # stop the gradient before the ray-gathering step; must be False for
        # filter optimization
        self.use_interp_projection = True  # backproject to nearest neighbor?
        self.normalize_across_whole_dataset = False  # for backproj; if false, then first normalize within B-scan, then
        # across whole dataset; if true, may produce streak artifacts due to ray crossing, but is less expesnive
        self.turn_off_all_normalization = False  # if true, recon is not normalized, regardless of
        # self.normalize_across_whole_dataset
        self.use_blur_Bscan_in_forward_pred = True

        # trainable variables (but not necessarily trained by default)
        self.A = None  # index kernel amplitudes to be defined elsewhere
        self.A_nonparametric = 0  # 0: initialize flat index on top of parametric index map, or supply an array
        self.A_parametric = None
        self.sig = .006  # index kernel widths
        self.xz0 = None
        self.xz_delta_init = None  # initial xz_delta (for regularization)
        self.xz_delta = None  # xz_delta after optimization
        self.y_delta = None
        self.recon_offset = .15  # single number to add to reconstruction
        self.PSF_general = None  # if doing arbitrary PSF optimization
        self.sigx = 1.3  # sigmax of gaussian kernel
        self.sigz = .5  # sigmaz of gaussian kernel
        self.h = .1  # height of gaussian kernel
        self.attenfact = .02  # attenuation factor
        self.attenexp = 3.5  # attenuation exponent
        self.angle_mod = None  # for arbitrary backreflection model; infer the angle-dependent backreflection model
        self.fbp_filters = None  # if optimizing filters after registration

        # for full rayprop with a kernel-parameterized index distribution:
        self.rotmats = None
        self.numgauss_s = None  # number of gaussian kernels
        self.Xc = None  # coordinates of the gaussian kernel centers
        self.Zc = None
        self.Xr = None  # spatial window kernels within which to consider
        self.Zr = None

        self.graph_built = False  # has the graph been built yet?

    def load_data_and_resolve_constants(self, data_directory):
        # 1. load data based on self.sample_id
        # 2. derive instance variables that depend on others (e.g., the ones defined in the constructor)

        start = time()
        self.downsample_x *= self.B_upsample
        self.step = 1.0 / self.Ascan_numpix * self.downsample

        # load from file
        filename = data_directory + self.sample_id + '_data.mat'
        data = scipy.io.loadmat(filename)
        Bscan_dims = data['Bscan_dims'][0]
        Bscans = data['Bscans']
        xzcoords = data['xzcoords']
        L = data['L'][0, 0]
        self.outer_radius = data['radii'][0, 0]
        mindists_ = data['mindists']
        inds_nonzero_ = data['window']
        self.xz_delta_init = data['tweaks']
        self.xz0 = data['center'][0]
        self.Bweights = data['weights'].astype(np.float32)

        if len(Bscans.shape) == 3:
            # if there is no volume, since matlab can't store trailing
            # singletons, expand dims
            Bscans = np.expand_dims(Bscans, -1)

        max_angle, self.angle_step = 360, 360. / Bscans.shape[0]
        self.ths = np.arange(0, max_angle, self.angle_step).astype(np.float32)

        self.numgauss_s = inds_nonzero_.shape[0]
        surround_dists_ = mindists_ * inds_nonzero_
        inner_dists_ = mindists_ * (1 - inds_nonzero_)
        surround_dists_ = surround_dists_.flatten()
        inner_dists_ = inner_dists_.flatten()
        self.y_delta = np.zeros(len(self.xz_delta_init), dtype=np.float32)
        xzcoords = .5 + (xzcoords - .5)
        self.dy = xzcoords[0, 1, 0] - xzcoords[0, 0, 0]  # spacing in x & y same

        # generate interpolation matrix
        x_left = np.arange(self.downsample_x)
        x_right = self.downsample_x - x_left
        z_up = np.arange(self.downsample)
        z_down = self.downsample - z_up

        normalize = self.downsample * self.downsample_x * 1.0
        DR = np.expand_dims(x_left, -1) * z_up.T / normalize
        UL = np.expand_dims(x_right, -1) * z_down.T / normalize
        DL = np.expand_dims(x_right, -1) * z_up.T / normalize
        UR = np.expand_dims(x_left, -1) * z_down.T / normalize
        self.interp_weights = np.vstack([UL.flatten(), UR.flatten(), DL.flatten(), DR.flatten()]).T
        self.interp_weights = self.interp_weights[None, None, None]

        # Bscan data stack:
        self.Bscans_filtered_np = np.copy(Bscans)
        self.Bscans_unfiltered_np = np.copy(Bscans)

        # upsample A-scans:
        if self.B_upsample != 1:
            sampling = np.arange(Bscans.shape[2])
            upsampling = np.arange(Bscans.shape[2] * self.B_upsample) * 1. / self.B_upsample
            # throw out extrapolated positions:
            upsampling = upsampling[:-self.B_upsample + 1]
            B_interpolator = scipy.interpolate.interp1d(sampling, Bscans, kind='cubic', axis=2)

            self.Bscans_filtered_np = B_interpolator(upsampling)
            self.Bscans_unfiltered_np = np.copy(self.Bscans_filtered_np)
            Bscan_dims[1] = self.Bscans_filtered_np.shape[2]

        self.x_max = (np.int32(Bscan_dims[1] / self.downsample_x) - 1) * self.downsample_x
        self.z_max = (np.int32(Bscan_dims[0] / self.downsample) - 1) * self.downsample

        self.Bscans_filtered_np = self.Bscans_filtered_np[:, :self.z_max, :self.x_max]
        self.Bscans_unfiltered_np = self.Bscans_unfiltered_np[:, :self.z_max, :self.x_max]

        # for all lateral positions for all angles
        self.valid_Ascans = np.arange(0, self.x_max + self.downsample_x,
                                      self.downsample_x, dtype=np.int32)
        self.valid_Ascans = np.tile(self.valid_Ascans, [len(self.ths), 1])

        self.numx_downsamp = np.int32(Bscan_dims[1] / self.downsample_x)
        self.numx = Bscan_dims[1]
        self.numz = np.int32(Bscan_dims[0] / self.downsample)

        # filter
        self.Bscans_filtered_np = self.filter_Bscans(self.Bscans_filtered_np)

        # flatten
        # save this shape before flattening:
        self.reduced_shape = self.Bscans_filtered_np.shape
        self.Bscans_filtered_np = self.Bscans_filtered_np.flatten()
        self.Bscans_unfiltered_np = self.Bscans_unfiltered_np.flatten()

        # load B-scan weights
        if self.use_Bweights or self.infer_backprojection_filter:
            if len(self.Bweights.shape) == 3:
                # because matlab doesn't allow trailing singleton dimensions
                self.Bweights = self.Bweights[:, :, :, None]
            if self.Bweights.shape[0] == 1:
                self.Bweights = np.tile(self.Bweights, [len(self.ths), 1, 1, 1])
            if self.B_upsample != 1:
                Bweights_interpolator = scipy.interpolate.interp1d(sampling, self.Bweights, kind='nearest', axis=2)
                self.Bweights = Bweights_interpolator(upsampling)

            self.Bweights = self.Bweights[:, :self.z_max, :self.x_max]
            self.Bweights = self.Bweights / self.Bweights.sum()
            self.Bweights = self.Bweights.flatten()
        else:
            self.Bweights = None

        if self.infer_backprojection_filter:
            # Bweights are always needed for filter inference
            self.Bweights_for_filter_inference = np.copy(self.Bweights)
        if not self.use_Bweights:
            self.Bweights = None

        self.angle_membership = (np.ones(np.hstack([[1], self.reduced_shape[1:]]), dtype=np.float32) *
                                 self.ths[:, None, None, None])
        self.angle_membership = self.angle_membership.flatten()

        # masks for index distribution
        self.external_TV_mask = 1 - np.exp(-np.reshape(surround_dists_, [self.numgauss_s, self.numgauss_s]) / .0005)
        self.internal_mask = 1 - np.exp(-np.reshape(inner_dists_, (self.numgauss_s, self.numgauss_s)) / .0005)
        self.external_TV_mask = self.external_TV_mask.astype(np.float32)
        self.internal_mask = self.internal_mask.astype(np.float32)

        self.x_init = np.zeros((len(self.ths), self.numx, 2), dtype=np.float32)
        if self.B_upsample != 1:
            xzcoords_interpolator = scipy.interpolate.interp1d(sampling, xzcoords, kind='linear', axis=1)
            xzcoords = xzcoords_interpolator(upsampling).astype(np.float32)
        self.x_init[:, :, 0] = (xzcoords[:, :, 0] - .5) * L + .5  # lateralscale
        # the first element from tf.scan will be the first + step:
        self.z_init = xzcoords[:, :, 1] - self.step

        # index kernel sampling window:
        half = (self.numneigh_s - 1) // 2
        self.Xr = np.arange(-half, half + 1)
        self.Zr = np.arange(-half, half + 1)
        self.Xr, self.Zr = np.meshgrid(self.Xr, self.Zr)
        self.Xr = self.Xr.flatten()[None]
        self.Zr = self.Zr.flatten()[None]

        # index kernel coordinates:
        self.Xc = np.linspace(0, 1, self.numgauss_s, dtype=np.float32)
        self.Zc = np.linspace(0, 1, self.numgauss_s, dtype=np.float32)
        self.Xc, self.Zc = np.meshgrid(self.Xc, self.Zc)
        self.Xc = self.Xc.flatten()
        self.Zc = self.Zc.flatten()

        print('data loaded: ' + str(time() - start) + ' sec')

    def build_graph(self, intra=0, inter=0):
        # creates all variables and ops;
        # intra sets the intra_op_parallelism_threads parameter in the tf.ConfigProto if using cpu rather than gpu
        # inter sets the inter_op_parallelism_threads parameter

        start = time()
        if self.graph_built:
            raise Exception('graph is already built')

        if self.use_gpu:
            self.sess = tf.InteractiveSession()
        else:
            config = tf.ConfigProto(device_count={'GPU': 0},
                                    intra_op_parallelism_threads=intra,
                                    inter_op_parallelism_threads=inter)
            self.sess = tf.InteractiveSession(config=config)

        self.create_tf_variables()
        self.create_losses()
        self.create_train_op()
        self.saver = tf.train.Saver()  # NOTE: saver only saves variables defined up until this line is run!
        self.modify_loss_and_train_op_and_initialize()

        self.graph_built = True
        print('graph built: ' + str(time() - start) + ' sec')

    def create_tf_variables(self):
        # this function defines all variables, placeholders, and constants based on the current instance variable
        # settings; be sure to set all desired hyperparameters beyond the constructor; i.e., run
        # load_data_and_constants, and change any constants desired thereafter; no need to call this function directly,
        # use the build_graph function

        if self.batchsize is not None:
            self.batchsize = tf.placeholder_with_default(self.batchsize, shape=None, name='batch_size')
        else:  # by default, batch is same size as total number of angles
            self.batchsize = tf.placeholder_with_default(len(self.ths), shape=None, name='batch_size')

        with tf.name_scope('index_parameters'):
            if np.size(self.sig) == 1:
                self.sig = tf.Variable(self.sig * np.ones(self.numgauss_s ** 2), dtype=np.float32, name='kernel_width')
            elif np.size(self.sig) == self.numgauss_s ** 2:
                self.sig = tf.Variable(self.sig.flatten(), dtype=np.float32, name='kernel_width')
            else:
                raise Exception('sig must be a number or array matching the size of the index map')
            if np.size(self.A_nonparametric) == 1:
                self.A_nonparametric = tf.Variable(self.A_nonparametric * np.ones(self.numgauss_s ** 2),
                                                   dtype=np.float32,
                                                   name='general_parameterization')
            elif np.size(self.A_nonparametric) == self.numgauss_s ** 2:
                self.A_nonparametric = tf.Variable(self.A_nonparametric.flatten(), dtype=np.float32,
                                                   name='general_parameterization')
            else:
                raise Exception('A_nonparametric must be a number or an array matching the size of the index map')
            wall_thickness_mm = (self.tube_diameter - 736.5 * self.depth / self.Ascan_numpix) / 2
            self.wall_thickness = wall_thickness_mm * self.n_back / self.depth
            self.A_parametric = tf.Variable([self.inner_index, self.outer_index], name='geometric_parameterization')
            self.A = tf.add(self.A_nonparametric, self.get_parametric_indexdist(self.A_parametric),
                            name='total_parameterization')

        with tf.name_scope('coordinate_based_parameters'):
            self.xz0 = tf.Variable(self.xz0, dtype=tf.float32, name='rotation_center')
            self.xz_delta = tf.Variable(self.xz_delta_init, dtype=tf.float32, name='Bscan_spatial_offsets')
            self.y_delta = tf.Variable(self.y_delta, dtype=tf.float32, name='Bscan_out_of_plane_offsets')

        with tf.name_scope('PSF_parameters'):
            self.sigx = tf.Variable(self.sigx, dtype=tf.float32, name='sigx')
            self.sigz = tf.Variable(self.sigz, dtype=tf.float32, name='sigz')
            self.h = tf.Variable(self.h, dtype=tf.float32, name='height_gaussian')
            self.PSF_general = np.zeros((self.sz, self.sx), dtype=np.float32)
            self.PSF_general[self.sz // 2, self.sx // 2] = 1.  # start with delta func
            self.PSF_general = tf.Variable(self.PSF_general, name='general')

        with tf.name_scope('reconstruction_related_parameters'):
            self.size_factor = tf.placeholder_with_default(self.size_factor_,
                                                           shape=(),
                                                           name='size_factor')
            self.recon_res = tf.to_int32(
                tf.stack([self.numgauss_s * self.size_factor,
                          self.numgauss_s * self.size_factor,
                          512], name='recon_res'))
            self.recon_offset = tf.Variable(self.recon_offset, dtype=tf.float32, name='reconstruction_offset_bias')
            self.attenfact = tf.Variable(self.attenfact, dtype=tf.float32, name='exponential_attenuation_factor')
            self.attenexp = tf.Variable(self.attenexp, dtype=tf.float32, name='exponential_attenuation_exponent')
            self.angle_mod = tf.Variable(np.ones(self.discret_levels, dtype=np.float32),
                                         name='angle_reflectance_profile')
            freqs = np.sqrt(np.abs(np.linspace(-1, 1, self.reduced_shape[1])))
            fbp_filters_ = np.tile(freqs[:, None], (1, self.reduced_shape[2]))
            self.fbp_filters = tf.Variable(fbp_filters_, dtype=tf.float32, name='fbp_filters')

        # other constants (not optimized):
        with tf.name_scope('initial_ray_positions'):
            # according to valid_Ascans, select only valid A-scans
            self.x_init = np.rollaxis(np.dstack([self.x_init[i, self.valid_Ascans[i]] for
                                                 i in range(self.batchsize.eval())]), 2, 0)
            self.z_init = np.vstack([self.z_init[i, self.valid_Ascans[i]] for i in range(self.batchsize.eval())])
            self.x_init = tf.placeholder_with_default(self.x_init, shape=None, name='x')
            self.z_init = tf.placeholder_with_default(self.z_init, shape=None, name='z')
        # rotation matrices
        rotmats = list()
        for theta in self.ths:
            theta *= (np.pi / 180)
            theta = theta.astype(np.float32)
            rotmats.append(np.array([[np.cos(theta), -self.rotdir * np.sin(theta)],
                                     [self.rotdir * np.sin(theta), np.cos(theta)]]))
        rotmats = np.rollaxis(np.dstack(rotmats), 2, 0).astype(np.float32)
        self.rotmats = tf.constant(rotmats, name='rotation_matrix')

        self.Bscans_filtered = tf.placeholder(tf.float32, shape=self.Bscans_filtered_np.shape,
                                              name='for_interpolation_filtered')
        self.Bscans_unfiltered = tf.placeholder(tf.float32, shape=self.Bscans_unfiltered_np.shape,
                                                name='for_interpolation_unfiltered')

        self.TV_reg_coeff = tf.placeholder_with_default(self.TV_reg_coeff_, shape=(),
                                                        name='index_spatial_regularization')

        # define these (initial) learning rates as tensors so that they can be annealed
        self.lr_parametric = tf.placeholder_with_default(.001, shape=(), name='learning_rate_for_parametric_optim')
        if self.use_parametric_optimization:
            self.lr_nonparametric = tf.placeholder_with_default(.0, shape=(),
                                                                name='learning_rate_for_nonparametric_optim')
        else:
            self.lr_nonparametric = tf.placeholder_with_default(.001, shape=(),
                                                                name='learning_rate_for_nonparametric_optim')

        # auxiliary tensors; for monitoring index distribution during training:
        xx = np.linspace(0, 1, self.imsize, dtype=np.float32)
        zz = np.linspace(0, 1, self.imsize, dtype=np.float32)
        [xc2, zc2] = np.meshgrid(xx, zz)
        xc2 = xc2.flatten()
        zc2 = zc2.flatten()
        xc2 = np.tile(xc2, (self.batchsize.eval(), 1))
        zc2 = np.tile(zc2, (self.batchsize.eval(), 1))
        RI = self.indexdist(xc2, zc2)[0][0]  # the indexdist function was designed to be used by ray propagation code,
        # so it contains extra stuff; here, we just want to see the index distribution
        self.RI = tf.reshape(RI * self.n_back, [self.imsize] * 2, name='index_image')

    def create_losses(self):
        # generate a list of losses, the sum of which is to be minimized

        # get the paths:
        with tf.name_scope('ray_propagation'):
            self.paths, self.dpathdz = self.integrate_scan_opl(
                self.x_init,
                self.z_init,
                return_derivs=True)
            self.paths = tf.identity(self.paths, name='trajectories')
            z_paths = self.paths[:, :, :, 0]  # for clarity
            x_paths = self.paths[:, :, :, 1]

        self.loss_terms = list()  # for nonparametric

        # data-dependent loss:
        with tf.name_scope('backprojection'):
            self.x_interp, self.z_interp = self.interp_rays_2d(x_paths, z_paths)
            loss_data = self.backprojection_tf(self.x_interp, self.z_interp)
        self.loss_terms.append(tf.multiply(10., loss_data, name='data_loss'))

        # give stuff names for saving/restoring later:
        with tf.name_scope('naming_OCRT_outputs'):
            self.recon = tf.identity(self.recon, name='reconstruction')
            self.error_map = tf.identity(self.error_map, name='error_map')
            self.normalize = tf.identity(self.normalize, name='bp_normalizer')
            self.forward = tf.identity(self.forward, name='forward_prediction')

        # regularization - spatial shift:
        with tf.name_scope('regularization'):
            if self.use_spatial_shifts:
                shift_reg = tf.reduce_sum((self.xz_delta - self.xz_delta_init) ** 2)
                shift_reg = tf.multiply(shift_reg, self.shift_reg_coeff, name='square_shift_reg')
                self.loss_terms.append(shift_reg)

                # counteract changes in radius:
                th_inds = np.arange(len(self.ths))
                th_inds180 = np.roll(th_inds, len(self.ths) // 2)

                xz_delta_x = self.xz_delta[:, 0]
                xz_delta_z = self.xz_delta[:, 1]
                diffx = xz_delta_x - tf.gather(xz_delta_x, th_inds180)
                diffz = xz_delta_z - tf.gather(xz_delta_z, th_inds180)
                tube_radius_change_reg = tf.reduce_sum(diffx ** 2 + diffz ** 2)
                tube_radius_change_reg = tf.multiply(tube_radius_change_reg, self.tube_radius_change_reg_coeff,
                                                     name='tube_expansion_reg')
                self.loss_terms.append(tube_radius_change_reg)

            # regularization for nonparametric optimization:
            support_reg = tf.multiply(4000., self.L2_mask(self.external_TV_mask.flatten()), name='support_reg')
            TV_reg = tf.multiply(self.TV_reg_coeff, self.TVreg_mask(A=self.A_nonparametric, mask=self.internal_mask,
                                                                    use_sqrt=True), name='TV_reg')
        self.loss_terms.append(support_reg)
        self.loss_terms.append(TV_reg)

        # for convenience
        self.loss_term_names = tf.constant([term.name for term in self.loss_terms], name='list_of_loss_terms')

    def create_train_op(self):
        # generate list of train_ops (which should be grouped)

        loss = tf.reduce_sum(self.loss_terms)
        with tf.name_scope('RI_train_op'):
            train_op_nonparametric = tf.train.AdamOptimizer(learning_rate=self.lr_nonparametric).minimize(
                loss, var_list=[self.A_nonparametric])

            if self.use_parametric_optimization:
                train_op_parametric = tf.train.AdamOptimizer(learning_rate=self.lr_parametric).minimize(
                    loss, var_list=[self.A_parametric])
                if self.switch_to_nonparametric:
                    # in this case, have both present, and change the learning
                    # rates during training
                    train_op_ = tf.group(train_op_parametric, train_op_nonparametric)
                else:  # parametric optimization only
                    train_op_ = train_op_parametric
            else:  # nonparametric only
                train_op_ = train_op_nonparametric

        self.train_ops = list()  # to be grouped together
        self.train_ops.append(train_op_)

        # train ops for other optimizable parameters other than index:
        with tf.name_scope('non_RI_train_ops'):
            if 'gaussian' in self.PSFconv_format:
                lr_PSF = .01
                train_op_PSF = tf.train.AdamOptimizer(learning_rate=lr_PSF).minimize(loss, var_list=[self.h])
                self.train_ops.append(train_op_PSF)
            if 'arbitrary' in self.PSFconv_format:
                lr_PSF = .01
                train_op_PSF2 = tf.train.AdamOptimizer(learning_rate=lr_PSF).minimize(loss, var_list=[self.PSF_general])
                self.train_ops.append(train_op_PSF2)

            if self.include_attenuation:
                lr_atten = .001
                train_op_atten1 = tf.train.AdamOptimizer(learning_rate=lr_atten).minimize(
                    loss, var_list=[self.attenfact])
                train_op_atten2 = tf.train.AdamOptimizer(learning_rate=.05).minimize(loss, var_list=[self.attenexp])
                self.train_ops.append(train_op_atten1)
                self.train_ops.append(train_op_atten2)

            if self.use_reflectance_model:
                train_op_refl = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss, var_list=[self.angle_mod])
                self.train_ops.append(train_op_refl)

            train_op_reconoffset = tf.train.AdamOptimizer(learning_rate=.001).minimize(
                loss, var_list=[self.recon_offset])
            self.train_ops.append(train_op_reconoffset)

            if self.use_spatial_shifts:
                train_op_spatial_shifts = tf.train.AdamOptimizer(learning_rate=.0005).minimize(
                    loss, var_list=[self.xz_delta])
                self.train_ops.append(train_op_spatial_shifts)

        # have a separate instance variable for the grouped train_op:
        # sometimes may want to modify the list of train_ops and then regroup
        self.train_op = tf.group(*self.train_ops)

    def modify_loss_and_train_op_and_initialize(self):
        # according to binary settings, modify the loss and/or train_op to do filter optimization; then, initialize all
        # variables

        if self.infer_backprojection_filter:  # after registration, load ckpt and optimize the backprojection filter
            assert not self.use_spatial_shifts
            # overwrite this:
            x_interp, z_interp = self.interp_rays_2d(self.paths[:, :, :, 1], self.paths[:, :, :, 0])
            self.stop_gradient_projection = False  # or else gradient can't flow
            loss_data = self.backprojection_tf(x_interp, z_interp)
            # replace:
            self.loss_terms = [tf.multiply(10., loss_data, name='data_loss')]
            self.recon = tf.identity(self.recon, name='reconstruction')
            self.error_map = tf.identity(self.error_map, name='error_map')
            self.normalize = tf.identity(self.normalize, name='bp_normalizer')
            self.forward = tf.identity(self.forward, name='forward_prediction')
            # for now, slice the appropriate dim, because we are not doing 3D:
            # (also, tf's total variation implemention is anisotropic)
            TV_recon = self.TVreg_mask(self.recon[:, :, 5], use_sqrt=True, numgauss_s=self.recon_res[0])
            TV_recon = tf.multiply(1e-7, TV_recon, name='TV_for_filter_opt')
            self.loss_terms.append(TV_recon)
            self.loss_term_names = tf.constant([term.name for term in self.loss_terms], name='list_of_loss_terms')

            self.train_op = tf.train.AdamOptimizer(learning_rate=.001).minimize(tf.reduce_sum(self.loss_terms),
                                                                                var_list=[self.fbp_filters])
            self.train_ops = [self.train_op]

            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, self.save_directory + '/model.ckpt')
            self.second_round_of_optimization = True

        else:  # optimize from scratch (without loading from a checkpoint)
            self.sess.run(tf.global_variables_initializer())
            self.second_round_of_optimization = False

    def save_graph(self):
        # save model and model parameters

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print('Created new directory: ' + self.save_directory)

        if self.second_round_of_optimization:
            # after filter optimization
            self.saver.save(self.sess, self.save_directory + '/model_round2.ckpt')
        else:
            # after registration
            self.saver.save(self.sess, self.save_directory + '/model.ckpt')

    def get_parametric_indexdist(self, params, transition_thickness=.003):
        # parametric index distribution for a circular tube with logistic transitions (the transition thickness is given
        # by the second argument); the output of this function is added to A

        x0 = self.xz0[0] - self.xz_delta_init[0, 0]
        z0 = self.xz0[1] - self.xz_delta_init[0, 1]
        r_inner = self.outer_radius - self.wall_thickness
        r_outer = self.outer_radius
        h_inner = params[0]
        h_outer = params[1]
        # radial distance from the center:
        rc = tf.sqrt((self.Xc - x0) ** 2 + (self.Zc - z0) ** 2)
        ring_inner = h_inner / (1 + tf.exp(
            tf.clip_by_value((rc - r_inner) / transition_thickness, -15, 15)))
        ring_outer = h_outer / (1 + tf.exp(
            tf.clip_by_value((rc - r_outer) / transition_thickness, -15, 15)))

        return ring_inner + ring_outer

    def indexdist(self, x_, z_):
        # returns 2D kernel-parameterized index and spatial derivatives at the given input coordinates;
        # based on nadaraya-watson estimator

        # define coordinates and rotate them:
        xz = tf.stack([x_, z_], 2)
        xz -= self.xz0  # subtract out center

        xzp = tf.matmul(xz, tf.transpose(self.rotmats, [0, 2, 1]))
        # small spatial tweaks; expand dims so that it's batchsizex1x2:
        xzp -= tf.expand_dims(self.xz_delta, 1)
        xzp += self.xz0  # add center back after rotating
        X = xzp[:, :, 0:1]
        Z = xzp[:, :, 1:2]

        Xround = tf.to_int32(tf.round(X * self.numgauss_s))
        Zround = tf.to_int32(tf.round(Z * self.numgauss_s))
        # these should be _xnumgauss_s:
        Xneigh = tf.mod(Xround + self.Xr, self.numgauss_s)
        Zneigh = tf.mod(Zround + self.Zr, self.numgauss_s)

        lininds = self.numgauss_s * Zneigh + Xneigh  # convert to linear indices

        Xg = tf.gather(self.Xc, lininds)
        Zg = tf.gather(self.Zc, lininds)

        # index A and sigma too
        A_ = tf.gather(self.A, lininds)
        sig_ = tf.gather(self.sig, lininds)

        # add something close to machine eps for float32 to avoid divide by 0:
        # intermediate, will be used several times:
        n_intermed = tf.exp(-((Xg - X) ** 2 + (Zg - Z) ** 2) * .5 / (sig_ ** 2)) + 2e-7
        # normalize the sum (cf, nadaraya-watson):
        norm = tf.reduce_sum(n_intermed, 2)
        n_unnorm = n_intermed * A_  # unnormalized, used to calculate gradients

        nx = n_unnorm * (Xg - X) / sig_ ** 2
        nz = n_unnorm * (Zg - Z) / sig_ ** 2
        nx = tf.reduce_sum(nx, 2)
        nz = tf.reduce_sum(nz, 2)

        norm_deriv_x = n_intermed * (Xg - X) / sig_ ** 2
        norm_deriv_z = n_intermed * (Zg - Z) / sig_ ** 2
        norm_deriv_x = tf.reduce_sum(norm_deriv_x, 2)
        norm_deriv_z = tf.reduce_sum(norm_deriv_z, 2)

        # now done with this, so collapse:
        n_unnorm = tf.reduce_sum(n_unnorm, 2)

        # now, compute the index distribution:
        n = n_unnorm / norm

        # ...and the spatial gradients (quotient rule)
        nx = (norm * nx - n_unnorm * norm_deriv_x) / norm ** 2
        nz = (norm * nz - n_unnorm * norm_deriv_z) / norm ** 2

        grad_ = tf.stack([nx, nz], 2)
        grad = tf.matmul(grad_, self.rotmats)

        return 1. + n, grad[:, :, 0], grad[:, :, 1]

    def rayeq_opl(self, z, x):
        # specifies the ray equation differential equation to be used in rk4;
        # returns the refractive index as well to divide by to encode opl

        (n, dndx, dndz) = self.indexdist(x[:, :, 0], z)
        dXdz1 = x[:, :, 1]
        dXdz2 = 1. / n * (dndx * (1 + dXdz1 ** 2) - dndz * dXdz1)
        return tf.stack([dXdz1, dXdz2], 2), n

    def rk4_step_opl(self, xz0, dummy):
        # adapted from: https://github.com/andyr0id/theano-rk4/blob/master/rk4.py;
        # this implementation of rk4 has a variable step size depending on opl;
        # xz0: not the center of rotation, but the current position;
        # dummy: tf.scan requires passing elems as the second argument, but we actually don't need it;

        z0 = xz0[:, :, 0]
        x0 = xz0[:, :, 1:3]
        # get the index at current step,to define the step size
        (deriv, n) = self.rayeq_opl(z0, x0)
        h = self.step / tf.expand_dims(n, -1)
        half_h = h * .5
        k1 = h * deriv

        # when adding, squeeze to avoid perpendicular addition (broadcasting):
        z2 = z0 + tf.squeeze(half_h)
        x2 = x0 + (k1 / 2)
        # ... but need expanded for broadcasting multiplication:
        k2 = h * self.rayeq_opl(z2, x2)[0]

        x3 = x0 + (k2 / 2)
        k3 = h * self.rayeq_opl(z2, x3)[0]

        z4 = z0 + tf.squeeze(h)  # see above regarding squeezing
        x4 = x0 + k3
        k4 = h * self.rayeq_opl(z4, x4)[0]

        xi = x0 + (k1 + 2. * k2 + 2 * k3 + k4) / 6.

        xzi = tf.concat([tf.expand_dims(z4, -1), xi], 2)

        return xzi

    def integrate_scan_opl(self, x_init0, z_init0, return_derivs=False):
        # this is the function that actually propagates the rays given initial conditions (first two arguments);
        # return_derivs: whether this function returns derivatives of the paths;

        # pack into one tensor:
        xz_init0 = tf.concat([tf.expand_dims(z_init0, -1), x_init0], 2)
        # just needs to be a tensor whose leading dimension is of size numz:
        dummy = tf.ones([self.numz])
        paths = tf.scan(self.rk4_step_opl, dummy, xz_init0, swap_memory=True)

        if return_derivs:  # return everything (paths and derivatives)
            everything = tf.transpose(paths, [1, 0, 2, 3])
            return everything[:, :, :, 0:2], everything[:, :, :, 2]
        else:
            # throw out the third one, which is the dx/dz:
            return tf.transpose(paths[:, :, :, 0:2], [1, 0, 2, 3])

    def interp_rays_2d(self, x_paths, z_paths):
        # the rays were sparsely propagated to generate x_paths and z_paths; this function returns
        # upsampled/interpolated paths;
        # xz_paths: the output of the ray propagation; it is of dimension:
        #     numangles by (reduced z) by (subset of x) by y;
        # interp weights has dimensions:
        #     1 by 1 by 1 by downsample*downsample_x by 4;
        # summing across the last dim gives all 1s

        x_paths = tf.expand_dims(x_paths, -1)  # numang by z by x by 1
        z_paths = tf.expand_dims(z_paths, -1)

        # these 8 variables have dims of ...
        # numang by z-1 by x-1 by (downsample*downsample_z):
        x_UL = x_paths[:, :-1, :-1] * self.interp_weights[..., 0]
        x_UR = x_paths[:, :-1, 1:] * self.interp_weights[..., 1]
        x_DL = x_paths[:, 1:, :-1] * self.interp_weights[..., 2]
        x_DR = x_paths[:, 1:, 1:] * self.interp_weights[..., 3]

        z_UL = z_paths[:, :-1, :-1] * self.interp_weights[..., 0]
        z_UR = z_paths[:, :-1, 1:] * self.interp_weights[..., 1]
        z_DL = z_paths[:, 1:, :-1] * self.interp_weights[..., 2]
        z_DR = z_paths[:, 1:, 1:] * self.interp_weights[..., 3]

        x_interp = x_UL + x_UR + x_DL + x_DR
        z_interp = z_UL + z_UR + z_DL + z_DR

        # reshape strange shape back to normal:
        x_interp = tf.reshape(x_interp,
                              (len(self.ths),
                               self.numz - 1,
                               self.numx_downsamp - 1,
                               self.downsample_x,
                               self.downsample))
        x_interp = tf.transpose(x_interp, (0, 1, 4, 2, 3))
        x_interp = tf.reshape(x_interp,
                              (len(self.ths), self.z_max, self.x_max))

        z_interp = tf.reshape(z_interp,
                              (len(self.ths),
                               self.numz - 1,
                               self.numx_downsamp - 1,
                               self.downsample_x,
                               self.downsample))
        z_interp = tf.transpose(z_interp, (0, 1, 4, 2, 3))
        z_interp = tf.reshape(z_interp,
                              (len(self.ths), self.z_max, self.x_max))

        return x_interp, z_interp

    def backprojection_tf(self, xpaths, zpaths):
        # this function backprojects the B-scan data along the input paths (the two arguments), which can be the
        # outputs of interp_rays_2d, to form the reconstruction, which is then gathered along the same ray paths, from
        # which the forward model is applied;
        #
        # Bscans_filtered: used for backprojection;
        #
        # Bscans_unfiltered: used for computing MSE;
        #
        # Bweights: weight vector, to scale the B-scans for backprojection and/or for gathering the values from the
        # reconstruction; in this implementation, it simply masks out the central reflection artifact;
        #
        # recon_res: a tuple that specifies the resolution of the reconstruction [x,z,y];
        #
        # angle_membership: specifies whether to factor in the angle at which the illumination is incident on the
        # sample; NOTE: this does not consider the angle of the rays themselves, but rather the angle of the overall
        # Bscan; also, the gradient direction map that is generated is not interpolated like recon is; if
        # angle_membership is not None, then it's the actual angle in degrees
        #
        # num_y_pixels: specifies the number of pixels in the y dimension of the reconstruction; the xz dims retain that
        # corresponding to recon_res;
        #
        # use_interp_projection: indicates whether to place the Bscan points onto the 4 surrounding points, distributed
        # according to distance to those points; otherwise, round to the nearest pixel;

        numang = len(self.ths)

        if not self.infer_backprojection_filter:
            Bscans_for_bp = self.Bscans_filtered
        else:  # then you actually need to symbolically filter the unfiltered

            Bscans_filtered_filtopt = self.filter_Bscans_tf(self.Bscans_unfiltered)
            Bscans_filtered_filtopt = tf.reshape(Bscans_filtered_filtopt, [-1])
            Bscans_for_bp = Bscans_filtered_filtopt

        # when doing backprojection, can use the Bweights to avoid central reflection:
        if self.use_Bweights_for_bp:
            Bscans_for_bp *= np.float32(self.Bweights > 0)

        if self.num_y_pixels % 2:
            raise Exception('y dim must be even')

        xzpaths = tf.stack([xpaths, zpaths], 3)
        xzpaths = tf.reshape(xzpaths, [numang, -1, 2])
        xzpaths -= self.xz0
        ypaths = tf.zeros((numang, self.z_max * self.x_max, 1))  # placeholder code for now; currently there is no 3D

        # rotate by Bscan:
        # rotmats is 60(i) by 2(k) by 2(l)
        # xzpaths is 60(i) by _(j) by 2(l) - a list of xz coordinates per angle
        # so basically, einsum does batch matmul, since tf.matmul doesn't broadcast:
        xz = tf.einsum('ikl,ijl->ijk', self.rotmats, xzpaths)

        # merge xz and y:
        xzy = tf.concat([xz, ypaths], axis=2)  # 60 by _ by 3

        # decenter and add spatial shifts:
        xzy += tf.concat([self.xz0, [0.]], axis=0)  # decenter
        # 60 by 3 (no need to tile here; broadcasting works!):
        xzy_delta = tf.concat([-self.xz_delta,
                               tf.expand_dims(self.y_delta, -1)], axis=1)
        xzy_delta = tf.expand_dims(xzy_delta, 1)  # 60 by 1 by 3
        xzy += xzy_delta

        # discretize and remove out of bound coordinates:
        xzy = tf.reshape(xzy, (-1, 3))  # now just a list of coordinates

        xzy_float = xzy * tf.to_float(self.recon_res)  # rescale from [0,1]
        xz_float = xzy_float[:, 0:2]
        y_float = xzy_float[:, 2:3]
        y_float += tf.to_float(self.num_y_pixels) / 2  # decenter from 0
        # clip values that are out of bounds:
        # assume recon_res[0]==recon_res[1]:
        xz_float = tf.clip_by_value(xz_float, 2, tf.to_float(self.recon_res[0] - 3))
        y_float = tf.clip_by_value(y_float, 1., tf.to_float(self.num_y_pixels - 2))

        xzy_float = tf.concat([xz_float, y_float], axis=1)  # recombine
        xzy = tf.to_int32(tf.round(xzy_float))

        # trilinear interp (for backprojection/scattering and gathering):
        x = xzy_float[:, 0]
        z = xzy_float[:, 1]
        y = xzy_float[:, 2]

        x_floor = tf.floor(x)
        x_ceil = tf.floor(x + 1)
        z_floor = tf.floor(z)
        z_ceil = tf.floor(z + 1)
        y_floor = tf.floor(y)
        y_ceil = tf.floor(y + 1)

        fx = x - x_floor
        cx = x_ceil - x
        fz = z - z_floor
        cz = z_ceil - z
        # fy = y-y_floor
        # cy = y_ceil-y

        # cast into integers:
        x_floor = tf.to_int32(x_floor)
        x_ceil = tf.to_int32(x_ceil)
        z_floor = tf.to_int32(z_floor)
        z_ceil = tf.to_int32(z_ceil)
        y_floor = tf.to_int32(y_floor)
        y_ceil = tf.to_int32(y_ceil)

        # y is ignored from here for simplicity (since only 2D is implemented for now)

        # generate the coordinates of the projection cells:
        xzyfff = tf.stack([x_floor, z_floor, y_floor], 1)
        xzyfcf = tf.stack([x_floor, z_ceil, y_floor], 1)
        xzycff = tf.stack([x_ceil, z_floor, y_floor], 1)
        xzyccf = tf.stack([x_ceil, z_ceil, y_floor], 1)

        # reconstruct:
        recon_size = tf.concat([self.recon_res[0:2], [self.num_y_pixels]], 0)
        if self.use_interp_projection:
            # compute the interpolated normalize tensor here; _8 is used to indicate 3D, as trilinear interpolation uses
            # 8 cubes, but here for simplicity of 2D, we use 4 squares;
            xzy_8 = tf.concat([xzyfff, xzyfcf, xzycff, xzyccf], 0)

            # compute the interpolated backprojection
            # gaussian-weighted factors:
            sig_proj = .42465  # chosen so that if the point is exactly halfway between two pixels, .5 weight is
            # assigned to each pixel
            fx = tf.exp(-fx ** 2 / 2. / sig_proj ** 2)
            fz = tf.exp(-fz ** 2 / 2. / sig_proj ** 2)
            cx = tf.exp(-cx ** 2 / 2. / sig_proj ** 2)
            cz = tf.exp(-cz ** 2 / 2. / sig_proj ** 2)

            Bscans_8 = tf.concat([
                Bscans_for_bp * fx * fz,
                Bscans_for_bp * fx * cz,
                Bscans_for_bp * cx * fz,
                Bscans_for_bp * cx * cz,
            ], 0)

            if self.normalize_across_whole_dataset:
                # this produces streak artifacts when there are focusing rays

                normalize = tf.scatter_nd(xzy_8, tf.ones_like(Bscans_8),
                                          recon_size) + 1e-7
                recon = tf.scatter_nd(xzy_8, Bscans_8, recon_size)

            else:
                # this block below normalizes within each B-scan first (to account for ray focusing), and then
                # normalizes across entire dataset; this is computationally intensive;

                # encode xzya_8 into integer:
                angle_membership_augmented = tf.tile(tf.to_int32(self.angle_membership / self.angle_step), [4])
                xzya_8 = tf.concat([xzy_8, angle_membership_augmented[:, None]], 1)
                encoder = tf.to_int64(tf.stack([1, recon_size[0], recon_size[0] * recon_size[1],
                                                recon_size[0] * recon_size[1] * recon_size[2]], 0))
                # encode to unique int64 integer:
                xzya_1d = tf.reduce_sum(tf.to_int64(xzya_8) * encoder[None], 1)

                # isolate unique coordinates per angle:
                xzya_1d_sorted, xzya_1d_sorted_ids = tf.nn.top_k(xzya_1d, k=tf.shape(xzya_1d)[0])
                xzya_1d_unique, segments = tf.unique(xzya_1d_sorted)

                # average segments in Bscans, per angle:
                Bscans_8_sorted = tf.gather(Bscans_8, xzya_1d_sorted_ids)
                Bscans_8_normalized = tf.segment_mean(Bscans_8_sorted, segments)

                # decode xzy:
                a_decode = xzya_1d_unique // encoder[3]
                # subtract out the largest place digit first:
                y_decode = (xzya_1d_unique - a_decode * encoder[3]) // encoder[2]
                z_decode = (xzya_1d_unique - a_decode * encoder[3] - y_decode * encoder[2]) // encoder[1]
                x_decode = (xzya_1d_unique - a_decode * encoder[3] - y_decode * encoder[2] - z_decode * encoder[1])

                # xzya_8_unique = tf.stack([x_decode,z_decode,y_decode,a_decode], 1)
                xzy_8_unique = tf.to_int32(tf.stack([x_decode, z_decode, y_decode], 1))

                recon = tf.scatter_nd(xzy_8_unique, Bscans_8_normalized, recon_size)
                normalize = tf.scatter_nd(xzy_8_unique, tf.ones_like(x_decode, dtype=tf.float32), recon_size) + 1e-7

                # this factor roughly to match a similar range to that of the normalize_across_whole_dataset:
                recon *= 4.5

        else:  # project by rounding to nearest pixel:
            recon = tf.scatter_nd(xzy, Bscans_for_bp, recon_size)
            normalize = tf.scatter_nd(xzy, tf.ones_like(Bscans_for_bp), recon_size) + 1e-7

        if not self.turn_off_all_normalization:
            recon = recon / normalize * len(self.ths)

        if self.stop_gradient_projection:
            recon = tf.stop_gradient(recon)  # this might save some computation and avoid some artifacts

        recon += self.recon_offset

        # gathering stage for computing the loss
        ff = tf.gather_nd(recon, xzyfff)
        fc = tf.gather_nd(recon, xzyfcf)
        cf = tf.gather_nd(recon, xzycff)
        cc = tf.gather_nd(recon, xzyccf)

        recon_interp_gathered = (cc * cx * cz + cf * cx * fz + fc * fx * cz + ff * fx * fz)

        # now, compute the loss:

        if self.angle_membership is not None:
            imdir, angles = self.image_gradient(recon)  # compute direction map
            imdir = tf.to_float(imdir)
            # gathering by nearest pixel, not linear interp like above:
            imdir_xzy = tf.gather_nd(imdir, xzy)

            # relative to each Bscan's angle:
            surface_normal = tf.mod(tf.to_int32(
                tf.round(imdir_xzy - self.angle_membership / 360 * 2 * self.discret_levels)), self.discret_levels)
            # if using arbitrary model, pass the int32 indices:
            rm = self.reflectance_model(surface_normal, model='arbitrary')

            recon_interp_gathered_angmod = recon_interp_gathered * rm

        if self.use_blur_Bscan_in_forward_pred:
            (recon_interp_gathered_conv,
             recon_interp_gathered_noconv) = self.convolve_PSF_recon_gathered(recon_interp_gathered)
            # run the function again to get the angmod version:
            recon_interp_gathered_angmod_conv, _ = self.convolve_PSF_recon_gathered(recon_interp_gathered_angmod)
        else:
            # ie, don't use self.convolve_PSF_recon_gathered
            # self.h acts as an intensity scaler
            recon_interp_gathered_conv = self.h * tf.reshape(recon_interp_gathered, self.reduced_shape)
            recon_interp_gathered_noconv = tf.reshape(recon_interp_gathered, self.reduced_shape)
            recon_interp_gathered_angmod_conv = self.h * tf.reshape(recon_interp_gathered_angmod, self.reduced_shape)

        if self.include_attenuation:  # apply after convolution
            recon_interp_gathered_conv = self.attenuate_Bscans_angmod(recon_interp_gathered_conv,
                                                                      recon_interp_gathered_angmod_conv)
        else:
            recon_interp_gathered_conv = recon_interp_gathered_angmod_conv

        # error between forward prediction and B-scan data
        error = (self.Bscans_unfiltered - tf.reshape(recon_interp_gathered_conv, [-1])) ** 2

        if self.Bweights is not None:
            error *= self.Bweights  # weighted sum
            MSE = tf.reduce_sum(error)
        else:
            MSE = tf.reduce_mean(error)

        # for diagnostic purposes
        self.error_map = tf.scatter_nd(xzy, error, recon_size)
        self.error_map /= normalize

        self.forward = [recon_interp_gathered_conv, recon_interp_gathered_noconv]

        self.diagnostics = (xzy, xzy_float)
        self.recon = recon
        self.normalize = normalize
        # recon: the normalized reconstruction
        # normalize: the normalization tensor
        return MSE

    def convolve_PSF_recon_gathered(self, gathered, conv_dim='2D'):
        # helper function for backprojection_tf; takes the gathered values, representing a distorted Bscan, and
        # convolves it with a PSF, which itself is a parameter that needs to be or can be inferred; the gathered tensor
        # needs to be reshaped to a B-scan because it has a strange shape: numang, z-1, x-1, downsample*downsample_x;
        # conv_dim can be '2D' or '3D'

        B = tf.reshape(gathered, self.reduced_shape)

        # define PSF:
        if conv_dim == '2D':
            PSF = 0
            if 'gaussian' in self.PSFconv_format:
                x_PSF = np.arange(-self.sx // 2 + 1, self.sx // 2 + 1, dtype=np.float32)
                z_PSF = np.arange(-self.sz // 2 + 1, self.sz // 2 + 1, dtype=np.float32)
                [x_PSF, z_PSF] = np.meshgrid(x_PSF, z_PSF)
                PSF_gauss = tf.exp(-x_PSF ** 2 / 2 / self.sigx ** 2
                                   - z_PSF ** 2 / 2 / self.sigz ** 2)
                PSF += PSF_gauss
            if 'arbitrary' in self.PSFconv_format:
                PSF += self.PSF_general
            PSF /= tf.reduce_sum(PSF)  # rescale the image only using h
            PSF = PSF[..., None, None]  # to conform with conv2d

            # convolution
            Bconv = tf.transpose(B, (0, 3, 1, 2))
            # for the 2D case, both angle and y represent batch dimensions
            Bconv = tf.reshape(Bconv, (len(self.ths) * self.numy,  # note, numy==reduced_shape[3]
                                       self.reduced_shape[1],
                                       self.reduced_shape[2], 1))
            Bconv = tf.nn.conv2d(Bconv, PSF, strides=[1, 1, 1, 1],
                                 padding='SAME')
            Bconv = tf.reshape(Bconv, (len(self.ths),  # reverse the reshape/transpose
                                       self.numy,
                                       self.reduced_shape[1],
                                       self.reduced_shape[2]))
            Bconv = tf.transpose(Bconv, (0, 2, 3, 1))
        elif conv_dim == '3D':
            raise Exception('3D not implemented')
        else:
            raise Exception('enter 2D or 3D')

        return self.h * Bconv, B  # B is the unconvolved version

    def attenuate_Bscans_angmod(self, Bscans, Bscans_angmod):
        # helper function for backprojection_tf; this function attenuates Bscans_angmod according to the intensities in
        # Bscans; currently, uses one set of attenuation parameters for all angles

        Bcumsum = tf.maximum(tf.cumsum(Bscans, axis=1), 0)
        Batten = Bscans_angmod * tf.exp(-tf.abs(Bcumsum * tf.clip_by_value(self.attenfact, 1e-3, 10)) **
                                         tf.clip_by_value(self.attenexp, 1, 100))
        return Batten

    def image_gradient(self, im):
        # helper function for backprojection_tf;
        # im: 3D; gradients are taken only in the first two dimensions
        # angle_discretization: specifies how many gabor filters to convolve with

        # change to the required format of tf.nn.conv2d:
        im = tf.expand_dims(tf.transpose(im, [2, 0, 1]), -1)

        gabor = list()
        angles = np.linspace(-90, 90, self.discret_levels + 1)
        angles = angles[:-1]  # exclude +90
        for th in angles:  # gabor filters from -90 to 90 degrees
            filt = cv2.getGaborKernel(ksize=(11, 11), sigma=3.0, theta=th / 180 * np.pi, lambd=10, gamma=.5, psi=0,
                                      ktype=cv2.CV_32F)
            gabor.append(filt)
        gabor = np.stack(gabor).transpose(1, 2, 0).astype(np.float32)
        gabor = tf.expand_dims(gabor, 2)  # k by k by 1 by _

        imconv = tf.nn.conv2d(im, gabor, strides=[1, 1, 1, 1], padding='SAME')  # of size y,x,z,_
        imconv = tf.transpose(imconv, [1, 2, 0, 3])  # of size x,z,y,_
        # find the index of the gabor filter giving the largest response:
        imdir = tf.argmax(imconv, axis=3)

        return imdir, angles

    def reflectance_model(self, angles, model):
        # helper function for backprojection_tf;
        # angles: a vector of angles in radians
        # model: specifies what scaling to apply to each angle

        # angles is a vector of angles in radians
        if model == 'lambertian':
            return tf.cos(angles)
        elif model == 'arbitrary':  # expect indices to be passed
            return tf.gather(self.angle_mod, angles)
        else:
            raise Exception('invalid reflectance model')

    def filter_Bscans_tf(self, Bscans):
        # 2D filtering of B-scans in tf

        def tf_fftshift(A):
            # 2D fftshift
            # apply fftshift to the last two dims
            s = A.shape
            s1 = s[-2] + 1
            s2 = s[-1] + 1
            A = tf.concat([A[..., s1 // 2:, :], A[..., :s1 // 2, :]], axis=-2)
            A = tf.concat([A[..., :, s2 // 2:], A[..., :, :s2 // 2]], axis=-1)
            return A

        def tf_ifftshift(A):
            # 2D ifftshift
            # apply ifftshift to the last two dims
            s = A.shape
            s1 = s[-2]
            s2 = s[-1]
            A = tf.concat([A[..., s1 // 2:, :], A[..., :s1 // 2, :]], axis=-2)
            A = tf.concat([A[..., :, s2 // 2:], A[..., :, :s2 // 2]], axis=-1)
            return A

        # unflatten; there is a trailing singleton dim:
        Bscans = tf.reshape(Bscans, self.reduced_shape)
        # since fft2 operates on last two dimensions:
        Bscans = tf.transpose(Bscans, (0, 3, 1, 2))
        Bscans = tf.complex(Bscans, 0.)
        filters = tf.complex(self.fbp_filters, 0.)
        fftd = tf_fftshift(tf.spectral.fft2d(Bscans)) * filters[None, None]
        Bfilt = tf.real(tf.spectral.ifft2d(tf_ifftshift(fftd)))
        Bfilt = tf.transpose(Bfilt, (0, 2, 3, 1))  # transpose back
        return Bfilt

    def filter_Bscans(self, Bscans_full):
        # filters Bscans_full using a filter invariant to x;
        # numpy version

        freqs = np.linspace(-1, 1, Bscans_full.shape[1])

        if self.freq_filter == 'ramlak':
            filt = np.abs(freqs)
        elif self.freq_filter == 'shepp-logan':
            filt = np.abs(freqs)
            filt *= np.sinc(freqs / 2)
        elif self.freq_filter == 'cosine':
            filt = np.abs(freqs)
            filt *= np.cos(freqs * np.pi / 2)
        elif self.freq_filter == 'sqrt':
            filt = np.abs(freqs)
            filt = np.sqrt(filt)
        elif self.freq_filter == 'hamming':
            filt = np.abs(freqs)
            filt *= .54 + .46 * np.cos(freqs * np.pi)
        elif self.freq_filter == 'no_filter':
            filt = np.ones_like(freqs)
        elif self.freq_filter == 'load_external':
            raise Exception('not yet implemented')
        else:
            raise Exception('invalid filter')

        filt = np.reshape(filt, (1, -1, 1, 1))
        fftd = fftshift(fft(Bscans_full, axis=1), 1) * filt
        Bfilt = np.real(ifft(ifftshift(fftd, 1), axis=1))

        return Bfilt

    def L2_mask(self, mask, n=0.):
        # L2 regularization weighted by mask
        # mask: a weight matrix the same size as A to weight the L2 reg
        # n: the value to which A should be regularized

        return tf.reduce_sum(mask * (self.A - n) ** 2) / tf.reduce_sum(mask)

    def TVreg_mask(self, A, mask=None, use_sqrt=False, numgauss_s=None):
        # total variation regularization spatially masked by mask
        # use_sqrt: whether to take the square root of the deviations squared (if False, then strictly speaking this is
        # not TV)
        # numgauss_s: if you would like to override self.numgauss_s
        # note that tf.image.total_variation implements the anisotropic version, while this version is isotropic

        if numgauss_s is not None:
            numgauss_s_ = numgauss_s
        else:
            numgauss_s_ = self.numgauss_s
        AA = tf.reshape(A, (numgauss_s_, numgauss_s_))
        d1 = (AA[0:numgauss_s_ - 1, :] - AA[1:numgauss_s_, :]) ** 2
        d2 = (AA[:, 0:numgauss_s_ - 1] - AA[:, 1:numgauss_s_]) ** 2

        if mask is not None:
            d1 *= mask[0:numgauss_s_ - 1, :]
            d2 *= mask[:, 0:numgauss_s_ - 1]

        if use_sqrt:
            # add something close to machine eps to avoid grads of sqrt at 0
            # indexing so that shapes match
            return tf.reduce_sum(tf.sqrt(d1[0:numgauss_s_ - 1,
                                         0:numgauss_s_ - 1] +
                                         d2[0:numgauss_s_ - 1,
                                         0:numgauss_s_ - 1] + 2e-7))
        else:
            return tf.reduce_sum(d1) + tf.reduce_sum(d2)

    def get_feed_dict(self):
        # Bscans have to be placeholders because they are too big and exceed the 2-GB limit

        feed = {self.Bscans_filtered: self.Bscans_filtered_np,
                self.Bscans_unfiltered: self.Bscans_unfiltered_np}
        return feed
