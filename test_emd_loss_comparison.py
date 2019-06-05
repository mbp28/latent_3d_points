#import torch
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import scipy
from pdb import set_trace
#from emd import EMDLoss as cuda_emd
from scipy.stats import wasserstein_distance as scipy_emd #scipy
#from pyemd import emd as py_emd #github python version
from cv2 import EMD as cv_emd #openCV
from external.structural_losses.tf_approxmatch import approx_match, match_cost

def main(n1, n2, dim, seed):
    # Generate data with numpy
    np.random.seed(seed)
    pts1 = np.random.randn(n1, dim)
    pts2 = np.random.randn(n2, dim)

    # Scipy EMD
    if dim == 1:
        # scipy only works on univariate data
        scipy_loss = scipy_emd(pts1.squeeze(1), pts2.squeeze(1))
        print("Scipy EMD {:.4f}".format(scipy_loss))

    # PyEMD
    # each point becomes a histogram bin, each point set becomes a binary vector to
    # indicate which bins (i.e. points) it contains # use pairwise distances
    # between histogram bins to get the correct emd
    # pts = np.concatenate([pts1, pts2])
    # dst = scipy.spatial.distance_matrix(pts, pts)
    # hist1 = (1 / n1) * np.concatenate([np.ones(n1), np.zeros(n2)])
    # hist2 = (1 / n2) * np.concatenate([np.zeros(n1), np.ones(n2)])
    # py_loss = py_emd(hist1, hist2, dst)
    # print("PyEMD {:.4f}".format(py_loss))

    # OpenCV
    # each signature is a matrix, first column gives weight (should be uniform for
    # our purposes) and remaining columns give point coordinates, transformation
    # from pts to sig is through function pts_to_sig
    def pts_to_sig(pts):
        # cv2.EMD requires single-precision, floating-point input
        sig = np.empty((pts.shape[0], 1 + pts.shape[1]), dtype=np.float32)
        sig[:,0] = (np.ones(pts.shape[0]) / pts.shape[0])
        sig[:,1:] = pts
        return sig
    sig1 = pts_to_sig(pts1)
    sig2 = pts_to_sig(pts2)
    cv_loss, _, flow = cv_emd(sig1, sig2, cv2.DIST_L2)
    print("OpenCV EMD {:.4f}".format(cv_loss))

    # CUDA_EMD
    # pts1_cuda = torch.from_numpy(pts1).cuda().float().reshape(1, n1, dim)
    # pts2_cuda = torch.from_numpy(pts2).cuda().float().reshape(1, n2, dim)
    # pts1_cuda.requires_grad = True
    # pts2_cuda.requires_grad = True
    # cuda_loss = cuda_emd()(pts1_cuda, pts2_cuda)
    # print("CUDA EMD {:.4f}".format(cuda_loss.item()))

    # Tensorflow
    # tf Graph
    pts1_tf = tf.convert_to_tensor(pts1.reshape(1,n1,dim), dtype=tf.float32)
    pts2_tf = tf.convert_to_tensor(pts2.reshape(1,n2,dim), dtype=tf.float32)
    match = approx_match(pts1_tf, pts2_tf)
    tf_loss = match_cost(pts1_tf, pts2_tf, match)
    # tf Session
    sess = tf.Session()
    print("Tensorflow EMD {:.4f}".format(sess.run(tf_loss[0])))
    # print(sess.run(tf_loss))
    sess.close()


    # set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n1', type=int, default=5)
    parser.add_argument('-n2', type=int, default=5)
    parser.add_argument('-dim', type=int, default=1)
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    main(args.n1, args.n2, args.dim, args.seed)
