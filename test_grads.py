import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from pdb import set_trace
from cv2 import EMD as cv_emd #openCV
from external.structural_losses.tf_approxmatch import approx_match, match_cost

def main(n1, n2, dim, seed):
    # Generate data with numpy
    np.random.seed(seed)
    pts1 = np.random.randn(n1, dim)
    pts2 = np.random.randn(n2, dim)
    grad_ix_n = np.random.randint(min(n1, n2), size=5)
    grad_ix_dim = np.random.randint(dim, size=5)

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

    # Tensorflow
    # tf Graph
    pts1_tf = tf.convert_to_tensor(pts1.reshape(1,n1,dim), dtype=tf.float32)
    pts2_tf = tf.convert_to_tensor(pts2.reshape(1,n2,dim), dtype=tf.float32)
    match = approx_match(pts1_tf, pts2_tf)
    tf_loss = match_cost(pts1_tf, pts2_tf, match)
    grads = tf.gradients([tf_loss], [pts1_tf, pts2_tf])
    # tf Session
    sess = tf.Session()
    print("Tensorflow EMD {:.4f}".format(sess.run(tf_loss[0])))
    pts1_grad_np, pts2_grad_np = sess.run(grads)
    print("CUDA EMD Grad t1 (mean) {:.4f}".format(pts1_grad_np.mean()))
    print("CUDA EMD Grad t1 (std) {:.4f}".format(pts1_grad_np.std()))
    print("CUDA EMD Grad t2 (mean) {:.4f}".format(pts2_grad_np.mean()))
    print("CUDA EMD Grad t2 (std) {:.4f}".format(pts2_grad_np.std()))
    print("CUDA EMD Grad t1 (random) {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(
        *pts1_grad_np[0, grad_ix_n, grad_ix_dim]))
    print("CUDA EMD Grad t2 (random) {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}".format(
        *pts2_grad_np[0, grad_ix_n, grad_ix_dim]))

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
