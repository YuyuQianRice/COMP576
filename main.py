# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from numpy import *
import scipy.linalg
import scipy.signal as ss
import scipy.sparse.linalg as sl
import numpy.fft as nfft
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    b = array([1, 2, 3, 4, 5, -6, 7, 8, 9, 10, 11])
    c = array([11, 12, 13, 24, 15, -16, 17, 18, 19, 110, 111])
    d = array([21, 22, 23, 34, 25, -26, 27, 28, 29, 210, 211])
    e = array([31, 32, 33, 44, 35, -36, 37, 38, 39, 310, 311])
    a = random.randint(1, 100, size=(11, 11))
    a2 = random.randint(1, 100, size=(11, 11))
    # a = random.rand(11, 11)
    # a2 = random.rand(11, 11)
    print("ndim(a): \n" + str(ndim(a)))
    print("a.ndim: \n" + str(a.ndim))
    print("size(a): \n" + str(size(a)))
    print("a.size: \n" + str(a.size))
    print("shape(a): \n" + str(shape(a)))
    print("a.shape: \n" + str(a.shape))
    print("a.shape[n-1]: \n" + str(a.shape[0]))
    print("array([[1., 2., 3.], [4., 5., 6.]]): \n" + str(array([[1., 2., 3.], [4., 5., 6.]])))
    print("block([[e,b], [c,d]]): \n" + str(block([[e, b], [c, d]])))
    # bl = block([[e, b], [c, d]])
    print("a[-1]: \n" + str(a[-1]))
    print("a[1, 4]: \n" + str(a[1, 4]))
    print("a[1]: \n" + str(a[1]))
    print("a[1, :]: \n" + str(a[1, :]))
    print("a[:5]: \n" + str(a[:5]))
    print("a[0:5]: \n" + str(a[0:5]))
    print("a[0:5,:]: \n" + str(a[0:5,:]))
    print("a[-5:]: \n" + str(a[-5:]))
    print("a[0:3][:,4:9]: \n" + str(a[0:3][:, 4:9]))
    print("a[ix_([1, 3, 4], [0, 2])]: \n" + str(a[ix_([1, 3, 4], [0, 2])]))
    print("a[ 2:21:2, :]: \n" + str(a[2:21:2, :]))
    print("a[::2, :]: \n" + str(a[::2, :]))
    print("	a[::-1, :]: \n" + str(a[::-1, :]))
    print("a[r_[:len(a), 0]]: \n" + str(a[r_[:len(a), 0]]))
    print("a.transpose(): \n" + str(a.transpose()))
    print("a.T: \n" + str(a.T))
    print("a.conj().transpose(): \n" + str(a.conj().transpose()))
    print("a.conj().T: \n" + str(a.conj().T))
    print("a @ a2: \n" + str(a @ a2))
    print("a * a2: \n" + str(a * a2))
    print("a / a2: \n" + str(a / a2))
    print("a ** 3: \n" + str(a ** 3))
    print("(a>0.5): \n" + str((a > 0.5)))
    print("nonzero(a>0.5): \n" + str(nonzero(a > 0.5)))
    print("a[:, nonzero(b>0.5)[0]]: \n" + str(	a[:, nonzero(b > 0.5)[0]]))
    print("a[:, b.T>0.5]: \n" + str(a[:, b.T > 0.5]))
    a[a < 0.5] = 0
    print("a[a < 0.5] = 0: \n" + str(a))
    print("a * (a>0.5): \n" + str(a * (a > 0.5)))
    a[:] = 3
    print("a[:] = 3: a\n" + str())
    a = random.rand(11, 11)
    x = array([b, c, d, e])
    y = x.copy()
    print("y = x.copy(): \n" + str(y))
    y = x[1, :].copy()
    print("y = x[1, :].copy() \n" + str(y))
    y = x.flatten()
    print("y = x.flatten(): \n" + str(y))
    print("arange(1.,11.): \n" + str(arange(1.,11.)))
    print("r_[1.:11.]: \n" + str(r_[1.:11.]))
    print("r_[1:10:10j]: \n" + str(r_[1:10:10j]))
    print("arange(10.): \n" + str(arange(10.)))
    print("r_[:10.]: \n" + str(r_[:10.]))
    print("r_[:9:10j]: \n" + str(r_[:9:10j]))
    print("arange(1.,11.)[:, newaxis]: \n" + str(arange(1.,11.)[:, newaxis]))
    print("zeros((3,4)): \n" + str(zeros((3, 4))))
    print("zeros((3,4,5)): \n" + str(zeros((3, 4, 5))))
    print("ones((3,4)): \n" + str(ones((3, 4))))
    print("eye(3): \n" + str(eye(3)))
    print("diag(a): \n" + str(diag(a)))
    print("diag(a,0): \n" + str(diag(a, 0)))
    print("random.rand(3,4): \n" + str(random.rand(3, 4)))
    print("random.random_sample((3, 4)): \n" + str(random.random_sample((3, 4))))
    print("linspace(1,3,4): \n" + str(linspace(1, 3, 4)))
    print("mgrid[0:9.,0:6.]: \n" + str(mgrid[0:9., 0:6.]))
    print("meshgrid(r_[0:9.],r_[0:6.]): \n" + str(meshgrid(r_[0:9.],r_[0:6.])))
    print("ogrid[0:9.,0:6.]: \n" + str(ogrid[0:9.,0:6.]))
    print("ix_(r_[0:9.],r_[0:6.]): \n" + str(ix_(r_[0:9.],r_[0:6.])))
    print("meshgrid([1,2,4],[2,4,5]): \n" + str(meshgrid([1, 2, 4],[2, 4, 5])))
    print("ix_([1,2,4],[2,4,5]): \n" + str(ix_([1, 2, 4], [2, 4, 5])))
    m = 3
    n = 4
    print("tile(a, (m, n)): \n" + str(tile(a, (m, n))))
    print("concatenate((a,a2),1): \n" + str(concatenate((a, a2), 1)))
    print("hstack((a,a2)): \n" + str(hstack((a, a2))))
    print("column_stack((a,a2)): \n" + str(column_stack((a, a2))))
    print("c_[a,a2]: \n" + str(c_[a, a2]))
    print("concatenate((a,a2)): \n" + str(concatenate((a, a2))))
    print("vstack((a,a2)): \n" + str(vstack((a, a2))))
    print("r_[a,a2]: \n" + str(r_[a, a2]))
    print("a.max(): \n" + str(a.max()))
    print("a.max(0): \n" + str(a.max(0)))
    print("a.max(1): \n" + str(a.max(1)))
    print("maximum(a, a2): \n" + str(maximum(a, a2)))
    v = b
    print("sqrt(v @ v): \n" + str(sqrt(v @ v)))
    print("np.linalg.norm(a): \n" + str(linalg.norm(a)))
    print("logical_and(a, a2): \n" + str(logical_and(a, a2)))
    print("logical_or(a, a2): \n" + str(logical_or(a, a2)))
    print("linalg.inv(a): \n" + str(linalg.inv(a)))
    print("linalg.pinv(a): \n" + str(linalg.pinv(a)))
    print("linalg.matrix_rank(a): \n" + str(linalg.matrix_rank(a)))
    print("linalg.solve(a, a2): \n" + str(linalg.solve(a, a2)))
    a = random.randint(1, 100, size=(4, 5))
    a2 = random.randint(1, 100, size=(4, 5))
    print("linalg.lstsq(a, a2): \n" + str(linalg.lstsq(a, a2, rcond=-1)))

    a = array([array([2, 14, 7]), array([4, 32, 78]), array([3, 12, 66])])
    a2 = array([array([3, 13, 11]), array([21, 22, 19]), array([17, 18, 33])])
    U, S, Vh = linalg.svd(a)
    V = Vh.T
    print("U, S, Vh = linalg.svd(a), V = Vh.T: \n" +
          "  U: \n" + str(U) + "\n" +
          "  S: \n" + str(S) + "\n" +
          "  Vh: \n" + str(Vh) + "\n" +
          "  V: \n" + str(V))
    print("linalg.cholesky(a).T: \n" + str(linalg.cholesky(a).T))
    D, V = linalg.eig(a)
    print("D,V = linalg.eig(a): \n" +
          "D: \n" + str(D) + "\n" +
          "V: \n" + str(V)
          )

    D, V = scipy.linalg.eig(a, a2)
    print("D,V = scipy.linalg.eig(a, a2): \n" +
          "D: \n" + str(D) + "\n" +
          "V: \n" + str(V)
          )
    Q, R = scipy.linalg.qr(a)
    print("Q, R = scipy.linalg.qr(a): \n" +
          "Q: \n" + str(Q) + "\n" +
          "R: \n" + str(R)
          )
    L = scipy.linalg.lu(a)[0]
    U = scipy.linalg.lu(a)[1]
    print("L = scipy.linalg.lu(a)[0] U = scipy.linalg.lu(a)[1]: \n" +
          "L: \n" + str(L) + "\n" +
          "U: \n" + str(U)
          )
    LU, P = scipy.linalg.lu_factor(a)
    print("LU, P = scipy.linalg.lu_factor(a): \n" +
              "LU: \n" + str(LU) + "\n" +
              "P: \n" + str(P)
              )
    print("scipy.sparse.linalg.cg: \n" + str(sl.cg))
    fftAns = nfft.fft(a)
    print("fft(a): \n" + str(fftAns))
    ifftAns = nfft.ifft(a)
    print("ifft(a): \n" + str(ifftAns))
    print("sort(a): \n" + str(sort(a)))
    print("a.sort(): \n" + str(a.sort()))
    x = array([0, 1, 2, 3])
    y = array([-1, 0.2, 0.9, 2.1])
    A = vstack([x, ones(len(x))]).T
    m = linalg.lstsq(A, y, rcond=-1)[0]
    c = linalg.lstsq(A, y, rcond=-1)[1]
    print("m = linalg.lstsq(A, y, rcond=-1)[0] c = linalg.lstsq(A, y, rcond=-1)[1]: \n" +
          "m: \n" + str(LU) + "\n" +
          "c: \n" + str(P)
          )
    y = linspace(0, 10, 20, endpoint=False)
    x = cos(-y ** 2 / 6.0)
    q = 10
    f = ss.resample(x, (int) (len(x) / q))
    print("scipy.signal.resample(x, len(x) / q): \n" + str(f))
    print("unique(a) / q): \n" + str(unique(a)))
    print("a.squeeze(): \n" + str(a.squeeze()))

    import matplotlib.pyplot as plt

    plt.plot([1, 2, 3, 4], [1, 2, 7, 14])
    plt.axis([0, 6, 0, 20])
    plt.show()

    # from scipy import signal
    # x = np.linspace(0, 20, 20, endpoint=False)
    # y = np.cos(-x ** 3 / 4.0)
    # f = signal.resample(y, 200)
    # xnew = np.linspace(0, 20, 200, endpoint=False)
    # import matplotlib.pyplot as plt
    # plt.plot(x, y, 'go-', xnew, f, '.-', 20, y[0], 'ro')
    # plt.legend(['data', 'resampled'], loc='best')
    # plt.show()

    x = linspace(0, 20, 20, endpoint=False)
    y = cos(-x**3/4.0)
    f = ss.resample(y, 200)
    xnew = linspace(0, 20, 200, endpoint=False)
    plt.plot(x, y, 'go-', xnew, f, '.-', 20, y[0], 'ro')
    plt.legend(['data', 'resampled'], loc='best')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
