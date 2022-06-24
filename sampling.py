"""Module implementing various samplers."""
import itertools
from typing import Generator
from collections.abc import Iterator

import numpy as np
from scipy import stats
from numba import vectorize, float64, int64


def gaussian_sampling(d: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding random normal (gaussian) samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """
    while True:
        yield np.random.normal(0.5, 0.5/3, size=(d, 1))


def sobol_sampling(d: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a Sobol sequence.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """
    sobol = Sobol(d, np.random.randint(2, max(3, d ** 2)))
    while True:
        yield next(sobol).reshape(-1, 1)


def halton_sampling(d: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a Halton sequence.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """
    halton = Halton(d)
    while True:
        yield next(halton).reshape(-1, 1)

def uniform_sampling(d: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding random uniform samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """
    while True:
        yield np.random.uniform(size=(d, 1))

        
def mirrored_sampling(sampler: Generator) -> Generator[np.ndarray, None, None]:
    """Generator yielding mirrored samples.

    For every sample from the input sampler (generator), both its
    original and complemented form are yielded.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray

    Yields
    ------
    numpy.ndarray

    """
    for sample in sampler:
        yield sample
        yield sample * -1


def orthogonal_sampling(
    sampler: Generator, n_samples: int
) -> Generator[np.ndarray, None, None]:
    """Generator yielding orthogonal samples.

    This function orthogonalizes <n_samples>, and succesively yields each
    of them. It uses the linalg.qr decomposition function of the numpy library.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray
    n_samples: int
        An integer indicating the number of sample to be orthogonalized.

    Yields
    ------
    numpy.ndarray

    """
    samples = []
    for sample in sampler:
        samples.append(sample)
        if len(samples) == max(max(sample.shape), n_samples):
            samples = np.hstack(samples)
            L = np.linalg.norm(samples, axis=0)
            Q, *_ = np.linalg.qr(samples.T)
            samples = [s.reshape(-1, 1) for s in (Q.T * L).T]
            for _ in range(n_samples):
                yield samples.pop()


class Halton(Iterator):
    """Iterator implementing Halton Quasi random sequences.

    Attributes
    ----------
    d: int
        dimension
    bases: np.ndarray
        array of primes
    index: itertools.count
        current index

    """

    def __init__(self, d, start=1):
        """Compute the bases, and set index to start."""
        self.d = d
        self.bases = self.get_primes(self.d)
        self.index = itertools.count(start)

    @staticmethod
    def get_primes(n: int) -> np.ndarray:
        """Return n primes, starting from 0."""
        def inner(n_):
            sieve = np.ones(n_ // 3 + (n_ % 6 == 2), dtype=np.bool)
            for i in range(1, int(n_ ** 0.5) // 3 + 1):
                if sieve[i]:
                    k = 3 * i + 1 | 1
                    sieve[k * k // 3:: 2 * k] = False
                    sieve[k * (k - 2 * (i & 1) + 4) // 3:: 2 * k] = False
            return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

        primes = inner(max(6, n))
        while len(primes) < n:
            primes = inner(len(primes) ** 2)
        return primes[:n]

    def __next__(self) -> np.ndarray:
        """Return next Halton sequence."""
        return self.vectorized_next(next(self.index), self.bases)

    @staticmethod
    @vectorize([float64(int64, int64)])
    def vectorized_next(index: int, base: int) -> float:
        """Vectorized method for computing halton sequence."""
        d, x = 1, 0
        while index > 0:
            index, remainder = divmod(index, base)
            d *= base
            x += remainder / d
        return x


class Sobol(Iterator):
    """Iterator implementing Sobol Quasi random sequences.

    This is an iterator version of the version implemented in the python
    package: sobol-seq==0.2.0. This version is 4x faster due to better usage of
    numpy vectorization.

    Attributes
    ----------
    d: int
        dimension
    seed: int
        sample seed
    v: np.ndarray
        array of sample directions
    recipd: int
        1/(common denominator of the elements in v)
    lastq: np.ndarray
        vector containing last sample directions

    """

    def __init__(self, d: int, seed: int = 0):
        """Intialize the v matrix, used for generating Sobol sequences.

        The values for v and poly were taken from the python package sobol-seq.
        """
        self.d = d
        self.seed = np.floor(max(0, seed)).astype(int)
        self.v = np.zeros((40, 30), dtype=int)

        self.v[0:40, 0] = np.ones(40)
        self.v[2:40, 1] = np.r_[
            np.tile([1, 3], 3),
            np.tile(np.r_[np.tile([3, 1], 4), np.tile([1, 3], 4)], 2),
        ]
        self.v[3:40, 2] = [
            7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7,
            5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3
        ]
        self.v[5:40, 3] = [
            1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9,
            13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9
        ]
        self.v[7:40, 4] = [
            9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31,
            11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9
        ]
        self.v[13:40, 5] = [
            37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9,
            49, 33, 19, 29, 11, 19, 27, 15, 25
        ]
        self.v[19:40, 6] = [
            13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3,
            113, 61, 89, 45, 107
        ]
        self.v[37:40, 7] = [7, 23, 39]
        poly = [
            1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109,
            103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299
        ]

        #  Find the number of bits in ATMOST.
        maxcol = Sobol.h1(2 ** 30 - 1)

        #  Initialize row 1 of V.
        self.v[0, :maxcol] = 1

        for i in range(2, self.d + 1):
            j = poly[i - 1]
            m = int(np.log2(j))
            includ = np.fromiter(format(j, "b")[1:], dtype=np.int)
            powers = 2 ** np.arange(1, m + 1)

            for j in range(m + 1, maxcol + 1):
                mask = np.arange(j - 1)[::-1][:m]
                self.v[i - 1, j - 1] = np.bitwise_xor.reduce(
                    np.r_[
                        self.v[i - 1, j - m - 1], powers * self.v[i - 1, mask] * includ
                    ]
                )

        i = np.arange(maxcol - 1)[::-1]
        powers = 2 ** np.arange(1, len(i) + 1)
        self.v[: self.d, i] = self.v[: self.d, i] * powers

        self.recipd = 1.0 / (2 * powers[-1])
        self.lastq = np.zeros(self.d, dtype=int)

        for loc in map(self.l0, range(self.seed)):
            self.lastq = np.bitwise_xor(self.lastq, self.v[: self.d, loc - 1])

    def __next__(self) -> np.ndarray:
        """Return next Sobol sequence."""
        loc = self.l0(self.seed)
        quasi = self.lastq * self.recipd
        self.lastq = np.bitwise_xor(self.lastq, self.v[: self.d, loc - 1])
        self.seed += 1
        return quasi

    @staticmethod
    def h1(n: int) -> int:
        """Return high 1 bit index for a given integer."""
        return len(format(n, "b")) - abs(format(n, "b").find("1"))

    @staticmethod
    def l0(n: int) -> int:
        """Return low 0 bit index for a given integer."""
        x = format(n, "b")[::-1].find("0")
        if x != -1:
            return x + 1
        return len(format(n, "b")) + 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       �Ah�}��70/m ~�A��������A��)9»*�J2.xPC��{>��s�XRa�9w��&N����:OZ"���t�:���Os�\GW���^�W���kq=ƿ���:��G�|J<���2)>��0��ۊ��bD��`��B����Ly�rr�{�x{�����
�&?�ʪI�ذS
^�svJ:���������1<"s��0]O�Ą}���ޱc��4z�ቍ�y�*r�H�
�X�z��U��/�?�/��+���)�
��T�Q��6t���!�w�]���e��h+mx(&�q+��]%�R�s�H�P��9�mmA���j��@uV��pN^��m�;5��Fj˦�m;��G���\��>��7Z]�΢Öс)�������?Y�<\�@����zdh��H��B��_+��Ǔ���|j>��%��J��p���^�`�3�������]���$�͢{k�o�v뵒���4��H�
�-�뇃����jlu� p�Tv�{	��u&���K��B�ŭ�9+�\b��Ѯ
R路m(
�?���h�1r�m�8����'mHk3`[�a�p��<=1ר ��L�R0B+�Q6���t� �����u�����W��/�mhG�eB{,��_�Y��W>�D"l��kv�Lxԅ�CN��ހg��V-�2yM�z���`^��_�Y&�K�,S�1�,�?� ��Y&��2u��`vQPl����a��YpC�Վ�=}ے�v����"0��d[����_�f�^���
��w��'��඙��I���\{�fa+o�61<�h`�f�E~��&��C�ŭm��8ɜ�Ɖ`�����H��`�Y��|�}��`
�%�[�]8�@�%"�O���!`�{�fք7�v����p���5�����	Ճ��O���v=��\$���PI�c'@��sl����!�>|�t�H7-;�~��{�v[��q��^�������!:�7o:�=Ȼgmz���:ϧ:tD7��"��v�wBga��y�a���M�S\�1o����*m��@�M}����= 霠���D~du��@�l������U����.�0݇d0�
q�����'x�`�U����WA�P=��hR�6P���>TR6z�it#7�*����*����� �ʇ�tj7�b
�M�x�*t�׺�Y�%�mm_lB�]�������|�d���݃����[���o�1��%[7��%�HO4�l Y�M"��D鮯:�n� Ai��6
+�p��wI.�� ;n�75�}o*fշ���1_��G8��`���"�+�)�v���?�x0o�|�ɘ6�CZ2Ӓ���wQ&��?GQ��P�3������1���g��� =�k��ܤ��v�gu�����:���4�}��R�G��hՆ�*d8L����4��-�>7�B
�+u��^���A�-�O�>�q��$8u��;?+�^�V�H�u��SHV��D�k4��Z�&�!X}����]6��������V��k�}n"Ok=�9C�YQeҲܕ.&쁞LH=�fw�,Zg���e���Dau�<
���^��!�,�8�3)J����a}K��,�bâYT��4�V����:�0UEp�X`��UT�q�߂�6�.�6�N v_�����fx�u����e��5�#��A9�/��>��v�"��<�� ������6ҫ��aܮcy����8�����4;83l9��K������ <���ϲI�L���|ϳ�5�T+.��=C�8���é����m�=��P�RZ�<�%��Ӿ�vfg�U�o��/��z��|��a��UT���"�i�e�"9��3�ú���n���x��z6��:t���x�X#��R�I8�O��� S5"�I�^���P=rk�>�K�W�R~�̾d��繄������@?4��,��)1,zJ�Q�Mi��R=T����4烙~��Y'I��N�;�wÒƦ�^�kr����B<����;a�}�����p��
�;����87��|8������l�Rل[�L�9?OS�"9���Ϫ۸wn�������~r���� ��O���Q�4�W�zMO�9�wu�%���HB.g��$�V�(�m�eGy5-<��Kj�Jj&��}�݇��O I+��Tmo������F��jx�.�0ܬ%w/Q�׳��$�`��.sO��܀G��Ng��fhTrgκ�@�U��6��8Eƴ�,%�^�Xr���$�~��%�u��8)��f"z�$�u����Z���@���GY�|�����P�8��[/s��1b�x�k�j�E��-�
S�!ccQ�X{UC97��Ka�0���v��b��t��Rx�2]d����q�������׫��|�W+�F,͒m=�ong�Ƭ6��a7q���%�l�y��T���b���rQ�I�t�ٍ��!��vI��n*���2�l�!o��O,��緓�l6�1��4K�w�m��Ǆ��6�m �fS����V�ٗ]��z�=���Ob����SV_v�,l�5�=��kS����0����!��xqn�aְ;E�A���U�l#y�*����A���{}�Sd�j	��v�G�!Ƞu��FΕ/��˾��[#��w*��qJ�ҡ-U%���uCM�|�wd^����y�\u����/��G�-�%��~�2f|����{��\����&L�,�#�~����a9]�
�B���#c��s�Db�B��y��ȶ�|�����p�uF��' ��׉��D(z���P:��W��U�4�;�ݡt�_��W��һ�oj��U�~�:D�DPޔV��Ӹ�	Q?���v���Y�\�W��E�5USn��m��ҭ��R�t�T��AOc�&��c�t���Z6���4�c��U?HV�m��N����.D�	ݝ�����R��8p�Ғ�b�8�x�%8�b?a�g՗�������#��o;�	(���`GZ�'���+����,��!Yc�`S������(�X�����{
�Ɇ�;d�y$�vz;Z�	�O�!bYN�X���rh'R�n�ZG��q}`:w{��f7�V�f�v�noܮ�w6뢸��h�5�(A��U�mz̤�-,v�@�4���HG��-Y�A_d�^�;P.O&�]�FDe�~��U	�Y�J��}7�~1�g0��o��2E# F����*��v`J�S�5�mBH:��[�$�#�=�$�Nh��7e;H�}�������>�)��I}
���a1oMb&ܬʭ�:���&e-*[%�4o��u�b�v�Mw�8h�uj[G�Keݮ�A��~��*���*u%�}��	��
E;��Fs��v��T\�����o�����!��do�4��� ]�͠�^P��q�ZkNyj��)@ӹ�ɲ=E<��Dr�Jw4��hٞ���n����Ɠ�h~�	�f��Vf��cL�T��P6ܠǚG�+"e��C�:�˕ލG~#��̍��A��ӟ� yfi�,E��H,��QU]���d��d��}��r�,`{,�5ySdm��#Z����l��2y�͖�?�v�N��,