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
        return len(format(n, "b")) + 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ¿AhŸ}æÄ70/m ~A¢“µ©ƒ’âAòÑ)9Â»*ŒJ2.xPCá§í{>ßÉsŒXRa›9w¾á&N½¶­…:OZ"–®ût¦:ëØOsÓ\GWæÁÑ^¢W¥£Ïkq=Æ¿‚¯î:ÒÏGò|J<ŸõÇ2)>­û0ôÕÛŠŒïbD¢Ù`¸âB”şÒÃLy¼rr{§x{—Ô²„
ÿ&?œÊªIØ°S
^°svJ:¶‹ªóƒ¶à1<"s¼£0]Oä¡Ä„}ÃóŞŞ±c÷Ï4z±á‰–y…*rH³
õX®z÷éU¿‘/ê–?¸/·§+©·É)³
ö’TÅQÔË6tğ†ã¯!ˆw‚]ŸÊÓe÷Åh+mx(&Úq+Ÿ°]%ÃRœsæH¾PÒì9õmmA«¯Íjş÷@uVŒ×pN^¼ëm¨;5ÈÁFjË¦âm;¡×Gš€Ó\œÚ>­Ë7Z]´Î¢Ã–Ñ)¿±ˆšø‡®?Yö<\Ã@ÊÖ»„zdhÒçH‰ÛB®†_+ıÑÇ“º«ª|j>«ñ%çöJ·ÅpÀÂ‹˜^õ`œ3ø»ÇÇö]ºğ³Ö$èÍ¢{kûoÿvëµ’…·î4ığH­
Ø-ùë‡ƒ‘‰”ªjluß pŸTvŸ{	ôÛu&š¶¥K›‰BÂÅ­Ñ9+Ã\bÕâ½Ñ®
Rè·¯m(
?µ¥ëhá1r­m„8÷·´'mHk3`[Úaƒp°<=1×¨ öêL©R0B+ÛQ6ÏÖğ¯t— ØÿØÇñu¯ú©½çWï†/·mhGÕeB{,­ƒ_ìY¦çW>ËD"l–Ékvï³LxÔ…†CN…ôŞ€g™¨V-Ï2yMîz–ÉÇ`^ãñŠ_øY&´Kı,SØ1ñ,?ñ ½ãY&ËÏ2uºĞ`vQPlÀóáa²—YpC®ÕÂ=}Û’şv÷‹¾"0†ïd[»„»À_îf‡^›Ûó
´Íw€Å'ıúà¶™„¿I„´ú\{­fa+oÍ61<òh`ûfÔE~Ş·&à×C›Å­m‹‰8ÉœÎÆ‰`…‹¿ÙûHü¹`•Y¿ï‚|–}àÿ`
È%ÿ[ƒ]8–@‚%"ãOŸî»Ï!`è„{fÖ„7ÀvŒäèúp’›Ü5Îÿ¬Ğ	ÕƒÀ–OøÈŠv=øÜ\$ıàúPIc'@ïæ¦sl¥¦Ş!—>|ìt¹H7-;¶~šÛ{ïv[ÛÓq¼^ŠßÑôŞñ½ğ!:é7o:ß=È»gmz½“›:Ï§:tD7è"³vâwBgaó‡‡²yğaˆ…êMS\ï1oÃô—â*mî¡·¡Ô@ŸM}ÿşÂ = éœ ßÔê²D~du†è@ÃlâÑÿ›âëŸUÊÿµÃ.á0İ‡d0‹
q¹ëŸÒà¨'xÇ`ğUĞİãÁWA­P=¼éhR‡6P« ¸>TR6zæ¿it#7Á*¨¦Ø*¨•« ¤Ê‡tj7Íb
­M¢x*t¯×º§Yï%Ûmm_lBö]²İŞôŞñ½ğ|î¿d»§ñİƒ¼ÛĞŞ[²¹®oĞ1‚%[7şÖ%›HO4–l Y²M"şDé®¯:¦n¸ Ai…«6
+±pëûwI.ÜÚ ;nÅ75}o*fÕ·Õù°1_áÀG8÷Ø`¼À"Ï+ö)Şvå²õ?ìx0o‹|½É˜6™CZ2Ó’¾çÊwQ&—å»?GQ½òPè3™îÊı”°1‰ƒøgôæ“ü =Œkñ…£µÜ¤ùÌv´guŸ‘‡÷µ:ÓşŠ4·}“ôR÷G‚‘hÕ†Å*d8L¼¨è4éê-…>7›B
½+uï^¥öA£-úO¥>ò»qîŞ$8u‚‚;?+À^VŠH¸u´SHV´àDÄk4“ëZç…&ô!X}ÁíĞ]6àå™ĞÙµáğ½ªVÌœk‚}n"Ok=ç9C YQeÒ²Ü•.&ìLH=Ôfw,ZgĞÂ×e…›šDauë<
»†Õ^ßõ!ä¬,ó8Û3)Jˆº…¥a}K§­,bÃ¢YT·²4ìV‹¾™æ·:Õ0UEp½X`àşUT¨qÈß‚è6ç.Å6ÃN v_¡ÔçıæfxàuÉí÷’eãñ‘5±#’ŠA9ß/ğ©Ê>æå¹v˜"§‘<¬¥ €µô‹ü‘à6Ò«½ÜaÜ®cy¾Ã•Ò8Îƒñ‚ÿÏ4;83l9ÿŞK„²¤¡˜ç <íêÎÏ²IêL¾Öä‰|Ï³Ã5‘T+.é¿ú=C 8£ò«©ñ¯³Ã©ù®ñìmŠ=áúPßRZ¾<‹%¯ôÓ¾Êvfg‚U‘o¤Š/Æz|¼ÀaİúUT·ñÃ"âiñeı"9õª3€Ãºõ«¨nå«æx¼˜z6ÖÂ:tÊÉíºxúX#Ò¡RI8¬O¿ŠÌ S5"ßI¤^ŸóÁP=rk¨>åK‹WõR~´Ì¾dãÍç¹„—¡¯ı¦¦@?4„Í,èÄ)1,zJËQ·Mi›©R=T¦é÷Ê4ï¤–~¼ŒY'I€¶Nò¯;­wÃ’Æ¦ƒ^Ôkræ²ÙB<ùïêÕ;aü}ôº­çãp˜·
Ô;‰£í87²Ÿ|8±ú«’¿œlÅRÙ„[âLÓ9?OS‰"9ü¸ÏªÛ¸wnºŒ÷“¡óç~rñ¸ ğ ¸‹O€ÊÇQä4ÚWîzMOà9ÿwuïŒ˜Á% ÂÖ™HB.gäû$§Vß(êm•eGy5-<à…KjöJj&Æ}åİ‡³äO I+›ÅTmoö¬ú˜ä¸F¢çjxÔ.†0Ü¬%w/Qİ×³œú$Ğ`×ş.sO£òÜ€Gş Ngô‘fhTrgÎº÷@ìUıÄ6Ñïˆ8EÆ´Ù,%‹^…XrÆÛä$â~ÎÕ%çu¬ë8)˜í·›f"zë$u°ŸùÇZÁô—@òÿÂGY—|Š‹º§ÓPı8ç¼È[/súÜ1bêxÈk±júEüÚ-˜
S¼!ccQÖX{UC97îªÕKaÃ0˜åËv¹˜b»‡t·‹Rx¢2]d«í  q»ÊâÍ‚ºü×«ÏÍ|ÆW+²F,Í’m=àongéÆ¬6Ûìa7qñà„%‹l’yı¡TÍæñb¥¡ôrQ·IétƒÙíî!›ÂvI¶Ùn*œ¶İ2›læ!o±˜O,Óüç·“Õl6‹1êâ4KƒwÒméÄÇ„—Í6«m ÚfS¶˜½şVÆÙ—]•²zÀ=«˜ğOb£º©òSV_v¿,l‰5Ê=Á‹kS¢µüÒ0ë£ü¦è¿!î±Æxqn«aÖ°;EõA¨ëæUğl#y±*„´ô‹ï›Aêïè{}‘Sd¶j	ŞÅv±Gì!È uËøFÎ•/ïñË¾ˆ[#ŒÒw*ÉŸqJ•Ò¡-U%ŸÁÁuCMŸ|ÿwd^œÍÓìyˆ\u›¿Äóï‡/şGß-éš%Àş~Ş2f|ûÿè¿{ŠÍ\°”‹ù&LÑ,ò#±~¯’æa9]¨
€Bêş’#cĞsÌDbıBõ€yÚüÈ¶˜|ŞÜŞèæpuFµÚ' †à×‰øëD(z×ÖìP:ØúWûUãº4®;´İ¡t‚_õîW½ëÒ»Îoj¶ÿUë~Õ:DëDPŞ”VÅØÓ¸ê	Q?”ÿÇvŒé—ÜYÿ\òWÄÓEš5USnş’m˜‡Ò­ç‡ç‘RÃtİT„êAOcû&¹–câtÌüâZ6ø„à4àc†U?HVğm±N£’Ÿé.DÚ	İÈü¿õ‹RÃï8pïÒ’ÜbîŒ8¯x¶%8»b?aµgÕ—ù€À¯Ûóä#ùÒo;è	(èóÈ`GZ„'ºâ+Ÿ„Ÿ,õ•!Yc¾`Sõ÷İßÿ(œX£õ®‹Ì{
úÉ†ğ;d’y$ìvz;ZÄ	ğOÓ!bYNÉXÕçÛrh'RñnZGƒ§q}`:w{»Öf7ëV¯f´vµnoÜ®Ôw6ë¢¸£µhÜ5Ÿ(Aáó¢U­mzÌ¤í’-,vü@›4”ï‰¾HGæó°-Y—A_dƒ^ì¡;P.O&â]©FDe~ ïU	ñYãJ°Ê}7×~1g0áÀoûæ2E# Fˆã§ÑÜ*ÉÎv`JSÂ5õmBH:€ü[ø$Ã#ñ=™$ÈNhè˜7e;H×}ö‚êÅß¶û>Å)ß÷I}
Ğßğ­a1oMb&Ü¬Ê­:õ˜·&e-*[%ì4oƒğu—bìv–Mwˆ8hÅuj[GÒKeİ®ºAº‡~‹¦*¾øš*u%À}Ûù	„§
E;ŠŸFs³ÖvµëT\…€”¸®o•¶¤È!Íğdo·4Á´û ]ë¥Í Ï^P½˜qƒZkNyj­ô)@Ó¹„É²=E<†Dr³Jw4ëÔhÙ”»ªn•·‡¢Æ“Éh~–	Õf‰…Vf¯¾cL½TÙï°P6Ü ÇšG+"eŞ“C·:áË•ŞG~#ê¤Ì£şA³ÖÓŸØ yfiÎ,EşñH,ã¿îQU]¡¹ûdÃî¾dˆİ}£ïr©,`{,Û5ySdmòÜ#Z‹ğğ¨ƒÆlâÃ2yÛÍ–‚?‘vªNõ®,