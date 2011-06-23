import numpy
def nextpow2(x):
    a = 1
    while a < x: a *= 2
    return a

def _findslope(a):
    # Helper function for driftalign
    # The provided matrix a contains a bright line whose slope we want to know,
    # against a noisy background.
    # The line starts at 0,0.  If the slope is positive, it runs toward the
    # center of the matrix (i.e. toward (-1,-1))
    # If the slope is negative, it wraps from 0,0 to 0,-1 and continues toward
    # the center, (i.e. toward (-1,0)).
    # The line segment terminates at the midline along the X direction.
    # We locate the line by simply checking the sum along each possible line
    # up to the Y-max edge of a.  The caller sets the limit by choosing the
    # size of a.
    # The function returns a floating-point slope assuming that the matrix
    # has "square pixels".
    Y,X = a.shape
    X /= 2
    x_pos = numpy.arange(1,X)
    x_neg = numpy.arange(2*X-1,X,-1)
    best_end = 0
    max_sum = 0
    for end in xrange(Y):
        y = (x_pos*end)//X
        s = numpy.sum(a[y,x_pos])
        if s > max_sum:
            max_sum = s
            best_end = end
        s = numpy.sum(a[y,x_neg])
        if s > max_sum:
            max_sum = s
            best_end = -end
    return float(best_end)/X

def affinealign(reference, targets, max_drift=0.02):
    """ Perform an affine registration between a reference and a number of 
    targets.  Designed for aligning the amplitude envelopes of recordings of
    the same event by different devices.
    
    @param reference: the reference signal to which others will be registered
    @type reference: array(number)
    @param targets: the signals to register
    @type targets: ordered iterable(array(number))
    @param max_drift: the maximum absolute clock drift rate
                  (i.e. stretch factor) that will be considered during search
    @type max_drift: positive L{float}
    @return: (offsets, drifts).  offsets[i] is the point in reference at which
           targets[i] starts.  drifts[i] is the speed of targets[i] relative to
           the reference (positive is faster, meaning the target should be
           slowed down to be in sync with the reference)
    """
    L = len(reference) + max(len(t) for t in targets) - 1
    L2 = nextpow2(L)
    bsize = int(20./max_drift) #NEEDS TUNING
    num_blocks = nextpow2(1.0*len(reference)//bsize) #NEEDS TUNING
    bspace = (len(reference)-bsize)//num_blocks
    reference -= numpy.mean(reference)

    # Construct FFT'd reference blocks
    freference_blocks = numpy.zeros((L2/2 + 1,num_blocks),dtype=numpy.complex)
    for i in xrange(num_blocks):
        s = i*bspace
        tmp = numpy.zeros((L2,))
        tmp[s:s+bsize] = reference[s:s+bsize]
        freference_blocks[:,i] = numpy.fft.rfft(tmp, L2).conj()
    freference_blocks[:10,:] = 0 # High-pass to ignore slow volume variations

    offsets = []
    drifts = []
    for t in targets:
        t -= numpy.mean(t)
        ft = numpy.fft.rfft(t, L2)
        #fxcorr is the FFT'd cross-correlation with the reference blocks
        fxcorr_blocks = numpy.zeros((L2/2 + 1,num_blocks),dtype=numpy.complex)
        for i in xrange(num_blocks):
            fxcorr_blocks[:,i] = ft * freference_blocks[:,i]
            fxcorr_blocks[:,i] /= numpy.sqrt(numpy.sum(fxcorr_blocks[:,i]**2))
        del ft
        # At this point xcorr_blocks would show a distinct bright line, nearly
        # orthogonal to time, indicating where each of these blocks found their
        # peak.  Each point on this line represents the time in t where block i
        # found its match.  The time-intercept gives the time in b at which the
        # reference starts, and the slope gives the amount by which the
        # reference is faster relative to b.
        
        # The challenge now is to find this line.  Our strategy is to reduce the
        # search to one dimension by first finding the slope.
        # The Fourier Transform of a smooth real line in 2D is an orthogonal
        # line through the origin, with phase that gives its position.
        # Unfortunately this line is not clearly visible in fxcorr_blocks, so
        # we discard the phase (by taking the absolute value) and then inverse
        # transform.  This places the line at the origin, so we can find its
        # slope.
        
        # Construct the half-autocorrelation matrix
        # (A true autocorrelation matrix would be ifft(abs(fft(x))**2), but this
        # is just ifft(abs(fft(x))).)
        # Construction is stepwise partly in an attempt to save memory
        # The width is 2*num_blocks in order to avoid overlapping positive and
        # negative correlations
        halfautocorr = numpy.fft.fft(fxcorr_blocks, 2*num_blocks, 1)
        halfautocorr = numpy.abs(halfautocorr)
        halfautocorr = numpy.fft.ifft(halfautocorr,None,1)
        halfautocorr = numpy.fft.irfft(halfautocorr,None,0)
        # Now it's actually the half-autocorrelation.
        # Chop out the bit we don't care about
        halfautocorr = halfautocorr[:bspace*num_blocks*max_drift,:]
        # Remove the local-correlation peak.
        halfautocorr[-1:2,-1:2] = 0 #NEEDS TUNING
        # Normalize each column (appears to be necessary)
        for i in xrange(2*num_blocks):
            halfautocorr[:,i] /= numpy.sqrt(numpy.sum(halfautocorr[:,i]**2))
        #from matplotlib.pyplot import imshow,show
        #imshow(halfautocorr,interpolation='nearest',aspect='auto');show()
        drift = _findslope(halfautocorr)/bspace
        del halfautocorr

        #inverse transform and shift everything into alignment
        xcorr_blocks = numpy.fft.irfft(fxcorr_blocks, None, 0)
        del fxcorr_blocks
        #TODO: see if phase ramps are worthwhile here
        for i in xrange(num_blocks):
            blockcenter = i*bspace + bsize/2
            shift = int(blockcenter*drift)
            if shift > 0:
                temp = xcorr_blocks[:shift,i].copy()
                xcorr_blocks[:-shift,i] = xcorr_blocks[shift:,i].copy()
                xcorr_blocks[-shift:,i] = temp
            elif shift < 0:
                temp = xcorr_blocks[shift:,i].copy()
                xcorr_blocks[-shift:,i] = xcorr_blocks[:shift,i].copy()
                xcorr_blocks[:-shift,i] = temp

        #from matplotlib.pyplot import imshow,show
        #imshow(xcorr_blocks,interpolation='nearest',aspect='auto');show()
        
        # xcorr is the drift-compensated cross-correlation
        xcorr = numpy.sum(xcorr_blocks, axis=1)
        del xcorr_blocks

        offset = numpy.argmax(xcorr)
        #from matplotlib.pyplot import plot,show
        #plot(xcorr);show()
        del xcorr
        if offset >= len(t):
            offset -= L2

        # now offset is the point in target at which reference starts and
        # drift is the speed with which the reference drifts relative to the
        # target.  We reverse these relationships for the caller.
        slope = 1 + drift
        offsets.append(-offset/slope)
        drifts.append(1/slope - 1)
    return offsets, drifts

if __name__ == '__main__':
    from sys import argv
    names = argv[1:]
    envelopes = [numpy.fromfile(n) for n in names]
    reference = envelopes[-1]
    offsets, drifts = affinealign(reference,envelopes,0.02)
    print offsets, drifts
    from matplotlib.pyplot import *
    clf()
    for i in xrange(len(envelopes)):
        t = offsets[i] + (1 + drifts[i])*numpy.arange(len(envelopes[i]))
        plot(t, envelopes[i]/numpy.sqrt(numpy.sum(envelopes[i]**2)))
    show()
