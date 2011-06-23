import numpy
from pitivi.timeline.extract import Extractee, RandomAccessAudioExtractor
from pitivi.stream import AudioStream
from pitivi.log.loggable import Loggable
from pitivi.timeline.alignalgs import affinealign

def nextpow2(n):
    i = 1
    while i < n:
        i *= 2
    return i

class EnvelopeExtractee(Extractee, Loggable):
    """ Class that computes the envelope of a 1-D signal
        (presumably audio).  The envelope is computed incrementally,
        so that the entire signal does not ever need to be stored. 
        
        The envelope is defined as the sum of the absolute value of the signal
        over a block."""

    def __init__(self, blocksize, callback, *cbargs):
        """
        @param blocksize: the number of samples in a block
        @type blocksize: L{int}
        @param callback: a function to call when the extraction is complete.
           The function's first argument will be a numpy array representing
           the envelope, and any later argument to this function will be passed
           as subsequent arguments to callback.
        """
        Loggable.__init__(self)
        self._blocksize = blocksize
        self._cb = callback
        self._cbargs = cbargs
        self._chunks = []
        self._leftover = numpy.zeros((0,))
        # self._leftover buffers up to blocksize-1 samples in case receive()
        # is called with a number of samples that is not divisible by blocksize
    
    def receive(self, a):
        if len(self._leftover) > 0:
            a = numpy.concatenate((self._leftover, a))
        lol = len(a) % self._blocksize # lol is leftover-length
        if lol > 0:
            self._leftover = a[-lol:]
            a = a[:-lol]
        else:
            self._leftover = numpy.zeros((0,)) # empty array means no leftover
        a = numpy.abs(a).reshape((len(a)//self._blocksize, self._blocksize))
        a = numpy.sum(a,1)
        # Relies on a being a floating-point type. If a is int16 then the sum
        # may overflow.
        self._chunks.append(a)
        self.debug("Chunk %i has size %i", len(self._chunks), len(a))
    
    def finalize(self):
        self.debug("Finalizing %i chunks", len(self._chunks))
        a = numpy.concatenate(self._chunks)
        a.tofile('/tmp/a%s' % self._cbargs[0])
        self._cb(a, *self._cbargs)

class AutoAligner(Loggable):
    """ Class for aligning a set of L{TimelineObject}s automatically based on
        their contents. """
    BLOCKRATE = 25

    @staticmethod
    def _getAudioTrack(to):
        for track in to.track_objects:
            if track.stream_type == AudioStream:
                return track
        return None

    def __init__(self, tobjects, callback):
        """
        @param tobjects: an iterable of L{TimelineObject}s.
            In this implementation, only L{TimelineObject}s with at least one
            audio track will be aligned.
        @type tobjects: iter(L{TimelineObject})
        @param callback: A function to call when alignment is complete.  No
            arguments will be provided.
        @type callback: function
        """
        Loggable.__init__(self)
        self._tos = dict.fromkeys(tobjects)
        # self._tos maps each object to its envelope.  The values are initially
        # None prior to envelope computation.
        self._callback = callback
    
    def _envelopeCb(self, array, to):
        self.debug("Receiving envelope for %s", to)
        self._tos[to] = array
        if not None in self._tos.itervalues(): # This was the last envelope
            self._performAlignment() 
    
    def start(self):
        for to in self._tos.iterkeys():
            a = self._getAudioTrack(to)
            if a is None:
                self._tos.remove(to)
                continue
            blocksize = a.stream.rate//self.BLOCKRATE # in units of samples
            e = EnvelopeExtractee(blocksize, self._envelopeCb, to)
            r = RandomAccessAudioExtractor(a.factory, a.stream)
            r.extract(e, a.in_point, a.out_point - a.in_point)
    
    def _chooseTemplate(self):
        # chooses the timeline object with lowest priority as the template
        def priority(to): return to.priority
        return min(self._tos.iterkeys(), key=priority)
    
    def _performAlignment(self):
        self.debug("performing alignment")
        template = self._chooseTemplate()
        tenv = self._tos.pop(template)
        # tenv is the envelope of template.  pop() also removes tenv and
        # template from future consideration.
        pairs = list(self._tos.iteritems())
        movables = [to for to,e in pairs]
        envelopes = [e for to,e in pairs]
        offsets, drifts = affinealign(tenv, envelopes, 0.015)
        for i in xrange(len(pairs)):
            # center_offset is the offset necessary to position the
            # middle of TimelineObject i correctly in the reference.
            # If we do not have the ability to speed up or slow down clips,
            # then this is the best shift to minimize the maximum error
            center_offset = offsets[i] + drifts[i]*len(envelopes[i])/2
            tshift = int((center_offset * int(1e9))/self.BLOCKRATE)
            # tshift is the offset rescaled to units of nanoseconds
            self.debug("Shifting %s to %i ns from %i", 
                             movables[i], tshift, template.start)
            newstart = template.start + tshift
            if newstart >= 0:
                movables[i].start = newstart
            else:
                # Timeline objects always must have a positive start point, so
                # if alignment would move an object to start at negative time,
                # we instead make it start at zero and chop off the required
                # amount at the beginning.
                movables[i].start = 0
                movables[i].in_point = movables[i].in_point - newstart
        self._callback()

    def _performShifts(self):
        self.debug("performing shifts")
        template = self._chooseTemplate()
        tenv = self._tos.pop(template)
        # tenv is the envelope of template.  pop() also removes tenv and
        # template from future consideration.
        # L is the maximum size of a cross-correlation between tenv and any of
        # the other envelopes.
        L = len(tenv) + max(len(e) for e in self._tos.itervalues()) - 1
        # We round up L to the next power of 2 for speed in the FFT.
        L = nextpow2(L)
        tenv -= numpy.mean(tenv)
        tenv = numpy.fft.rfft(tenv, L).conj()
        for movable, menv in self._tos.iteritems():
            # Estimates the relative shift between movable and template
            # by locating the maximum of the cross-correlation of their
            # (mean-subtracted) envelopes.
            menv -= numpy.mean(menv)
            # Compute cross-correlation
            xcorr = numpy.fft.irfft(tenv*numpy.fft.rfft(menv, L))
            xcorr.tofile('/tmp/xc%s' % movable)
            p = numpy.argmax(xcorr)
            # p is the shift, in units of blocks, that maximizes xcorr
            # WARNING: p maybe a numpy.int32, not a python integer
            if p > L - len(menv): # Negative shifts appear large and positive
                p -= L # This corrects them to be negative
            tshift = (int(p) * int(1e9))//self.BLOCKRATE
            # tshift is p rescaled to units of nanoseconds
            self.debug("Shifting %s to %i ns from %i", 
                             movable, tshift, template.start)
            newstart = template.start - tshift
            if newstart >= 0:
                movable.start = newstart
            else:
                # Timeline objects always must have a positive start point, so
                # if alignment would move an object to start at negative time,
                # we instead make it start at zero and chop off the required
                # amount at the beginning.
                movable.start = 0
                movable.in_point = movable.in_point - newstart
        self._callback()
