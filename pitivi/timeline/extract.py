import gst
from pitivi.elements.singledecodebin import SingleDecodeBin
from pitivi.elements.arraysink import ArraySink
from pitivi.log.loggable import Loggable
import pitivi.utils as utils

class Extractee:
    """ Abstract base class for objects that receive raw data from an
        L{Extractor}."""
        
    def receive(self, array):
        """ Receive a chunk of data from an Extractor.
        
        @param array: The chunk of data as an array
        @type array: any kind of numeric array
        """
        raise NotImplementedError
    
    def finalize(self):
        """ Indicates that the extraction is complete, so the Extractee should
            process the data it has received. """
        raise NotImplementedError

class Extractor(Loggable):
    """ Abstract base class for extraction of raw data from a stream.
        Closely modeled on Previewer. """

    def __init__(self, factory, stream_):
        """ Create a new Extractor.
        
        @param factory: the factory with which to decode the stream
        @type factory: L{ObjectFactory}
        @param stream_: the stream to decode
        @type stream_: L{Stream}
        """
        Loggable.__init__(self)
        self.debug("Initialized with %s %s" % (factory, stream_))

    def extract(self, e, start, duration):
        """ Extract the raw data corresponding to a segment of the stream.
        
        @param e: the L{Extractee} that will receive the raw data
        @type e: L{Extractee}
        @param start: The point in the stream at which the segment starts (nanoseconds)
        @type start: L{long}
        @param duration: The duration of the segment (nanoseconds)
        @type duration: L{long}"""
        raise NotImplementedError

class RandomAccessExtractor(Extractor):
    """ Abstract class for L{Extractor}s of random access streams, closely
    inspired by L{RandomAccessPreviewer}."""
    def __init__(self, factory, stream_):
        Extractor.__init__(self, factory, stream_)
        # FIXME:
        # why doesn't this work?
        # bin = factory.makeBin(stream_)
        uri = factory.uri
        caps = stream_.caps
        bin = SingleDecodeBin(uri=uri, caps=caps, stream=stream_)

        self._pipelineInit(factory, bin)

    def _pipelineInit(self, factory, bin):
        """Create the pipeline for the preview process. Subclasses should
        override this method and create a pipeline, connecting to callbacks to
        the appropriate signals, and prerolling the pipeline if necessary."""
        raise NotImplementedError

class RandomAccessAudioExtractor(RandomAccessExtractor):
    """L{Extractor} for random access audio streams, closely
    inspired by L{RandomAccessAudioPreviewer}."""

    def __init__(self, factory, stream_):
        self.tdur = 30 * gst.SECOND
        self._queue = []
        RandomAccessExtractor.__init__(self, factory, stream_)

    def _pipelineInit(self, factory, sbin):
        self.spacing = 0

        self.audioSink = ArraySink()
        conv = gst.element_factory_make("audioconvert")
        self.audioPipeline = utils.pipeline({
            sbin : conv,
            conv : self.audioSink,
            self.audioSink : None})
        bus = self.audioPipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::segment-done", self._busMessageSegmentDoneCb)
        bus.connect("message::error", self._busMessageErrorCb)

        self._audio_cur = None
        self.audioPipeline.set_state(gst.STATE_PAUSED)
        import time; time.sleep(5)

    def _busMessageSegmentDoneCb(self, bus, message):
        self.debug("segment done")
        self._finishSegment()

    def _busMessageErrorCb(self, bus, message):
        error, debug = message.parse_error()
        print "Event bus error:", str(error), str(debug)

        return gst.BUS_PASS

    def _startSegment(self, timestamp, duration):
        self.debug("processing segment with timestamp=%i and duration=%i" % (timestamp, duration))
        self._audio_cur = timestamp, duration
        res = self.audioPipeline.seek(1.0,
            gst.FORMAT_TIME,
            gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_ACCURATE | gst.SEEK_FLAG_SEGMENT,
            gst.SEEK_TYPE_SET, timestamp,
            gst.SEEK_TYPE_SET, timestamp + duration)
        if not res:
            self.warning("seek failed %s", timestamp)
        self.audioPipeline.set_state(gst.STATE_PLAYING)

        return res

    def _finishSegment(self):
        samples = self.audioSink.samples
        e, start, duration = self._queue[0]
        e.receive(samples)
        self.audioSink.reset()
        start += self.tdur
        duration -= self.tdur
        if duration > 0:
            self._queue[0] = (e, start, duration)
        else:
            self._queue.pop(0)
            e.finalize()
        if len(self._queue) > 0:
            self._run()
    
    def extract(self, e, start, duration):
        stopped = len(self._queue) == 0
        self._queue.append((e, start, duration))
        if stopped:
            self._run()
    
    def _run(self):
        # Control flows in a cycle:
        # _run -> _startSegment -> busMessageSegmentDoneCb -> _finishSegment -> _run
        # This forms a loop that extracts one block in each cycle.  The cycle
        # runs until the queue of Extractees empties.  If the cycle is not
        # running, extract() will kick it off again.
        e, start, duration = self._queue[0]
        self._startSegment(start, min(duration, self.tdur))
            
