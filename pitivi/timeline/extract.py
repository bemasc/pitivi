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
        self.debug("Initialized with %s %s", factory, stream_)

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
        self._ready = False

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
        self._donecb_id = bus.connect("message::async-done",
                                                self._busMessageAsyncDoneCb)

        self.audioPipeline.set_state(gst.STATE_PAUSED)
        # The audiopipeline.set_state() method does not take effect immediately,
        # but the extraction process (and in particular self._startSegment) will
        # not work properly until self.audioPipeline reaches the desired
        # state (STATE_PAUSED).  To ensure that this is the case, we wait until
        # the ASYNC_DONE message is received before setting self._ready = True,
        # which enables extraction to proceed.

    def _busMessageSegmentDoneCb(self, bus, message):
        self.debug("segment done")
        self._finishSegment()

    def _busMessageErrorCb(self, bus, message):
        error, debug = message.parse_error()
        print "Event bus error:", str(error), str(debug)

        return gst.BUS_PASS

    def _busMessageAsyncDoneCb(self, bus, message):
        self.debug("Pipeline is ready for seeking")
        bus.disconnect(self._donecb_id) #Don't call me again
        self._ready = True
        if len(self._queue) > 0: #Someone called .extract() before we were ready
            self._run()

    def _startSegment(self, timestamp, duration):
        self.debug("processing segment with timestamp=%i and duration=%i",
                                                            timestamp, duration)
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
        # Pull the raw data from the array sink
        samples = self.audioSink.samples
        e, start, duration = self._queue[0]
        # transmit it to the Extractee
        e.receive(samples)
        self.audioSink.reset()
        # Chop off that bit of the segment
        start += self.tdur
        duration -= self.tdur
        # If there's anything left of the segment, keep processing it.
        if duration > 0:
            self._queue[0] = (e, start, duration)
        # Otherwise, throw it out and finalize the extractee.
        else:
            self._queue.pop(0)
            e.finalize()
        # If there's more to do, keep running
        if len(self._queue) > 0:
            self._run()
    
    def extract(self, e, start, duration):
        stopped = len(self._queue) == 0
        self._queue.append((e, start, duration))
        if stopped and self._ready:
            self._run()
        # if self._ready is False, self._run() will be called from
        # self._busMessageDoneCb().
    
    def _run(self):
        # Control flows in a cycle:
        # _run -> _startSegment -> busMessageSegmentDoneCb -> _finishSegment -> _run
        # This forms a loop that extracts one block in each cycle.  The cycle
        # runs until the queue of Extractees empties.  If the cycle is not
        # running, extract() will kick it off again.
        e, start, duration = self._queue[0]
        self._startSegment(start, min(duration, self.tdur))
            
