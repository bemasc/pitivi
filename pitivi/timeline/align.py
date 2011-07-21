# PiTiVi , Non-linear video editor
#
#       timeline/align.py
#
# Copyright (c) 2011, Benjamin M. Schwartz <bens@alum.mit.edu>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
# Boston, MA 02110-1301, USA.

"""
Classes for automatic alignment of L{TimelineObject}s
"""

import numpy
import array
import time
import gobject
import gst
from pitivi.utils import beautify_ETA, call_false
from pitivi.timeline.extract import Extractee, RandomAccessAudioExtractor
from pitivi.stream import AudioStream
from pitivi.log.loggable import Loggable
from pitivi.timeline.alignalgs import rigidalign


def getAudioTrack(to):
    """ Helper function for getting an audio track from a TimelineObject

    @param to: The TimelineObject from which to locate an audio track
    @type to: L{TimelineObject}
    @returns: An audio track from to, or None if to has no audio track
    @rtype: audio L{TrackObject} or L{NoneType}
    """
    for track in to.track_objects:
        if track.stream_type == AudioStream:
            return track
    return None


class ProgressMeter:
    """ Abstract interface representing a progress meter. """

    def addWatcher(self, r):
        """ Add a progress watching callback function.  This callback will
        always be called from the main thread.

        @param r: a function to call with progress updates.
        @type r: function(fractional_progress, time_remaining_text)
        """
        raise NotImplementedError


class ProgressAggregator(ProgressMeter):
    """ A ProgressMeter that aggregates progress reports from multiple
        sources into a unified progress report """

    def __init__(self):
        self._targets = []
        self._portions = []
        self._start = time.time()
        self._watchers = []

    def getPortionCB(self, target):
        """ Prepare a new input for the Aggregator.  Given a target size
            (in arbitrary units, but should be consistent across all calls on
            a single ProgressAggregator object), it returns a callback that
            can be used to update progress on this portion of the task.

            @param target: the total task size for this portion
            @type target: number
            @returns: a callback that can be used to inform the Aggregator of subsequent
                updates to this portion
            @rtype: function(x), where x should be a number indicating the absolute
                amount of this subtask that has been completed.
        """
        i = len(self._targets)
        self._targets.append(target)
        self._portions.append(0)

        def cb(thusfar):
            self._portions[i] = thusfar
            gobject.idle_add(self._callForward)
        return cb

    def addWatcher(self, w):
        self._watchers.append(w)

    def _callForward(self):
        t = sum(self._targets)
        p = sum(self._portions)
        if t == 0:
            return
        frac = min(1.0, float(p) / t)
        now = time.time()
        remaining = (now - self._start) * (1 - frac) / frac
        for w in self._watchers:
            w(frac, beautify_ETA(int(remaining * gst.SECOND)))
        return False


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
        self._blocks = numpy.zeros((0,), dtype=numpy.float32)
        self._empty = array.array('f', [])
        self._samples = array.array('f', [])
        self._threshold = 2000 * blocksize
        # self._leftover buffers up to blocksize-1 samples in case receive()
        # is called with a number of samples that is not divisible by blocksize
        self._progress_watchers = []

    def receive(self, a):
        self._samples.extend(a)
        if len(self._samples) < self._threshold:
            return
        else:
            self._process_samples()

    def addWatcher(self, w):
        """ Add a function to call with progress updates.

        @param w: callback function
        @type w: function(# of samples received so far)
        """
        self._progress_watchers.append(w)

    def _process_samples(self):
        excess = len(self._samples) % self._blocksize
        if excess != 0:
            a = self._samples[:-excess]
            self._samples = self._samples[-excess:]
        else:
            a = self._samples
            self._samples = array.array('f', [])
        self.debug("Adding %s samples to %s blocks", len(a), len(self._blocks))
        newblocks = len(a) // self._blocksize
        a = numpy.abs(a).reshape((newblocks, self._blocksize))
        self._blocks.resize((len(self._blocks) + newblocks,))
        self._blocks[-newblocks:] = numpy.sum(a, 1)
        # Relies on a being a floating-point type. If a is int16 then the sum
        # may overflow.
        for w in self._progress_watchers:
            w(self._blocksize * len(self._blocks) + excess)

    def finalize(self):
        self._process_samples()  # absorb any remaining buffered samples
        self._cb(self._blocks, *self._cbargs)


class AutoAligner(Loggable):
    """ Class for aligning a set of L{TimelineObject}s automatically based on
        their contents. """

    # The AutoAligner works by computing the "amplitude envelope" of each audio
    # stream.  We define an amplitude envelope as the absolute value of the
    # audio samples, downsampled to a low samplerate.  This samplerate, in Hz,
    # is given by BLOCKRATE.  (It is given this name because the downsampling
    # filter is implemented by very simple averaging over blocks, i.e. a
    # box filter.)  25 Hz appears to be a good choice because it
    # evenly divides all common audio samplerates (e.g. 11025 and 8000).
    # Lower blockrate requires less CPU time but produces less accurate
    # alignment.  Higher blockrate is the reverse (and also cannot evenly divide
    # all samplerates).
    BLOCKRATE = 25

    def __init__(self, timeline_objects, callback):
        """
        @param timeline_objects: an iterable of L{TimelineObject}s.
            In this implementation, only L{TimelineObject}s with at least one
            audio track will be aligned.
        @type timeline_objects: iter(L{TimelineObject})
        @param callback: A function to call when alignment is complete.  No
            arguments will be provided.
        @type callback: function
        """
        Loggable.__init__(self)
        # self._timeline_objects maps each object to its envelope.  The values
        # are initially None prior to envelope extraction.
        self._timeline_objects = dict.fromkeys(timeline_objects)
        self._callback = callback
        # stack of (Track, Extractee) pairs waiting to be processed
        # When start() is called, the stack will be populated, and then
        # processed sequentially.  Only one item from the stack will be
        # actively in process at a time.
        self._extraction_stack = []

    def _envelopeCb(self, array, timeline_object):
        self.debug("Receiving envelope for %s", timeline_object)
        self._timeline_objects[timeline_object] = array
        if self._extraction_stack:  # extract an envelope from the next track
            audiotrack, extractee = self._extraction_stack.pop()
            r = RandomAccessAudioExtractor(audiotrack.factory,
                                                              audiotrack.stream)
            r.extract(extractee, audiotrack.in_point,
                                     audiotrack.out_point - audiotrack.in_point)
        else:  # This was the last envelope
            self._performShifts()
            self._callback()

    def start(self):
        """ Initiate the auto-alignment process

        @returns: a L{ProgressMeter} indicating the progress of the alignment
        @rtype: L{ProgressMeter}
        """
        p = ProgressAggregator()
        pairs = []  # (TimelineObject, {audio}TrackObject) pairs
        for timeline_object in self._timeline_objects.keys():
            audiotrack = getAudioTrack(timeline_object)
            if audiotrack is not None:
                pairs.append((timeline_object, audiotrack))
            else:  # forget any TimelineObject without an audio track
                self._timeline_objects.pop(timeline_object)
        if len(pairs) >= 2:
            for timeline_object, audiotrack in pairs:
                # blocksize is the number of samples per block
                blocksize = audiotrack.stream.rate // self.BLOCKRATE
                extractee = EnvelopeExtractee(blocksize, self._envelopeCb,
                                                                timeline_object)
                # numsamples is the total number of samples in the track,
                # which is used by the ProgressAggregator (p) to determine
                # the percent completion.
                numsamples = ((audiotrack.duration / gst.SECOND) *
                                                         audiotrack.stream.rate)
                extractee.addWatcher(p.getPortionCB(numsamples))
                self._extraction_stack.append((audiotrack, extractee))
            audiotrack, extractee = self._extraction_stack.pop()
            r = RandomAccessAudioExtractor(audiotrack.factory,
                                                              audiotrack.stream)
            r.extract(extractee, audiotrack.in_point,
                                     audiotrack.out_point - audiotrack.in_point)
        else:  # We can't do anything without at least two audio tracks
            # After we return, call the callback function (once)
            gobject.idle_add(call_false, self._callback)
        return p

    def _chooseTemplate(self):
        # chooses the timeline object with lowest priority as the template
        def priority(timeline_object):
            return timeline_object.priority
        return min(self._timeline_objects.iterkeys(), key=priority)

    def _performShifts(self):
        self.debug("performing shifts")
        template = self._chooseTemplate()
        template_envelope = self._timeline_objects.pop(template)
        # pop() also removes the template and its envelope
        # from further consideration.
        pairs = list(self._timeline_objects.items())
        # We call list() because we need a reliable ordering of the pairs
        # (In python 3, dict.items() returns an unordered dictview)
        envelopes = [p[1] for p in pairs]
        offsets = rigidalign(template_envelope, envelopes)
        for (movable, envelope), offset in zip(pairs, offsets):
            tshift = int((offset * gst.SECOND) / self.BLOCKRATE)
            # tshift is the offset rescaled to units of nanoseconds
            self.debug("Shifting %s to %i ns from %i",
                             movable, tshift, template.start)
            newstart = template.start + tshift
            if newstart >= 0:
                movable.start = newstart
            else:
                # Timeline objects always must have a positive start point, so
                # if alignment would move an object to start at negative time,
                # we instead make it start at zero and chop off the required
                # amount at the beginning.
                movable.start = 0
                movable.in_point = movable.in_point - newstart
                movable.duration += newstart
