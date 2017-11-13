
class Segment(object):
    """A Segment taken from a segmentation file, representing the smallest"""

    def __init__(self, seg):
        """
        :type seg: string
        :param seg: the line taken from a seg file"""
        self.fileName = str(seg[0])
        self.start = int(seg[2])
        self.duration = int(seg[3])
        self.gender = str(seg[4])
        self.env = str(seg[5])
        self.speaker = str(seg[7])
        self.seg = seg


    def merge(self, adjSeg):
        """Merge two adjacent segments, the otr in to the original. """
        self.duration = adjSeg.start() - self.start
        self.duration += adjSeg.get_duration()
        self.seg[3] = self.duration

    def rename(self, speaker):
        """Change the speaker of the segment. """
        self.seg[7] = self.speaker = speaker

    def get_end(self):
        return self.start + self.duration

