from wssd_pw_common import *

class GC_data(object):

    def __init__(self, fn_GC_DTS, contig, fn_contigs):

        self.contig = contig
        self.wnd_GC = DenseTrackSet(fn_contigs,
                                    fn_GC_DTS,
                                    overwrite=False,
                                    openMode='r')
        
        self.wnd_starts = self.wnd_GC["starts"][contig][:]
        self.wnd_ends = self.wnd_GC["ends"][contig][:]
        self.GC = self.wnd_GC["GC"][contig][:]
    
    def get_GC(self, contig, start, end):
        assert contig == self.contig

        start_idx = np.searchsorted(self.wnd_starts, start)
        end_idx = np.searchsorted(self.wnd_ends, end)
        
        if end_idx<=start_idx:
            end_idx = start_idx+1
        elif end_idx-start_idx>1:
            start_idx+=1
        
        return np.mean(self.GC[start_idx:end_idx])
         
