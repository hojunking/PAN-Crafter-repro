import unittest
from .report import row_equal,validate_corrected
class ReportTests(unittest.TestCase):
    def test_fr67_readback_and_omitted_blanks(self):
        r=['']*67;r[0]='Teacher';r[7]=.9637;r[62]='unique'
        self.assertTrue(row_equal(r,r));self.assertTrue(row_equal(r[:63],r))
        wrong=list(r);wrong[7]=.96;self.assertFalse(row_equal(wrong,r))
        self.assertFalse(row_equal(r+['foreign'],r))
    def test_incomplete_scenes_rejected(self):
        with self.assertRaisesRegex(ValueError,'FR20'):
            validate_corrected({'fr':{'per_scene':[]}}, {'per_scene':[]},'WV3')
if __name__=='__main__':unittest.main()
