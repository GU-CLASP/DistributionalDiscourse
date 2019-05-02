import unittest
from preproc import tokenize

class TestTokenizer(unittest.TestCase):
    def test_punkt(self):
        self.assertEqual(tokenize("uh-huh.  /"),"uh-huh . /")
        self.assertEqual(tokenize("# [ wh-, + what ] happens if you were to fail?  /"), "# [ wh - , + what ] happens if you were to fail ? /")
        self.assertEqual(tokenize("{f huh? } /"),  "{f huh ? } /")
        self.assertEqual(tokenize("i hadn't heard that. /"), "i had n't heard that . /")
        self.assertEqual(tokenize("{f uh, } as far as my defense budget, {f uh, } they're cutting  it back now, what, twenty-five percent?  /"), "{f uh , } as far as my defense budget , {f uh , } they 're cutting it back now , what , twenty-five percent ? /")
    def test_laughter(self):
        self.assertEqual(tokenize("{c but } again i'd like to see something on the other <laughter> end back into education. but not in the education we have today.  /", laughters=False), "{c but } again i 'd like to see something on the other end back into education . but not in the education we have today . /")
    def test_disfl(self):
        self.assertEqual(tokenize("{f huh? } /", disfluencies=False),  "huh ? /")        
        
