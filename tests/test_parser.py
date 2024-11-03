import abc
from unittest import TestCase
from pathformer.datasets.path_dataset_v2 import parse_document, create_merged_2D_command, SVGPath

class TestParser(TestCase):

    def test_parse_document(self):
        #path = "/home/amor/Documents/code_dw/pathformer/dataset/www.svgrepo.com/show/475393/cigarette.svg"
        path = "/home/amor/Downloads/svg_data/output.svg"
        rez = parse_document(path)
        self.assertTrue(all(type(x) == dict for x in rez))

    def test_create_merged_2D_command(self):
        rez = create_merged_2D_command([complex(1, 2), complex(2, 3)])
        self.assertEqual(len(rez), 4)
        for i, expected in enumerate([1, 2, 2, 3]):
            self.assertEqual(rez[i]["value"], expected, i)

    def test_parse_nodes(self):
        attrs = {'d': 'M207.340874 688.760491', "fill": '#404041'}
        rez = SVGPath.parse_node(attrs)
        self.assertTrue(all(type(x) == dict for x in rez))