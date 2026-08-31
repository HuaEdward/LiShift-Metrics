import unittest

import cv2
import numpy as np

from libian_metrics.preprocess import preprocess


class PreprocessTests(unittest.TestCase):
    def test_square_padding_remains_binary_background(self):
        image = np.full((180, 120, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 20), (70, 160), (0, 0, 0), thickness=8)

        binary, skeleton, metadata = preprocess(image)

        self.assertEqual(binary.shape[0], binary.shape[1])
        self.assertEqual(binary[0, 0], 0)
        self.assertEqual(binary[-1, -1], 0)
        self.assertEqual(skeleton[0, 0], 0)
        self.assertIsInstance(metadata["quality_flag"], bool)


if __name__ == "__main__":
    unittest.main()
