"""Validate the loss tolerance ring, crash-safe checkpoints, resumed epoch logs and saved overlays."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from swinunet.evaluation import prediction_writer, spread_over_sources
    from swinunet.losses import HairlineCrackLoss
    from swinunet.training import open_epoch_log, save_checkpoint
except ImportError:  # the workflow tests still run without PyTorch
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.zeros(1, 1, 32, 32)
        self.target[0, 0, 16, 4:28] = 1  # a one-pixel horizontal crack on row 16
        self.valid = torch.ones_like(self.target)

    def gradient(self, tolerance):
        logits = torch.randn(1, 1, 32, 32, requires_grad=True)
        HairlineCrackLoss(label_tolerance=tolerance)(logits, self.target, self.valid).backward()
        return logits.grad[0, 0]

    def test_strict_loss_supervises_pixels_beside_the_label(self):
        self.assertTrue((self.gradient(0)[14:19, 4:28] != 0).all())

    def test_tolerance_ring_is_left_out_of_the_loss(self):
        gradient = self.gradient(2)
        self.assertTrue((gradient[[14, 15, 17, 18], 4:28] == 0).all())  # within 2 px of the label
        self.assertTrue((gradient[16, 4:28] != 0).all())                # the label itself
        self.assertTrue((gradient[[13, 19], 4:28] != 0).all())          # 3 px away: background again

    def test_checkpoint_is_replaced_whole(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            save_checkpoint({'epoch': 1}, path)
            save_checkpoint({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 2)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ['last.pt'])

    def test_resumed_log_drops_the_repeated_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training_history.csv'
            # Killed after logging epoch 3 but before saving last.pt, so epoch 3 runs again.
            path.write_text('epoch,loss\n1,0.9\n2,0.8\n3,0.7\n', encoding='utf-8')
            with open_epoch_log(path, ['epoch', 'loss'], 3) as handle:
                handle.write('3,0.6\n')
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss', '1,0.9', '2,0.8', '3,0.6'])
            with open_epoch_log(path, ['epoch', 'loss'], 1):  # a fresh run starts the file again
                pass
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss'])

    def test_saved_overlays_cover_every_source(self):
        domain_ids = [0] * 300 + [1] * 50 + [2] * 3  # one large source first, as in the size-grouped loader
        chosen = spread_over_sources(domain_ids, 40)
        per_source = [sum(domain_ids[index] == domain for index in chosen) for domain in range(3)]
        self.assertEqual(per_source, [19, 18, 3])  # the small source is saved whole, the rest shared
        self.assertGreater(max(index for index in chosen if domain_ids[index] == 0), 250)  # not just the first files
        self.assertEqual(spread_over_sources(domain_ids, 10**6), set(range(len(domain_ids))))  # a large limit saves all

    def test_overlays_are_written_for_the_selected_images_only(self):
        class Tiles:  # the parts of CrackSegmentationDataset the writer reads
            domain_ids = [0, 0, 0, 1]
            shapes = [(32, 32)] * 4
            pairs = [(Path(f'source{domain}_{index}.png'), None) for index, domain in enumerate(domain_ids)]

        with tempfile.TemporaryDirectory() as folder:
            write = prediction_writer(Tiles, Path(folder), 2, 0.5)
            write(torch.rand(4, 3, 32, 32), torch.rand(4, 1, 32, 32), self.target.expand(4, -1, -1, -1),
                  torch.arange(4))
            names = sorted(item.name for item in Path(folder).iterdir())
            self.assertEqual(names, ['source0_0_overlay.png', 'source0_0_probability.png',
                                     'source1_3_overlay.png', 'source1_3_probability.png'])


if __name__ == '__main__':
    unittest.main()
