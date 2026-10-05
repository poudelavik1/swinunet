"""Validate configuration overrides and reading a live, partially written CSV."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowTests(unittest.TestCase):
    def test_configuration_preserves_cli_overrides_and_dataset(self):
        train = load_script('train')
        args = train.configured_arguments(ROOT / 'configs/training.toml',
                                          ['/external/dataset', '--epochs', '2'])
        self.assertEqual(args[-3:], ['/external/dataset', '--epochs', '2'])
        self.assertEqual(args[:2], ['--epochs', '80'])

    def test_live_csv_skips_incomplete_epoch(self):
        monitor = load_script('monitor_epochs')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training_history.csv'
            path.write_text('epoch,val_f1\n1,0.7\n2,0.', encoding='utf-8')
            self.assertEqual(monitor.read_rows(path), [{'epoch': '1', 'val_f1': '0.7'}])
            path.write_text('epoch,val_f1\n1,0.7\n2,0.8\n', encoding='utf-8')
            self.assertEqual(len(monitor.read_rows(path)), 2)

    def test_invalid_configuration_is_rejected(self):
        train = load_script('train')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.toml'
            path.write_text('[training]\nbatch_size = 0\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                train.configured_arguments(path, [])


if __name__ == '__main__':
    unittest.main()
