import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seaanchor.data import nautilus


class ProjectTests(unittest.TestCase):
    def test_python_and_local_imports(self):
        for path in ROOT.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            compile(tree, str(path), 'exec')
            package = '.'.join(path.relative_to(ROOT).parts[:-1])
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                name = '.' * node.level + (node.module or '')
                if node.level:
                    name = importlib.util.resolve_name(name, package)
                if name == 'seaanchor' or name.startswith('seaanchor.'):
                    module = ROOT.joinpath(*name.split('.'))
                    self.assertTrue(module.with_suffix('.py').is_file() or (module / '__init__.py').is_file(), name)
                    module_file = module.with_suffix('.py') if module.with_suffix('.py').is_file() else module / '__init__.py'
                    definitions = ast.parse(module_file.read_text(encoding='utf-8'))
                    exported = {n.name for n in definitions.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                    for item in definitions.body:
                        if isinstance(item, (ast.Import, ast.ImportFrom)):
                            exported.update(a.asname or a.name for a in item.names)
                        if isinstance(item, (ast.Assign, ast.AnnAssign)):
                            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                            exported.update(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name))
                    for alias in node.names:
                        if alias.name == '*':
                            continue
                        self.assertTrue(alias.name in exported or (module / (alias.name + '.py')).is_file(), name + ':' + alias.name)

    def test_names_and_assets(self):
        forbidden = [''.join(['cau', 'vis']), ''.join(['up', 're'])]
        for path in ROOT.rglob('*'):
            if '__pycache__' in path.parts:
                continue
            self.assertFalse(any(word in path.name.lower() for word in forbidden), path)
            if path.is_file() and path.suffix in {'.py', '.txt', '.md', '.yaml'}:
                content = path.read_text(encoding='utf-8').lower()
                self.assertFalse(any(word in content for word in forbidden), path)
        self.assertGreater((ROOT / 'seaanchor/clip/bpe_simple_vocab_16e6.txt.gz').stat().st_size, 1000000)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'train.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--dinov3-weights', result.stdout)

    def fixture(self, root):
        annotations = root / nautilus.ANNOTATION_DIR
        annotations.mkdir()
        categories = [{'id': i, 'name': name} for i, name in enumerate(nautilus.NAUTILUS_CLASSES, 1)]
        for group in nautilus.ALL_GROUPS:
            image_dir = root / nautilus.IMAGE_DIR / group
            image_dir.mkdir(parents=True)
            Image.new('RGB', (32, 24)).save(image_dir / 'sample.png')
            Image.new('RGB', (32, 24)).save(image_dir / 'undeclared.png')
            content = {'categories': categories, 'images': [{'id': 7, 'file_name': 'sample.png', 'width': 1, 'height': 1}, {'id': 8, 'file_name': 'missing.png'}], 'annotations': [
                {'id': 1, 'image_id': 7, 'category_id': 1, 'bbox': [1, 1, 10, 10]},
                {'id': 2, 'image_id': 7, 'category_id': 1, 'bbox': [1, 1, -1, 10]},
                {'id': 3, 'image_id': 8, 'category_id': 1, 'bbox': [1, 1, 10, 10]},
            ]}
            (annotations / (group + '_annotations.json')).write_text(json.dumps(content), encoding='utf-8')

    def test_prepare_entry_from_other_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'train.py'), '--data-root', str(root), '--output-dir', str(root / 'logs'), '--prepare-only'], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            paths = nautilus.merged_annotation_paths(root)
            self.assertEqual(nautilus.summarize_coco_json(paths['train'])['images_with_boxes'], 4)
            self.assertEqual(nautilus.summarize_coco_json(paths['test'])['annotations'], 2)
            data = json.loads(Path(paths['train']).read_text(encoding='utf-8'))
            self.assertEqual([x['id'] for x in data['images']], [1, 2, 3, 4])
            self.assertTrue(all(x['width'] == 32 and x['height'] == 24 for x in data['images']))
            self.assertEqual(set(nautilus.summarize_coco_json(paths['train'])['domain_counts']), set(nautilus.TRAIN_GROUPS))
            self.assertEqual(nautilus.discover_thing_classes(root), list(nautilus.NAUTILUS_CLASSES))
            for group in nautilus.TEST_GROUPS:
                self.assertTrue((Path(paths['test']).parent / ('nautilus_test_' + group + '.json')).is_file())

    def test_strict_missing_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with self.assertRaises(FileNotFoundError):
                nautilus.prepare_merged_coco(root, strict_images=True)

    def test_invalid_category_schema(self):
        with self.assertRaises(ValueError):
            nautilus._validate_categories([{'id': 1, 'name': 'wrong'}])


if __name__ == '__main__':
    unittest.main()
