import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset


class RegDB(BaseImageDataset):
    """RegDB visible-infrared dataset.

    Expected split files:
        <root>/RegDB/idx/train_visible_<split>.txt
        <root>/RegDB/idx/train_infrared_<split>.txt
        <root>/RegDB/idx/test_visible_<split>.txt
        <root>/RegDB/idx/test_infrared_<split>.txt
    """

    dataset_dir = 'RegDB'

    def __init__(self, root='', verbose=True, cfg=None, **kwargs):
        super(RegDB, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self._resolve_dataset_dir(self.root)
        self.idx_dir = osp.join(self.dataset_dir, 'idx')
        split = 1
        query_modality = 'thermal'
        if cfg is not None:
            split = int(getattr(cfg.DATASETS, 'REGDB_SPLIT', 1))
            query_modality = str(getattr(cfg.DATASETS, 'REGDB_QUERY_MODALITY', 'thermal')).strip().lower()
        if query_modality in ('infrared', 'ir'):
            query_modality = 'thermal'
        if query_modality not in ('visible', 'thermal'):
            raise ValueError(
                "DATASETS.REGDB_QUERY_MODALITY must be 'visible' or 'thermal', got {}".format(query_modality))

        train_visible_path = self._split_path('train_visible', split)
        train_infrared_path = self._split_path('train_infrared', split)
        test_visible_path = self._split_path('test_visible', split)
        test_infrared_path = self._split_path('test_infrared', split)
        self._check_before_run([
            train_visible_path, train_infrared_path,
            test_visible_path, test_infrared_path,
        ])

        train_visible_raw = self._read_split(train_visible_path, modality=0)
        train_infrared_raw = self._read_split(train_infrared_path, modality=1)
        train_pids = sorted({pid for _, pid, _, _ in train_visible_raw + train_infrared_raw})
        pid2label = {pid: label for label, pid in enumerate(train_pids)}
        train_visible = self._group_by_pid(train_visible_raw, pid2label=pid2label)
        train_infrared = self._group_by_pid(train_infrared_raw, pid2label=pid2label)
        train = self._make_pairs(train_visible, train_infrared)

        test_visible = self._read_split(test_visible_path, modality=0, duplicate=True)
        test_infrared = self._read_split(test_infrared_path, modality=1, duplicate=True)
        if query_modality == 'thermal':
            query, gallery = test_infrared, test_visible
        else:
            query, gallery = test_visible, test_infrared

        if verbose:
            print("=> RegDB loaded (split {}, {} query)".format(split, query_modality))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    @classmethod
    def _resolve_dataset_dir(cls, root):
        candidates = [osp.join(root, cls.dataset_dir), root]
        for path in candidates:
            if osp.isdir(osp.join(path, 'idx')):
                return path
        return candidates[0]

    def _split_path(self, stem, split):
        candidates = [
            osp.join(self.idx_dir, '{}_{}.txt'.format(stem, split)),
            osp.join(self.dataset_dir, '{}_{}.txt'.format(stem, split)),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        return candidates[0]

    @staticmethod
    def _check_before_run(paths):
        for path in paths:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    def _resolve_image_path(self, img_path):
        if osp.isabs(img_path):
            return img_path
        cleaned = img_path.lstrip('./')
        candidates = [
            osp.join(self.dataset_dir, cleaned),
            osp.join(self.root, cleaned),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        return candidates[0]

    def _read_split(self, split_path, modality, duplicate=False):
        camid = 0 if modality == 0 else 1
        dataset = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError("Invalid RegDB split line in {}: {}".format(split_path, line))
                img_path = self._resolve_image_path(parts[0])
                pid = int(parts[1])
                path = [img_path, img_path] if duplicate else img_path
                dataset.append((path, pid, camid, modality))
        return dataset

    @staticmethod
    def _group_by_pid(dataset, pid2label=None):
        grouped = defaultdict(list)
        for img_path, pid, camid, modality in dataset:
            label = pid2label[pid] if pid2label is not None else pid
            grouped[label].append((img_path, label, camid, modality))
        return grouped

    @staticmethod
    def _make_pairs(visible_by_pid, infrared_by_pid):
        dataset = []
        for pid in sorted(set(visible_by_pid) & set(infrared_by_pid)):
            visible = visible_by_pid[pid]
            infrared = infrared_by_pid[pid]
            pair_count = max(len(visible), len(infrared))
            for idx in range(pair_count):
                vis_path, label, vis_cam, _ = visible[idx % len(visible)]
                ir_path, _, _, _ = infrared[idx % len(infrared)]
                dataset.append(([vis_path, ir_path], label, vis_cam, 0))
        return dataset

