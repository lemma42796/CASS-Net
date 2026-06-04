import glob
import os.path as osp
import re
from collections import defaultdict

from .bases import BaseImageDataset


class SYSUMM01(BaseImageDataset):
    """SYSU-MM01 visible-infrared dataset.

    Expected layout:
        <root>/SYSU-MM01/cam1/<pid>/*.jpg
        ...
        <root>/SYSU-MM01/cam6/<pid>/*.jpg
        <root>/SYSU-MM01/exp/train_id.txt
        <root>/SYSU-MM01/exp/val_id.txt
        <root>/SYSU-MM01/exp/test_id.txt

    Training samples are same-identity visible/infrared pairs. Evaluation uses
    infrared query images and visible gallery images.
    """

    dataset_dir = 'SYSU-MM01'
    visible_cams = (1, 2, 4, 5)
    infrared_cams = (3, 6)
    indoor_visible_cams = (1, 2)

    def __init__(self, root='', verbose=True, cfg=None, **kwargs):
        super(SYSUMM01, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self._resolve_dataset_dir(self.root)
        self.exp_dir = osp.join(self.dataset_dir, 'exp')
        self.eval_mode = 'all'
        if cfg is not None:
            self.eval_mode = str(getattr(cfg.DATASETS, 'SYSU_EVAL_MODE', 'all')).strip().lower()
        if self.eval_mode not in ('all', 'indoor'):
            raise ValueError("DATASETS.SYSU_EVAL_MODE must be 'all' or 'indoor', got {}".format(self.eval_mode))

        self._check_before_run()
        train_ids = self._read_ids(osp.join(self.exp_dir, 'train_id.txt'))
        val_path = osp.join(self.exp_dir, 'val_id.txt')
        if osp.exists(val_path):
            train_ids += self._read_ids(val_path)
        train_ids = sorted(set(train_ids))
        test_ids = sorted(set(self._read_ids(osp.join(self.exp_dir, 'test_id.txt'))))

        pid2label = {pid: label for label, pid in enumerate(train_ids)}
        train_visible = self._collect_by_pid(train_ids, self.visible_cams, pid2label=pid2label, modality=0)
        train_infrared = self._collect_by_pid(train_ids, self.infrared_cams, pid2label=pid2label, modality=1)
        train = self._make_pairs(train_visible, train_infrared)

        gallery_cams = self.indoor_visible_cams if self.eval_mode == 'indoor' else self.visible_cams
        query = self._flatten(self._collect_by_pid(test_ids, self.infrared_cams, modality=1, duplicate=True))
        gallery = self._flatten(self._collect_by_pid(test_ids, gallery_cams, modality=0, duplicate=True))

        if verbose:
            print("=> SYSU-MM01 loaded ({} search)".format(self.eval_mode))
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
            if all(osp.isdir(osp.join(path, 'cam{}'.format(cam))) for cam in range(1, 7)):
                return path
        return candidates[0]

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        for cam in range(1, 7):
            cam_dir = osp.join(self.dataset_dir, 'cam{}'.format(cam))
            if not osp.isdir(cam_dir):
                raise RuntimeError("'{}' is not available".format(cam_dir))
        for name in ('train_id.txt', 'test_id.txt'):
            path = osp.join(self.exp_dir, name)
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    @staticmethod
    def _read_ids(path):
        with open(path, 'r') as f:
            text = f.read()
        return [int(x) for x in re.findall(r'\d+', text)]

    @staticmethod
    def _pid_dir(pid):
        return '{:04d}'.format(int(pid))

    def _collect_by_pid(self, ids, cams, pid2label=None, modality=0, duplicate=False):
        data = defaultdict(list)
        for pid in ids:
            label = pid2label[pid] if pid2label is not None else pid
            for cam in cams:
                img_dir = osp.join(self.dataset_dir, 'cam{}'.format(cam), self._pid_dir(pid))
                img_paths = sorted(glob.glob(osp.join(img_dir, '*.jpg')))
                for img_path in img_paths:
                    path = [img_path, img_path] if duplicate else img_path
                    data[label].append((path, label, cam - 1, modality))
        return data

    @staticmethod
    def _flatten(grouped):
        data = []
        for pid in sorted(grouped):
            data.extend(grouped[pid])
        return data

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
