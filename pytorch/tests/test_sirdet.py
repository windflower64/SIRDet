"""Compatibility checks for the paper architecture and released checkpoint."""

import hashlib
from copy import deepcopy

import pytest
import torch

from sirdet_common import MODEL, WEIGHTS
import sirdet_common
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.rgbt_guidance import DualModalRefine, LCR, SIM, TargetAwareGuide
from ultralytics.nn.tasks import DetectionModel


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_paper_aliases_preserve_class_identity():
    assert LCR is DualModalRefine
    assert SIM is TargetAwareGuide


def test_original_checkpoint_hash():
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == "10e743adeab01cdadd4e539a5e05adf0e7bd37df3f619c810c7a56a347345c62"


def test_paper_config_exact_state_dict_and_predictions():
    original = YOLO(str(WEIGHTS)).model.float().eval()
    paper = DetectionModel(str(MODEL), ch=4, nc=1, verbose=False).eval()
    paper.load_state_dict(original.state_dict(), strict=True)
    assert sum(p.numel() for p in original.parameters()) == 4_325_460
    assert len([m for m in paper.modules() if isinstance(m, LCR)]) == 8
    assert len([m for m in paper.modules() if isinstance(m, SIM)]) == 4
    assert all(not m.spatial_mode and not m.channel_mode for m in paper.modules() if isinstance(m, SIM))
    image = torch.rand(1, 4, 64, 96)
    with torch.no_grad():
        assert torch.equal(paper(image)[0], original(image)[0])
    fused = deepcopy(original).fuse(verbose=False)
    assert sum(p.numel() for p in fused.parameters()) == 4_314_732


def test_sim_is_scalar_modulation_without_spatial_mask():
    module = SIM(16).eval()
    visible = torch.ones(2, 16, 5, 7)
    ir = torch.rand_like(visible)
    output = module([visible, ir])
    assert torch.equal(output, output[:, :1, :1, :1].expand_as(output))


def test_paper_architecture_training_loss_and_gradients():
    model = DetectionModel(str(MODEL), ch=4, nc=1, verbose=False).train()
    model.args = get_cfg(overrides={"use_wiseiou": True})
    batch = dict(img=torch.rand(2, 4, 64, 64), cls=torch.zeros(2, 1),
                 bboxes=torch.tensor([[.4, .5, .2, .2], [.6, .4, .2, .2]]),
                 batch_idx=torch.tensor([0., 1.]))
    loss, items = model(batch)
    assert items.numel() == 3 and torch.isfinite(loss).all()
    loss.sum().backward()
    guides = [m for m in model.modules() if isinstance(m, SIM)]
    assert all(m.ir_heatmap_head[-1].weight.grad is not None for m in guides)


def test_test_manifest_rejects_changed_visible_labels(tmp_path, monkeypatch):
    import csv
    import yaml
    monkeypatch.setattr(sirdet_common, "BUNDLE", tmp_path)
    (tmp_path / "data").mkdir()
    root = tmp_path / "dataset"
    for relative in ("test/visible/images", "test/infrared/images", "test/visible/labels"):
        (root / relative).mkdir(parents=True)
    (root / "test/visible/images/a.jpg").touch()
    (root / "test/infrared/images/a.jpg").touch()
    label = root / "test/visible/labels/a.txt"
    label.write_text("0 .5 .5 .1 .1\n", encoding="utf-8")
    with (tmp_path / "data/test_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("visible", "infrared", "visible_label", "visible_label_sha256", "boxes"))
        writer.writerow(("test/visible/images/a.jpg", "test/infrared/images/a.jpg", "test/visible/labels/a.txt",
                         hashlib.sha256(label.read_bytes()).hexdigest(), 1))
    resolved = tmp_path / "data.yaml"
    resolved.write_text(yaml.safe_dump(dict(path=str(root), test="test/visible/images")), encoding="utf-8")
    assert sirdet_common.verify_test_collection(resolved)["pairs"] == 1
    label.write_text("0 .6 .5 .1 .1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label missing or changed"):
        sirdet_common.verify_test_collection(resolved)
