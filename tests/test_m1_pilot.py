"""Checks for scientific validity of the compact pilot's inference boundary."""
import numpy as np
import torch

from models.pilot import (PilotDetector, proposals, features, select_features, encode,
                          decode, predict_images, load_weights)
from eval.metrics import ap, operating_point


def test_student_inference_is_independent_of_swir():
    rng = np.random.default_rng(42)
    images = dict(rgb=rng.random((64,80,3),dtype=np.float32),
                  lwir=rng.random((64,80,1),dtype=np.float32))
    images["lwir"][20:26,35:45] = 8
    torch.manual_seed(1)
    student = PilotDetector(["rgb","lwir"]).eval()
    before = predict_images(student, images, ["rgb","lwir"])
    images["swir"] = np.full((64,80,1),float("nan"))
    after = predict_images(student, images, ["rgb","lwir"])
    assert before == after


def test_features_selected_for_student_match_direct_extraction():
    rng = np.random.default_rng(9)
    images = {"rgb":rng.random((64,80,3),dtype=np.float32),
              "lwir":rng.random((64,80,1),dtype=np.float32),
              "swir":rng.random((64,80,1),dtype=np.float32)}
    box = np.array([[10,12,25,30],[2,3,8,9]],np.float32)
    direct = features(images,box,["rgb","lwir"])
    teacher = features(images,box,["rgb","lwir","swir"])
    np.testing.assert_array_equal(direct,select_features(teacher,["rgb","lwir","swir"],["rgb","lwir"]))


def test_box_round_trip():
    proposals_ = np.array([[5,10,30,40],[120,130,130,134]],np.float32)
    labels = np.array([[7,8,25,38],[121,128,133,135]],np.float32)
    np.testing.assert_allclose(decode(proposals_,encode(proposals_,labels)),labels,atol=1e-5)


def test_flat_image_can_return_no_detection():
    images = {"lwir":np.ones((64,80,1),np.float32)*0.5}
    bx,sc = proposals(images,["lwir"])
    assert bx.shape == (0,4) and len(sc)==0
    assert predict_images(PilotDetector(["lwir"]).eval(),images,["lwir"])["boxes"] == []


def test_saved_weights_reproduce_predictions(tmp_path):
    torch.manual_seed(3)
    model = PilotDetector(["lwir"]).eval()
    path = tmp_path/"best.pt"
    torch.save(dict(architecture="m1-proposal-mlp-v1",bands=["lwir"],hidden=64,state_dict=model.state_dict()),path)
    restored,_ = load_weights(path)
    x = torch.randn(12,76)
    assert all(torch.equal(a,b) for a,b in zip(model(x),restored(x)))


def test_duplicate_and_bad_localization_are_false_positives():
    r = dict(gts=[dict(box=[10,10,20,20],label=0,size_bin="8-16")],
             preds=dict(boxes=[[10,10,20,20],[10,10,20,20],[30,30,40,40]],scores=[.9,.8,.7],labels=[0,0,0]))
    cfg = dict(score_thresh=.5,negative_match_iou=.1)
    assert ap([r],cfg=cfg)==1.0
    op = operating_point([r],cfg)
    assert op["tp"]==1 and op["fp"]==2 and op["precision"]==1/3


def test_coco_export_supports_swir_only_data(tmp_path):
    import json
    from data.coco_export import export
    meta=dict(scene_id="s000024",split="val",camera=dict(width=640,height=512,hfov_deg=8.78),
              conditions={},modalities=dict(swir=dict(png="swir/s000024.png",layers="layers/swir/s000024.npz")),
              extra=dict(objects=[]))
    (tmp_path/"meta").mkdir()
    (tmp_path/"meta/s000024.json").write_text(json.dumps(meta))
    assert export(str(tmp_path)) == {"val":(1,0)}
    result=json.loads((tmp_path/"coco/val.json").read_text())
    assert result["images"][0]["file_name"]=="swir/s000024.png"
    assert result["images"][0]["fyp_preview_band"]=="swir"
