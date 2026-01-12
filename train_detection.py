#!/usr/bin/env python3
"""
使用 DINOv3 预训练 backbone 训练目标检测器

该脚本使用冻结的 DINOv3 ViT backbone 作为特征提取器，训练 DETR 检测头。
支持 COCO 格式的数据集。

使用方法：
    单 GPU:
    python train_detection.py --backbone vitb16 --data-path /path/to/coco --output-dir ./output

    多 GPU 分布式训练:
    torchrun --nproc_per_node=8 train_detection.py --backbone vitb16 --data-path /path/to/coco --output-dir ./output
"""

import argparse
import os
import random
import subprocess
from pathlib import Path

import dinov3.distributed as distributed
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import CocoDetection

# Import DINOv3 detection modules
from dinov3.hub.backbones import dinov3_vitb16, dinov3_vits16, dinov3_vitl16, Weights
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.detr import build_model, PostProcess
from dinov3.eval.detection.train import SetCriterion, HungarianMatcher
from dinov3.eval.detection.engine import train_one_epoch, evaluate
from dinov3.eval.detection.util.misc import collate_fn

# COCO category names
COCO_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# VisDrone2019-DET category names (10 classes)
VISDRONE_CATEGORY_NAMES = [
    "__background__",
    "pedestrian",      # 1
    "people",          # 2
    "bicycle",         # 3
    "car",             # 4
    "van",             # 5
    "truck",           # 6
    "tricycle",        # 7
    "awning-tricycle", # 8
    "bus",             # 9
    "motor"            # 10
]


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """Initialize distributed training mode.

    Supports both torchrun/elastic launch and manual setup via environment variables.
    """
    # Check if running with torchrun (RANK and WORLD_SIZE are set)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.dist_url = "env://"
        os.environ["LOCAL_SIZE"] = str(torch.cuda.device_count())

    # Check if running with SLURM
    elif "SLURM_PROCID" in os.environ:
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            "scontrol show hostname {} | head -n1".format(node_list)
        )
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
        os.environ["LOCAL_SIZE"] = str(num_gpus)

        args.dist_url = "env://"
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus

    else:
        # Not using distributed mode
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"

    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    dist.barrier()
    setup_for_distributed(args.rank == 0)


def get_backbone_class(name):
    """Get backbone class by name."""
    backbones = {
        "vits16": dinov3_vits16,
        "vitb16": dinov3_vitb16,
        "vitl16": dinov3_vitl16,
    }
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(backbones.keys())}")
    return backbones[name]


def get_args_parser():
    parser = argparse.ArgumentParser("Train DINOv3 detector", add_help=False)

    # Model
    parser.add_argument("--backbone", type=str, default="vitb16",
                        choices=["vits16", "vitb16", "vitl16"],
                        help="DINOv3 backbone name")
    parser.add_argument("--pretrained-weights", type=str, default="LVD1689M",
                        help="Pre-trained backbone weights (default: LVD1689M)")
    parser.add_argument("--num-queries-one2one", type=int, default=300,
                        help="Number of object queries for one-to-one matching")
    parser.add_argument("--num-queries-one2many", type=int, default=0,
                        help="Number of object queries for one-to-many matching (0 to disable)")
    parser.add_argument("--reparam", action="store_true", default=True,
                        help="Use reparameterized box loss")

    # Dataset
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--dataset", type=str, default="visdrone",
                        choices=["coco", "visdrone"],
                        help="Dataset type (coco or visdrone)")
    parser.add_argument("--ann-file", type=str, default="train_coco.json",
                        help="Annotation file relative to data-path")
    parser.add_argument("--img-folder", type=str, default="images",
                        help="Image folder relative to data-path")
    parser.add_argument("--num-classes", type=int, default=11,
                        help="Number of classes (91 for COCO, 11 for VisDrone)")

    # Training
    parser.add_argument("--epochs", type=int, default=12,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for detector head")
    parser.add_argument("--lr-backbone", type=float, default=0,
                        help="Learning rate for backbone (0 to freeze)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--lr-drop-milestones", type=int, nargs="+", default=[8, 11],
                        help="Milestones for learning rate drop")
    parser.add_argument("--clip-max-norm", type=float, default=0.1,
                        help="Max norm for gradient clipping")
    parser.add_argument("--k-one2many", type=int, default=0,
                        help="k for one-to-many loss (0 to disable hybrid training)")
    parser.add_argument("--lambda-one2many", type=float, default=1.0,
                        help="Weight for one-to-many loss")

    # Loss weights
    parser.add_argument("--cls-loss-coef", type=float, default=2.0,
                        help="Classification loss coefficient")
    parser.add_argument("--bbox-loss-coef", type=float, default=5.0,
                        help="Bounding box L1 loss coefficient")
    parser.add_argument("--giou-loss-coef", type=float, default=2.0,
                        help="GIoU loss coefficient")
    parser.add_argument("--set-cost-class", type=float, default=1.0,
                        help="Cost for class in Hungarian matching")
    parser.add_argument("--set-cost-bbox", type=float, default=5.0,
                        help="Cost for bbox L1 in Hungarian matching")
    parser.add_argument("--set-cost-giou", type=float, default=2.0,
                        help="Cost for GIoU in Hungarian matching")

    # Evaluation
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation only")
    parser.add_argument("--eval-ann-file", type=str,
                        default="val_coco.json",
                        help="Annotation file for evaluation")
    parser.add_argument("--eval-img-folder", type=str, default="images",
                        help="Image folder for evaluation")
    parser.add_argument("--topk", type=int, default=100,
                        help="Top-k predictions for evaluation")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Start epoch")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Number of distributed processes")
    parser.add_argument("--dist-url", type=str, default="env://",
                        help="URL used to set up distributed training")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of current process")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loader workers")

    # Mixed precision
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision training")

    return parser


def build_detection_config(args, backbone):
    """Build detection head configuration."""
    hidden_dim = backbone.embed_dim  # Match backbone embedding dimension

    config = DetectionHeadConfig(
        with_box_refine=True,
        two_stage=True,
        mixed_selection=True,
        look_forward_twice=True,
        k_one2many=6,
        lambda_one2many=1.0,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        reparam=args.reparam,
        position_embedding=PositionEncoding.SINE,
        num_feature_levels=1,
        dec_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        norm_type="pre_norm",
        proposal_feature_levels=4,
        proposal_min_size=50,
        decoder_type="global_rpe_decomp",
        decoder_use_checkpoint=False,
        decoder_rpe_hidden_dim=512,
        decoder_rpe_type="linear",
        layers_to_use=None,
        blocks_to_train=None,
        add_transformer_encoder=True,
        num_encoder_layers=6,
        backbone_use_layernorm=False,
        num_classes=args.num_classes,
        aux_loss=True,
        topk=args.topk,
        hidden_dim=hidden_dim,
        nheads=8,
        n_windows_sqrt=2,  # May need tuning for different backbones
        proposal_in_stride=backbone.patch_size,
        proposal_tgt_strides=[int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)],
    )

    # Set layers_to_use based on backbone depth
    if config.layers_to_use is None:
        config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]

    return config


def create_model(args, device):
    """Create detection model with frozen backbone."""
    # Load backbone
    backbone_class = get_backbone_class(args.backbone)

    # Convert weights string to Weights enum if needed
    if isinstance(args.pretrained_weights, str) and args.pretrained_weights in ["LVD1689M", "SAT493M"]:
        weights = getattr(Weights, args.pretrained_weights)
    else:
        weights = args.pretrained_weights

    backbone = backbone_class(
        pretrained=True,
        weights=weights,
        check_hash=False,
    )
    backbone.eval()

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Build detection config
    config = build_detection_config(args, backbone)

    # Build model
    model = build_model(backbone, config)
    model.to(device)

    # Build postprocessor for evaluation
    postprocessor = PostProcess(config.topk, config.reparam)

    return model, postprocessor


def build_criterion(args):
    """Build loss criterion."""
    # Build matcher
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_bbox_type='reparam' if args.reparam else 'l1',
    )

    # Build weight dict
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }

    # Add auxiliary losses
    if True:  # aux_loss is always True in config
        aux_weight_dict = {}
        for i in range(5):  # dec_layers - 1
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Add one-to-many losses
    if args.num_queries_one2many > 0 or args.k_one2many > 0:
        new_dict = {}
        for key, value in weight_dict.items():
            new_dict[key] = value
            new_dict[key + "_one2many"] = value
        weight_dict = new_dict

    # Build criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        num_classes=args.num_classes - 1,  # Exclude background
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=0.25,
        reparam=args.reparam,
    )

    return criterion


def get_transform(train=True):
    """Get image transforms."""
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((640, 640)),  # Fixed size for training
            T.RandomCrop((640, 640)),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            normalize,
        ])


def convert_to_coco_format(target, image_id=None):
    """Convert target to COCO format."""
    anno = target
    # Filter outcrowded objects (optional)
    if "iscrowd" in anno:
        anno["iscrowd"] = torch.where(anno["iscrowd"] > 0, 1, 0).to(anno["boxes"].dtype)

    # Ensure boxes are in XYXY format
    if "boxes" in anno:
        boxes = anno["boxes"]
        if boxes.shape[-1] == 4:
            # Convert from CXCYWH to XYXY if needed
            if boxes.max() <= 1.0:  # Normalized CXCYWH
                h, w = anno.get("size", (1, 1))
                if isinstance(h, torch.Tensor):
                    h, w = h.item(), w.item()
                boxes = boxes.clone()
                boxes[:, 0] = boxes[:, 0] * w - boxes[:, 2] * w / 2
                boxes[:, 1] = boxes[:, 1] * h - boxes[:, 3] * h / 2
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2] * w
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * h
                anno["boxes"] = boxes

    return anno


class CocoDetectionWithSize(CocoDetection):
    """COCO detection with original image size info."""

    def __init__(self, img_folder, ann_file, transform=None):
        super().__init__(img_folder, ann_file)
        self.transform = transform

    def __getitem__(self, idx):
        # Get PIL image and target first
        img, target = super().__getitem__(idx)

        # Get image size using width and height attributes (before transform)
        w, h = img.width, img.height

        # Convert target to detection format
        target = self._convert_to_detection_target(target, w, h, idx)

        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def _convert_to_detection_target(self, anno, w, h, idx):
        """Convert COCO annotation to detection target format."""
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Filter out invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([self.ids[idx]])

        # Add area and iscrowd
        area = torch.tensor([obj["area"] for obj in anno], dtype=torch.float32)
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=torch.int64
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return target


def build_dataset(args, train=True):
    """Build COCO dataset."""
    # For VisDrone, use the COCO JSON files which reference images in VisDrone2019-DET-train/images
    if train:
        ann_file = os.path.join(args.data_path, args.ann_file)
        # Use VisDrone2019-DET-train/images for VisDrone dataset
        if args.dataset == "visdrone" and os.path.basename(args.data_path) == "visdrone":
            img_folder = os.path.join(args.data_path, "VisDrone2019-DET-train", "images")
        else:
            img_folder = os.path.join(args.data_path, args.img_folder)
    else:
        ann_file = os.path.join(args.data_path, args.eval_ann_file)
        if args.dataset == "visdrone" and os.path.basename(args.data_path) == "visdrone":
            img_folder = os.path.join(args.data_path, "VisDrone2019-DET-val", "images")
        else:
            img_folder = os.path.join(args.data_path, args.eval_img_folder)

    transform = get_transform(train)

    dataset = CocoDetectionWithSize(img_folder, ann_file, transform=transform)

    return dataset


def main(args):
    # Set default num_classes based on dataset type
    if args.num_classes == 11:  # Default value, adjust based on dataset
        if args.dataset == "visdrone":
            args.num_classes = 11  # 10 classes + background
        else:
            args.num_classes = 91  # COCO default

    # Initialize distributed training
    init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = distributed.get_world_size()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    seed = 42 + distributed.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Print info
    if distributed.get_rank() == 0:
        print("=" * 60)
        print("Training DINOv3 Object Detector")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Backbone: {args.backbone}")
        print(f"Pretrained weights: {args.pretrained_weights}")
        print(f"Number of classes: {args.num_classes}")
        print(f"Number of queries (1to1): {args.num_queries_one2one}")
        print(f"Number of queries (1tomany): {args.num_queries_one2many}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Learning rate (backbone): {args.lr_backbone}")
        print(f"FP16: {args.fp16}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

    # Build dataset
    dataset_train = build_dataset(args, train=True)
    dataset_val = build_dataset(args, train=False)

    # Build sampler
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=distributed.get_rank(),
        shuffle=True,
    )
    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=world_size,
        rank=distributed.get_rank(),
        shuffle=False,
    )

    # Build data loader
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model
    model, postprocessor = create_model(args, device)

    # Build criterion
    criterion = build_criterion(args)

    # Build optimizer
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    ]
    if args.lr_backbone > 0:
        param_dicts.append({
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        })

    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Build scheduler
    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=args.lr_drop_milestones,
        gamma=0.1,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Resume from checkpoint
    if args.resume:
        if distributed.get_rank() == 0:
            print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    # Evaluation only
    if args.eval:
        base_ds = None  # Will be created in evaluate function
        stats, _ = evaluate(
            model,
            criterion,
            {"bbox": postprocessor},
            data_loader_val,
            base_ds,
            device,
            output_dir,
            step=args.start_epoch,
            use_wandb=False,
            reparam=args.reparam,
        )
        return

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if distributed.get_rank() == 0:
            print(f"\nEpoch {epoch}/{args.epochs - 1}")

        # Train
        _ = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            lr_scheduler,
            max_norm=args.clip_max_norm,
            k_one2many=args.k_one2many,
            lambda_one2many=args.lambda_one2many,
            use_fp16=args.fp16,
            scaler=scaler,
        )

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoint
        if distributed.get_rank() == 0:
            checkpoint_path = output_dir / f"checkpoint_{epoch:04d}.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Evaluate
        base_ds = None
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            stats, _ = evaluate(
                model,
                criterion,
                {"bbox": postprocessor},
                data_loader_val,
                base_ds,
                device,
                output_dir,
                step=epoch,
                use_wandb=False,
                reparam=args.reparam,
            )
            if distributed.get_rank() == 0 and "coco_eval_bbox" in stats:
                ap = stats["coco_eval_bbox"][0]
                print(f"mAP@0.5: {ap:.4f}")

    if distributed.get_rank() == 0:
        print("\nTraining completed!")

    distributed.destroy_process_group()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
