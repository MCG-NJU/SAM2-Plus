# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2_plus.modeling.sam.unified_decoder import UnifiedDecoder  # from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
from sam2.modeling.sam2_base import NO_OBJ_SCORE

import copy
from sam2.modeling.sam2_base import SAM2Base
from training.dataset_plus.point.utils import revert_sigmoid


class SAM2Plus(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,


        #----------------- New Parameters -----------------#
        separate_image_encoder: bool = False,
        separate_memory_attention: bool = False,
        separate_memory_encoder: bool = False,

        input_point_task_coord_xy: bool = True,
        input_point_task_gaussian_mask: bool = True,

        unified_decoder_box_head_freeze_bn: bool = False,
        unified_decoder_box_head_inner_dim: int = 256,
        unified_decoder_box_head_pred_masks: bool = False,
        #--------------------------------------------------#
    ):
        ## sam2_plus parameters
        self.task = None

        ## separate parameters
        self.separate_image_encoder = separate_image_encoder
        self.separate_memory_attention = separate_memory_attention
        self.separate_memory_encoder = separate_memory_encoder
        ## point input parameters
        self.input_point_task_coord_xy = input_point_task_coord_xy
        self.input_point_task_gaussian_mask = input_point_task_gaussian_mask
        assert self.input_point_task_coord_xy or self.input_point_task_gaussian_mask, "At least one of the point input task should be True"
        ## unified_decoder box_head parameters
        self.unified_decoder_box_head_inner_dim = unified_decoder_box_head_inner_dim
        self.unified_decoder_box_head_freeze_bn = unified_decoder_box_head_freeze_bn
        self.unified_decoder_box_head_pred_masks = unified_decoder_box_head_pred_masks

        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            num_maskmem=num_maskmem,
            image_size=image_size,
            backbone_stride=backbone_stride,
            sigmoid_scale_for_mem_enc=sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc=sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc=binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam=use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn=max_cond_frames_in_attn,
            directly_add_no_mem_embed=directly_add_no_mem_embed,
            use_high_res_features_in_sam=use_high_res_features_in_sam,
            multimask_output_in_sam=multimask_output_in_sam,
            multimask_min_pt_num=multimask_min_pt_num,
            multimask_max_pt_num=multimask_max_pt_num,
            multimask_output_for_tracking=multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval=memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc=non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder=use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs=add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs=proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs=use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval=only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            fixed_no_obj_ptr=fixed_no_obj_ptr,
            soft_no_obj_ptr=soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj=use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial=no_obj_embed_spatial,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            compile_image_encoder=compile_image_encoder,
        )

        #----------------- separate memory attention and memory encoder -----------------#
        if self.separate_image_encoder:
            self.image_encoder_mask = copy.deepcopy(self.image_encoder)
            self.image_encoder_point = copy.deepcopy(self.image_encoder)
            self.image_encoder_box = copy.deepcopy(self.image_encoder)
            del self.image_encoder
        if self.separate_memory_attention:
            self.memory_attention_mask = copy.deepcopy(self.memory_attention)
            self.memory_attention_point = copy.deepcopy(self.memory_attention)
            self.memory_attention_box = copy.deepcopy(self.memory_attention)
            del self.memory_attention
        if self.separate_memory_encoder:
            self.memory_encoder_mask = copy.deepcopy(self.memory_encoder)
            self.memory_encoder_point = copy.deepcopy(self.memory_encoder)
            self.memory_encoder_box = copy.deepcopy(self.memory_encoder)
            del self.memory_encoder

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    """
    Overwrite sam2/modeling/sam2_base.py:SAM2Base._build_sam_heads()
    1) Change self.sam_mask_decoder = MaskDecoder -> self.sam_unified_decoder = UnifiedDecoder
        Add a new head for point tracking and a new head for single object tracking
        More input: image_embedding_size, input_image_size, unified_decoder_box_head_freeze_bn, unified_decoder_box_head_inner_dim, unified_decoder_box_head_pred_masks
    """
    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_unified_decoder = UnifiedDecoder(  # self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,

            #----------------- New Parameters -----------------#
            sam_image_embedding_size=self.sam_image_embedding_size,
            image_size=self.image_size,
            unified_decoder_box_head_freeze_bn=self.unified_decoder_box_head_freeze_bn,
            unified_decoder_box_head_inner_dim=self.unified_decoder_box_head_inner_dim,
            unified_decoder_box_head_pred_masks=self.unified_decoder_box_head_pred_masks,
            #--------------------------------------------------#

            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    """
    Overwrite sam2/modeling/sam2_base.py: SAM2Base._forward_sam_heads()
    1) Change self.sam_mask_decoder -> self.sam_unified_decoder
    2) more input self.task to self.sam_unified_decoder to route task, and get more output: 'boxes_xyxy_norm_multiboxes'
    3) handle box output: set 0 if is_obj_appearing, unnorm, choose
    4) More return output: boxes_xyxy_norm_multiboxes, boxes_xyxy_multiboxes, boxes_xyxy_norm, boxes_xyxy
    """
    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - boxes_xyxy_norm_multiboxes: [B, M, 4]
        - boxes_xyxy_multiboxes: [B, M, 4]
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - boxes_xyxy_norm: [B, 1, 4]
        - boxes_xyxy: [B, 1, 4]
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        - object_score_logits: [B, 1] shape
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            boxes_xyxy_norm_multiboxes,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_unified_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
            task=self.task
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )
            boxes_xyxy_norm_multiboxes = torch.where(
                is_obj_appearing[:, None],
                boxes_xyxy_norm_multiboxes,
                0,
            ) if boxes_xyxy_norm_multiboxes is not None else None

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # convert normalized box to image space
        boxes_xyxy_multiboxes = (boxes_xyxy_norm_multiboxes * self.image_size) if boxes_xyxy_norm_multiboxes is not None else None

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            boxes_xyxy_norm = boxes_xyxy_norm_multiboxes[batch_inds, best_iou_inds].unsqueeze(1) if boxes_xyxy_norm_multiboxes is not None else None
            boxes_xyxy = boxes_xyxy_multiboxes[batch_inds, best_iou_inds].unsqueeze(1) if boxes_xyxy_multiboxes is not None else None
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
            boxes_xyxy_norm, boxes_xyxy = boxes_xyxy_norm_multiboxes, boxes_xyxy_multiboxes

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            boxes_xyxy_norm_multiboxes,
            boxes_xyxy_multiboxes,
            ious,
            low_res_masks,
            high_res_masks,
            boxes_xyxy_norm,
            boxes_xyxy,
            obj_ptr,
            object_score_logits,
        )

    """
    Overwrite sam2/modeling/sam2_base.py: SAM2Base._use_mask_as_output()
    1) Handle more return dummy value from in the unified_decoder.
    2) Return dummy values
    """
    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            None,   # boxes_xyxy_norm
            None,   # boxes_xyxy,
            ious,
            low_res_masks,
            high_res_masks,
            None,   # boxes_xyxy_norm,
            None,   # boxes_xyxy,
            obj_ptr,
            object_score_logits,
        )

    """
    Overwrite sam2/modeling/sam2_base.py: SAM2Base._use_mask_as_output()
    1) Due to 0~1 mask input, revert_sigmoid() is used to get high_res_masks, instead of mask_inputs_float * out_scale + out_bias.
    2) ** When use_obj_ptrs_in_encoder, passing _forward_sam_heads() with point_inputs beyond mask_inputs. **
    3) Handle more return dummy value from in the unified_decoder.
    4) Return dummy values
    """
    def _use_point_as_output(self, backbone_features, high_res_features, mask_inputs, point_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = revert_sigmoid(mask_inputs_float)  # high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                point_inputs=point_inputs if self.input_point_task_coord_xy else None,
                mask_inputs=self.mask_downsample(mask_inputs_float) if self.input_point_task_gaussian_mask else None,
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            None,   # boxes_xyxy_norm
            None,   # boxes_xyxy,
            ious,
            low_res_masks,
            high_res_masks,
            None,   # boxes_xyxy_norm,
            None,   # boxes_xyxy,
            obj_ptr,
            object_score_logits,
        )

    """
    Overwrite sam2/modeling/sam2_base.py: SAM2Base._use_mask_as_output()
    1) ** bulid box outputs from point inputs **
    2) ** When use_obj_ptrs_in_encoder, passing _forward_sam_heads() with point_inputs instead of mask_inputs. **
    3) Handle more return dummy value from in the unified_decoder.
    4) Return dummy values
    """
    def _use_box_as_output(self, backbone_features, high_res_features, mask_inputs, point_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # build box outputs from point inputs
        boxes_xyxy = point_inputs["point_coords"].clone()
        boxes_xyxy = boxes_xyxy.view(boxes_xyxy.size(0), 4)   # [Bs_obj, 4]
        boxes_xyxy_norm = boxes_xyxy / self.image_size
        boxes_xyxy_norm, boxes_xyxy = boxes_xyxy_norm.unsqueeze(1), boxes_xyxy.unsqueeze(1)  # [Bs_obj, 4] -> [Bs_obj, 1, 4]

        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the point input (box)
            _, _, _, _, _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                point_inputs=point_inputs,
                # mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            boxes_xyxy_norm,
            boxes_xyxy,
            ious,
            low_res_masks,
            high_res_masks,
            boxes_xyxy_norm,
            boxes_xyxy,
            obj_ptr,
            object_score_logits,
        )

    """
    Overwrite sam2/modeling/sam2_base.py: SAM2Base.forward_image()
    1) Route task self.sam_mask_decoder.conv_s0 and self.sam_mask_decoder.conv_s1
    """
    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        if not self.separate_image_encoder:
            image_encoder = self.image_encoder
        else:
            if self.task == "mask":
                image_encoder = self.image_encoder_mask
            elif self.task == "point":
                image_encoder = self.image_encoder_point
            elif self.task == "box":
                image_encoder = self.image_encoder_box
            else:
                raise ValueError(f"Unknown task: {self.task}")
        backbone_out = image_encoder(img_batch)
        
        if self.use_high_res_features_in_sam:
            if self.task == "mask":
                conv_s0, conv_s1 = self.sam_unified_decoder.conv_s0_mask, self.sam_unified_decoder.conv_s1_mask
            elif self.task == "point":
                conv_s0, conv_s1 = self.sam_unified_decoder.conv_s0_point, self.sam_unified_decoder.conv_s1_point
            elif self.task == "box":
                conv_s0, conv_s1 = self.sam_unified_decoder.conv_s0_box, self.sam_unified_decoder.conv_s1_box
            else:
                raise ValueError(f"Unknown task: {self.task}")
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = conv_s0(    # self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = conv_s1(    # self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    '''
    1) Route [mask, point, box] in memory_attention
    '''
    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        if not self.separate_memory_attention:
            memory_attention = self.memory_attention
        else:
            if self.task == "mask":
                memory_attention = self.memory_attention_mask
            elif self.task == "point":
                memory_attention = self.memory_attention_point
            elif self.task == "box":
                memory_attention = self.memory_attention_box
            else:
                raise ValueError(f"Unknown task: {self.task}")
        pix_feat_with_mem = memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    '''
    1) Route [mask, point, box] in memory_encoder
    '''
    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        if not self.separate_memory_encoder:
            memory_encoder = self.memory_encoder
        else:
            if self.task == "mask":
                memory_encoder = self.memory_encoder_mask
            elif self.task == "point":
                memory_encoder = self.memory_encoder_point
            elif self.task == "box":
                memory_encoder = self.memory_encoder_box
            else:
                raise ValueError(f"Unknown task: {self.task}")
        maskmem_out = memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    '''
    Overwrite sam2/modeling/sam2_base.py: SAM2Base._track_step()
    1) when GT is given and use_mask_input_as_output_without_sam=True, directly output the mask input
        but route the task _use_mask_as_output / _use_box_as_output_pred_mask / _use_box_as_output_not_pred_mask / _use_point_as_output
    '''
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if self.task == 'mask' and mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            assert point_inputs is None, "Mask task requires mask_inputs but not point_inputs."
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        elif self.task == 'box' and point_inputs is not None and mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            assert torch.all(point_inputs["point_labels"] > 1), f"clean Box prompt (given mask for first memory), not Point Prompt={point_inputs['point_labels']}"
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_box_as_output(
                pix_feat, high_res_features, mask_inputs=mask_inputs, point_inputs=point_inputs
            )
        elif self.task == 'point' and point_inputs is not None and self.use_mask_input_as_output_without_sam:
            assert mask_inputs is not None, "Point task requires both point_inputs and mask_inputs."
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_point_as_output(
                pix_feat, high_res_features, mask_inputs=mask_inputs, point_inputs=point_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs if not (self.task == 'point' and not self.input_point_task_coord_xy) else None,
                mask_inputs=mask_inputs if not (self.task == 'point' and not self.input_point_task_gaussian_mask) else None,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat
    '''
    1) More output boxes_xyxy, boxes_xyxy_norm from sam_outputs = self.self._track_step() <- _forward_sam_heads(), and add them into current_out
    '''
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            boxes_xyxy_norm,
            boxes_xyxy,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["pred_boxes_xyxy_norm"] = boxes_xyxy_norm
        current_out["pred_boxes_xyxy"] = boxes_xyxy
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    '''
    1) mask task: unchanged
    2) box task: due to we want multimask_output, we only count num_pts from point prompt
    3) point task: follow mask task
    '''
    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        if self.task == "mask":
            num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        elif self.task == "box":    # ** for box task with one box prompt, due to we want multimask_output, we only count num_pts from point prompt **
            if point_inputs is None:
                num_pts = 0
            elif torch.all(point_inputs["point_labels"] > 1): # only box prompt
                assert point_inputs['point_coords'].shape[:2] == point_inputs['point_labels'].shape[:2], f"point_coords and point_labels should have same shape, but {point_inputs['point_coords'].shape} and {point_inputs['point_labels'].shape}"
                assert point_inputs['point_coords'].shape[1] == 2, f"point_coords in Box prompt should have shape [B, 2, 2], but {point_inputs['point_coords'].shape}"
                num_pts = 0 # point_inputs["point_labels"].size(1) - 2
            elif torch.all(point_inputs["point_labels"] < 2): # only point prompt
                assert point_inputs['point_coords'].shape[:2] == point_inputs['point_labels'].shape[:2], f"point_coords and point_labels should have same shape, but {point_inputs['point_coords'].shape} and {point_inputs['point_labels'].shape}"
                num_pts = point_inputs["point_labels"].size(1)
            else:   # both point prompt and box prompt (if use_pt_input, init_cond_frames also be used as frames_to_add_correction_pt, _iter_correct_pt_sampling will add more point prompt)
                assert point_inputs['point_coords'].shape[:2] == point_inputs['point_labels'].shape[:2], f"point_coords and point_labels should have same shape, but {point_inputs['point_coords'].shape} and {point_inputs['point_labels'].shape}"
                num_pts = point_inputs["point_labels"].size(1) - 2
        elif self.task == "point":
            num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output
