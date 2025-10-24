import torch
import torch.nn as nn
from einops import rearrange
from typing import Any, Mapping
from sam2.modeling.sam2_base import MaskDecoder
from sam2.modeling.sam.mask_decoder import LayerNorm2d, MLP, Optional, List, Tuple
from training.dataset_plus.box.utils import box_to_mask
from sam2_plus.modeling.sam.box_head import Pyramid_Corner_Predictor_multi_box, Pyramid_Corner_Predictor_multi_box_use_high_res_features


class UnifiedDecoder(MaskDecoder):
    """
    Contains mask/box/point decoders.
    """
    def __init__(self,
                 sam_image_embedding_size: int,
                 image_size: int,
                 unified_decoder_box_head_freeze_bn: bool,
                 unified_decoder_box_head_inner_dim: int,
                 unified_decoder_box_head_pred_masks: bool,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # delete original mask head
        del self.output_upscaling
        if self.use_high_res_features:
            del self.conv_s0 
            del self.conv_s1
        del self.output_hypernetworks_mlps

        activation = kwargs.get("activation", nn.GELU)

        self.sam_image_embedding_size = sam_image_embedding_size # the basic feature size
        self.image_size = image_size
        assert self.image_size // 16 == self.sam_image_embedding_size, "The backbone_stride must be 16."
        self.unified_decoder_box_head_freeze_bn = unified_decoder_box_head_freeze_bn
        self.unified_decoder_box_head_inner_dim = unified_decoder_box_head_inner_dim
        self.unified_decoder_box_head_pred_masks = unified_decoder_box_head_pred_masks

        self._construct_mask_head(self.transformer_dim, activation, self.use_high_res_features)
        self._constuct_box_head(self.transformer_dim, activation, self.use_high_res_features)
        self._construct_point_head(self.transformer_dim, activation, self.use_high_res_features)

    def _construct_mask_head(self,
                             transformer_dim: int,
                             activation: nn.Module,
                             use_high_res_features: bool):
        """
        Construct mask heads, modified from sam2/modeling/sam/mask_decoder.py#L65~90:MaskDecoder.__init__()
        """
        self.output_upscaling_mask = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        # self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0_mask = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1_mask = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps_mask = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
    
    def _constuct_box_head(self,
                           transformer_dim: int,
                           activation: nn.Module,
                           use_high_res_features: bool):
        """
        We assume that the input_image_size is [H,W], image_embedding_size is [H//16, W//16], 
        The high resolution features are [H//4, W//4] and [H//8, W//8], 
        If use_high_res_features is True, Pyramid_Corner_Predictor will use the (H//4, W//4) as final score map size.
        Otherwise, Corner_Predictor will use the [H//16, W//16] as final score map size.
        """
        if use_high_res_features:
            feat_sz = self.sam_image_embedding_size * 4
            stride = self.image_size // feat_sz
            assert stride == 4, f"sam_image_embedding_size: {self.sam_image_embedding_size}, image_size: {self.image_size}, feat_sz: {feat_sz}, stride: {stride}"
            self.box_head = Pyramid_Corner_Predictor_multi_box_use_high_res_features(inplanes=transformer_dim, channel=self.unified_decoder_box_head_inner_dim, feat_sz=feat_sz, stride=stride, freeze_bn=self.unified_decoder_box_head_freeze_bn)
        else:
            feat_sz = self.sam_image_embedding_size * 4
            stride = self.image_size // feat_sz
            assert stride == 4, f"sam_image_embedding_size: {self.sam_image_embedding_size}, image_size: {self.image_size}, feat_sz: {feat_sz}, stride: {stride}"
            self.box_head = Pyramid_Corner_Predictor_multi_box(inplanes=transformer_dim, channel=self.unified_decoder_box_head_inner_dim, feat_sz=feat_sz, stride=stride, freeze_bn=self.unified_decoder_box_head_freeze_bn)

        self.output_upscaling_box = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        # self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0_box = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1_box = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps_box = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
    
    def _construct_point_head(self,
                              transformer_dim: int,
                              activation: nn.Module,
                              use_high_res_features: bool):
        """
        Construct point tracking heads, modified from sam2/modeling/sam/mask_decoder.py#L65~90:MaskDecoder.__init__()
        """
        self.output_upscaling_point = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        # self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0_point = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1_point = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps_point = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    """
    Overwrite sam2/modeling/sam/mask_decoder.py:MaskDecoder.forward()
    1) More input parameter: 'task' for different task, and input it to self.predict_masks
    2) input the added parameter 'task' into self.predict_masks and get more output 'box' from it
    3) More return output: 'box'
    """
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        task: str = "mask",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
          task (str): The type of head to use. One of ['mask', 'point', 'box'].

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predicted bounding boxes (normalized xyxy). If task is not 'box', return None.
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, boxes, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            task=task,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            boxes = boxes[:, 1:, :] if boxes is not None else None
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, boxes, iou_pred = self._dynamic_multimask_via_stability(masks, boxes, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            boxes = boxes[:, 0:1, :] if boxes is not None else None
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, boxes, iou_pred, sam_tokens_out, object_score_logits

    """
    Overwrite sam2/modeling/sam/mask_decoder.py:MaskDecoder.predict_masks()
    1) More input parameter: task, from self.forward() for different task
    2) Move MaskDecoder.predict_masks()#L218~234 to self._forward_XX_head_get_XX()
    3) According task to route self._forward_XX_head_get_XX()
    4) More return output: 'box (normalized xyxy)'
    """
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        task: str = "mask",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        if task == "mask":
            masks = self._forward_mask_head_get_normal_mask(src, high_res_features, mask_tokens_out, b, c, h, w)
            boxes = None
        elif task == "point":
            masks = self._forward_point_head_get_guassian_mask(src, high_res_features, mask_tokens_out, b, c, h, w)
            boxes = None
        elif task == "box":
            masks, boxes = self._forward_box_head_get_box_xyxy_norm(src, high_res_features, mask_tokens_out, b, c, h, w)
        else:
            raise ValueError(f"task: {task} not supported.")

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, boxes, iou_pred, mask_tokens_out, object_score_logits

    def _forward_mask_head_get_normal_mask(self,
                                           src,
                                           high_res_features,
                                           mask_tokens_out,
                                           b, c, h, w):
        """
        Modified from sam2/modeling/sam/mask_decoder.py:MaskDecoder.predict_masks()#L218~234: output_upscaling -> output_upscaling_mask, output_hypernetworks_mlps -> output_hypernetworks_mlps_mask
        Input:
            @src:               [B, H//16*W//16, C], the low resolution feature map.
            @high_res_features: [B, C//8, H//4, W//4] and [B, C//4, H//8, W//8], the high resolution feature maps.
            @mask_tokens_out:   [B, num_token, C], the mask tokens.
        Return:
            masks: [B, num_mask_tokens, H//4, W//4], the predicted masks.
        """
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling_mask(src)    # upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling_mask  # dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps_mask[i](mask_tokens_out[:, i, :])    # self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks

    def _forward_point_head_get_guassian_mask(self,
                                              src,
                                              high_res_features,
                                              mask_tokens_out,
                                              b, c, h, w):
        """
        Modified from sam2/modeling/sam/mask_decoder.py:MaskDecoder.predict_masks()#L218~234: output_upscaling -> output_upscaling_point, output_hypernetworks_mlps -> output_hypernetworks_mlps_point
        Input:
            @src:               [B, H//16*W//16, C], the low resolution feature map.
            @high_res_features: [B, C//8, H//4, W//4] and [B, C//4, H//8, W//8], the high resolution feature maps.
            @mask_tokens_out:   [B, num_token, C], the mask tokens.
        Return:
            masks: [B, num_mask_tokens, H//4, W//4], the predicted masks.
        """
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling_point(src)   # upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling_point # dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps_point[i](mask_tokens_out[:, i, :])   # self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks

    def _forward_box_head_get_box_xyxy_norm(self,
                                            src,
                                            high_res_features,
                                            mask_tokens_out,
                                            b, c, h, w):
        """
        Modified from sam2/modeling/sam/mask_decoder.py:MaskDecoder.predict_masks()#L218~234: 
        Get box from box_head.
        Input:
            @src:               [B, H//16*W//16, C], the low resolution feature map.
            @high_res_features: [B, C//8, H//4, W//4] and [B, C//4, H//8, W//8], the high resolution feature maps.
            @mask_tokens_out:   [B, num_token, C], the mask tokens. Here we ignore mask_tokens_out.
        Return:
            masks: [B, num_mask_tokens, H//4, W//4], the predicted/square masks.
            box: [B, num_mask_tokens, 4], the predicted box coordinates (normalized). (x_tl, y_tl, x_br, y_br)
        """
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling_box(src)   # upscaled_embedding = self.output_upscaling(src)

            box_input = src  # [B, inplanes, H_s//16, W_s//16]
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling_box # dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

            feat_s0, feat_s1 = high_res_features
            box_input = [feat_s0, feat_s1, src]  # [B, inplanes//8, H_s//4, W_s//4], [B, inplanes//4, H_s//8, W_s//8], [B, inplanes, H_s//16, W_s//16]

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps_box[i](mask_tokens_out[:, i, :])   # self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        boxes = self.box_head(box_input, hyper_in)  # [B, num_mask_tokens, 4], normalized box coordinates xyxy

        if self.unified_decoder_box_head_pred_masks:
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        else:
            with torch.no_grad():   # box to mask
                boxes_flat = rearrange(boxes, 'B N c -> (B N) c')
                masks_flat = box_to_mask(box=boxes_flat.detach(), 
                                    height=self.image_size//4,
                                    width=self.image_size//4,
                                    target_visible=torch.ones(boxes_flat.size(0), device=boxes_flat.device).bool(),
                                    normalized=True)    # 0 or 1, empty mask if unvalid box

                # scale the mask to -10 or 10
                out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
                masks_flat = masks_flat * out_scale + out_bias  # -10 or 10

                masks = rearrange(masks_flat, '(B N) H W -> B N H W', B=boxes.size(0), N=self.num_mask_tokens)  # [B, num_mask_tokens, H, W], the rectangle mask

        return masks, boxes

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_boxes, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multiboxes = all_boxes[:, 1:, :] if all_boxes is not None else None
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multiboxes = multiboxes[batch_inds, best_scores_inds] if multiboxes is not None else None
        best_multiboxes = best_multiboxes.unsqueeze(1) if best_multiboxes is not None else None
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singleboxes = all_boxes[:, 0:1, :] if all_boxes is not None else None
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        boxes_out = torch.where(
            is_stable[..., None].expand_as(singleboxes),
            singleboxes,
            best_multiboxes,
        ) if all_boxes is not None else None
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, boxes_out, iou_scores_out
