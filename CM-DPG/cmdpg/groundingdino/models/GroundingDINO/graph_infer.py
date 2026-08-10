# modified from https://github.com/suprosanna/relationformer/blob/scene_graph/inference.py
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
import copy

from cmdpg.groundingdino.models.GroundingDINO.spatial import compute_spatial_encodings


def build_directed_node_pairs(node_ids, allow_self_relations=False):
    if allow_self_relations:
        return torch.cartesian_prod(node_ids, node_ids)

    pairs = torch.combinations(node_ids)
    if pairs.numel() == 0:
        return pairs

    return torch.cat((pairs, pairs[:, [1, 0]]), 0)


# def graph_infer(outputs : List[Dict],
#                 # rln_proj, rln_classifier,
#                 rln_proj, rln_classifier, spatial_head, spatial_head_ovr,
#                 rln_freq_bias,
#                 text_dict,
#                 name2predicates,
#                 tokenizer,
#                 use_sigmoid=False,
#                 use_classifier=False,
#                 save_features=False):
def graph_infer(outputs: List[Dict],
                rln_proj, rln_classifier, spatial_head, spatial_head_ovr,
                pair_dif_proj,
                pair_sum_proj,
                pair_arg_proj,
                pair_visual_attn,
                pair_visual_norm,
                rln_freq_bias,
                text_dict,
                name2predicates,
                tokenizer,
                use_sigmoid=False,
                use_classifier=False,
                save_features=False,
                allow_self_relations=False):
    dst = []
    if rln_freq_bias is not None:
        use_sigmoid = False

    for batch_id, output in enumerate(outputs):
        obj_token = output['obj_token']  # (#obj, dim)
        rln_token = output['rln_token']  # (#query, dim)

        if not use_classifier and text_dict is not None:
            encoded_text = text_dict['encoded_text'][batch_id]
            text_mask = text_dict['text_token_mask'][batch_id]
            input_ids = text_dict['input_ids'][batch_id]
            sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]

        boxes = copy.deepcopy(output['boxes'])  # (#obj, 4)
        scores = copy.deepcopy(output['scores'])  # (#obj)
        labels = copy.deepcopy(output['labels'])  # (#obj)

        node_id = torch.nonzero(labels).flatten()

        obj_token = obj_token[node_id]
        pred_classes = labels[node_id]
        pred_cls_score = scores[node_id]

        pred_boxes = boxes[node_id]
        pred_boxes_score = pred_cls_score
        pred_boxes_class = pred_classes

        min_nodes = 1 if allow_self_relations else 2
        if node_id.nelement() >= min_nodes:
            # all possible node pairs in all token ordering
            tmp = torch.arange(len(node_id))
            node_pairs = build_directed_node_pairs(
                tmp,
                allow_self_relations=allow_self_relations,
            )

            id_rel = torch.tensor(list(range(len(node_id))))
            node_pairs_rel = build_directed_node_pairs(
                id_rel,
                allow_self_relations=allow_self_relations,
            )

            # Visual branch in G-VRD:
            # ============================================================
            # G-VRD Visual Branch
            #
            # This branch follows Eq. (10). It first constructs pair-wise
            # object interaction features using the difference and summation
            # of subject/object tokens:
            #
            #   V_pair = f_arg(f_dif(V_sub - V_obj) concat f_sum(V_sub + V_obj))
            #
            # Then V_pair is concatenated with the relation query token and
            # projected by rln_proj. The projected pair feature attends to
            # encoder visual memory V_ec to obtain V_pv, which is matched
            # with relation text embeddings for open-vocabulary prediction.
            # ============================================================
            # V_pair = f_arg(f_dif(V_sub - V_obj) concat f_sum(V_sub + V_obj)).
            sub_feat = obj_token[node_pairs[:, 0], :]
            obj_feat = obj_token[node_pairs[:, 1], :]
            pair_dif = pair_dif_proj(sub_feat - obj_feat)
            pair_sum = pair_sum_proj(sub_feat + obj_feat)
            pair_feat = pair_arg_proj(torch.cat((pair_dif, pair_sum), dim=1))

            relation_feat = torch.cat((
                pair_feat,
                rln_token.flatten().repeat(len(node_pairs), 1),
            ),
                dim=1)
            relation_feat = rln_proj(relation_feat)
            # V_pair attends to V_ec, producing V_pv.
            if pair_visual_attn is not None and output.get('enc_memory', None) is not None:
                v_pair = relation_feat

                enc_memory = output['enc_memory']  # [num_enc_tokens, hidden_dim]
                enc_mask = output.get('enc_mask', None)  # [num_enc_tokens]

                query = v_pair.unsqueeze(1)  # [num_pairs, 1, hidden_dim]
                key = enc_memory.unsqueeze(1)  # [num_enc_tokens, 1, hidden_dim]
                value = enc_memory.unsqueeze(1)  # [num_enc_tokens, 1, hidden_dim]

                key_padding_mask = None
                if enc_mask is not None:
                    key_padding_mask = enc_mask.unsqueeze(0)  # [1, num_enc_tokens]

                v_pv, _ = pair_visual_attn(
                    query=query,
                    key=key,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )

                relation_feat = v_pv.squeeze(1)

                if pair_visual_norm is not None:
                    relation_feat = pair_visual_norm(relation_feat)

            # spatial feature
            '''
            Spatial Encoding in Geometric branch of G-VRD Module
            '''
            # ============================================================
            # G-VRD Geometric Branch
            #
            # spatial_head and spatial_head_ovr encode subject-object
            # geometric layouts into relation-aware features. The closed-set
            # branch predicts predicate logits directly, while the open-
            # vocabulary branch projects spatial encodings into the shared
            # text-aligned embedding space.
            # ============================================================
            spatial_l = pred_boxes[node_pairs[:, 0]]
            spatial_r = pred_boxes[node_pairs[:, 1]]
            spatial_feature = compute_spatial_encodings([spatial_l], [spatial_r])

            if use_classifier:
                relation_logits = rln_classifier(relation_feat)
                '''
                MLP in Geometric branch of G-VRD Module
                '''
                spa_logits = spatial_head(spatial_feature)
                relation_logits = 0.7 * relation_logits + 0.3 * spa_logits
                if rln_freq_bias is not None:
                    bias = rln_freq_bias( \
                        torch.stack((pred_classes[node_pairs[:, 0]],
                                     pred_classes[node_pairs[:, 1]]), 1))
                    relation_logits += bias
            else:
                relation_logits = torch.einsum("a d, b d -> a b", relation_feat, encoded_text)
                relation_logits.masked_fill(~text_mask, float('-inf'))
                spatial_feature = spatial_head_ovr(spatial_feature)
                spa_logits = torch.einsum("a d, b d -> a b", spatial_feature, encoded_text)
                spa_logits.masked_fill_(~text_mask, float('-inf'))
                relation_logits = 0.7 * relation_logits + 0.3 * spa_logits

            all_node_pairs = node_pairs_rel.cpu()
            if use_sigmoid:
                relation_prob = relation_logits.sigmoid().detach().cpu()
            else:
                relation_prob = relation_logits.softmax(-1).detach().cpu()

            if use_classifier:
                all_relation = relation_prob
            else:
                all_relation = torch.zeros((relation_prob.shape[0], len(name2predicates)))
                for ii in range(1, len(sep_idx)):
                    right_idx = sep_idx[ii]
                    left_idx = sep_idx[ii - 1] + 1
                    if left_idx >= right_idx:
                        continue
                    name = tokenizer.decode(input_ids[left_idx:right_idx])
                    all_relation[:, name2predicates[name]] = relation_prob[:, left_idx:right_idx].mean(-1)

            # sort by score: relation score * subject score * object score
            rel_score = all_relation[:, 1:].max(1)[0]

            obj_score0 = pred_boxes_score[all_node_pairs[:, 0]]
            obj_score1 = pred_boxes_score[all_node_pairs[:, 1]]
            rel_score = rel_score.to(obj_score0.device) * obj_score0 * obj_score1

            rel_idx = rel_score.sort(descending=True)[1].to(all_relation.device)
            all_relation = all_relation[rel_idx]
            all_node_pairs = all_node_pairs[rel_idx]

        else:
            assert node_id.nelement() == 1, "#obj != 1"

            print("Warning: #obj==1!")
            all_node_pairs = torch.zeros(1, 2).long()
            all_relation = torch.zeros(1, 51)  #
            relation_feat = None
            pred_boxes = pred_boxes.view(1, -1).repeat(2, 1)
            pred_boxes_score = pred_boxes_score.view(1, -1).repeat(2, 1)
            pred_boxes_class = pred_boxes_class.view(1, -1).repeat(2, 1)
            # pred_boxes_score.fill_(0.)

        out = {}
        if all_relation is not None:
            out['node_id'] = node_id.cpu()
            out['pred_boxes'] = pred_boxes.cpu()
            out['pred_boxes_score'] = pred_boxes_score.cpu()
            out['pred_boxes_class'] = pred_boxes_class.cpu()

            out['all_node_pairs'] = all_node_pairs
            out['all_relation'] = all_relation
            if save_features and relation_feat is not None:
                out['rln_features'] = relation_feat.data.cpu()

        dst.append(out)

    return dst
