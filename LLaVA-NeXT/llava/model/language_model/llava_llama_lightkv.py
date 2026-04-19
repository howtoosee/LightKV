#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from loguru import logger as eval_logger

from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig


# , LlamaModel, LlamaForCausalLM, GenerationConfig
# from .modeling_llama import LlamaModel, LlamaForCausalLM
from .modeling_llama_lightkv import LlamaLightKVModel, LlamaLightKVForCausalLM

# from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from lmms_eval.mymodels.lightkv import LightKVBase, MergeModuleDict, get_merge_modules


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama_lightkv"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}


class LlavaLlamaLightKVModel(LlavaMetaModel, LlamaLightKVModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaLightKVModel, self).__init__(config)


class LlavaLlamaLightKVForCausalLM(LlamaLightKVForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaLightKVForCausalLM.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_llama_lightkv"
        # config.rope_scaling = None

        self.model = LlavaLlamaLightKVModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        merge_modules: Optional[MergeModuleDict] = None,
        profile: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                merge_modules=merge_modules,
                profile=profile,
            )

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     images: Optional[torch.Tensor] = None,
    #     image_sizes: Optional[torch.Tensor] = None,
    #     modalities: Optional[List[str]] = ["image"],
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
    #     position_ids = kwargs.pop("position_ids", None)
    #     attention_mask = kwargs.pop("attention_mask", None)
    #     if "inputs_embeds" in kwargs:
    #         raise NotImplementedError("`inputs_embeds` is not supported")

    #     if images is not None:
    #         (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
    #     else:
    #         inputs_embeds = self.get_model().embed_tokens(inputs)

    #     return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        until_tokens: Optional[set[int]] = None,
        max_new_tokens: Optional[int] = None,
        use_cache: Optional[bool] = True,
        merge_args: Optional[Dict] = None,
        profile: Optional[bool] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple]]:
        eval_logger.debug(f"Attention impl: {self.model.config._attn_implementation}")

        merge_modules = get_merge_modules(total_layers=self.config.num_hidden_layers, **merge_args)

        past_key_values = None
        img_token_indices = None

        sequences = tuple()
        eval_logger.debug(f"Until tokens: {until_tokens}")

        combined_orig_idxs = None
        source = None

        if images is not None and (not use_cache or len(sequences) == 0):  # multimodal
            (
                inputs,
                position_ids,
                attention_mask,
                _,  # past_key_values
                inputs_embeds,
                _,  # new_labels
                img_token_indices,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                position_ids=position_ids,  # position_ids
                attention_mask=attention_mask,
                past_key_values=None,  # past_key_values
                labels=None,
                images=images,
                modalities=modalities,
                image_sizes=image_sizes,
                output_img_token_indices=True,
            )
        else:  # text only
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # eval_logger.debug(f"Image_sizes: {image_sizes}")
        # eval_logger.debug(f"Inputs_embeds: {inputs_embeds.size()}")

        if merge_modules and img_token_indices is not None:
            for mm in merge_modules.values():
                mm.set_img_indices(img_token_indices)
        # eval_logger.debug(f"Merge modules: {merge_modules}")

        while (len(sequences) == 0 or sequences[-1].item() not in until_tokens) and (max_new_tokens is None or len(sequences) < max_new_tokens):

            merge_modules = merge_modules if len(sequences) == 0 else None

            output = self(
                # input_ids=input_ids,          # input_ids not needed anymore
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
                profile=profile,
                merge_modules=merge_modules,
            )

            # if profile:
            #     model_flops = output["flops"]
            #     model_flops_history += (model_flops, )

            logits = output["logits"][:, -1, :]

            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            sequences += (next_token.cpu(),)
            # eval_logger.debug(f"Token generated: {next_token}")
            # eval_logger.debug(f"Sequences: {sequences}")

            if use_cache is True:
                inputs_embeds = self.model.embed_tokens(next_token)
            else:
                inputs_embeds = torch.cat([inputs_embeds, self.get_model().embed_tokens(next_token.unsqueeze(0).unsqueeze(0))], dim=1)

            past_key_values = output["past_key_values"]

            # import pdb; pdb.set_trace()
            if combined_orig_idxs is None and output["combined_orig_idxs"]:
                combined_orig_idxs = [(x.tolist() if isinstance(x, torch.Tensor) else None) for x in output["combined_orig_idxs"]]

            if output["source"]:
                source = {k: (v.cpu().tolist() if isinstance(v, torch.Tensor) else v) for k, v in output["source"].items()}

            # hidden_states_sequence_lens = [hs.size(1) for hs in output.hidden_states]
            # past_num_tokens_prefill_total += (sum(hidden_states_sequence_lens), )
            # past_num_tokens_prefill_breakdown += (hidden_states_sequence_lens, )
            # past_logits += (output.logits.clone(), )
            # past_hidden_states += (output.hidden_states, )
            # past_attentions += (output.attentions, )

        return dict(
            sequences=torch.hstack(sequences).cpu(),
            combined_orig_idxs=combined_orig_idxs,
            source=source,
            # past_logits=past_logits,
            # past_hidden_states=past_hidden_states,
            # past_attentions=past_attentions,
            past_key_values=past_key_values.to_legacy_cache(),
            # past_key_values_shapes=past_key_values_shapes,
            # past_embeds_shapes=past_embeds_shapes,
            # past_num_tokens_prefill_total=past_num_tokens_prefill_total,
            # past_num_tokens_prefill_breakdown=past_num_tokens_prefill_breakdown,
            # model_flops_history=model_flops_history,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llama_lightkv", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaLightKVForCausalLM)
