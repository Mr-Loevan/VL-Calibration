
from typing import Optional, List, Tuple, Dict

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> Tuple[StatefulDataLoader, any]:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    if config.val_datasets is not None and len(config.val_datasets) > 0:
        val_dataloaders = {}
        for val_config in config.val_datasets:
            if val_config.name:
                dataset_name = val_config.name
            else:
                import os
                dataset_name = os.path.splitext(os.path.basename(val_config.files.split("@")[0]))[0]
            
            val_dataset = RLHFDataset(
                data_path=val_config.files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
            )
            
            if config.val_batch_size == -1:
                val_batch_size = len(val_dataset)
            else:
                val_batch_size = config.val_batch_size
            
            val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=False,
            )
            
            val_info = {"dataloader": val_dataloader}
            if val_config.reward_function:
                val_info["reward_function"] = val_config.reward_function
            
            val_dataloaders[dataset_name] = val_info
            print(f"Size of val dataloader '{dataset_name}': {len(val_dataloader)}")
        
        assert len(train_dataloader) >= 1
        for name, info in val_dataloaders.items():
            assert len(info["dataloader"]) >= 1
        print(f"Size of train dataloader: {len(train_dataloader)}")
        print(f"Created {len(val_dataloaders)} validation dataloaders from YAML config")
        
        return train_dataloader, val_dataloaders
    
    elif config.val_files_multi is not None:
        val_files_list = [f.strip() for f in config.val_files_multi.split(",")]
        
        val_reward_funcs = []
        if config.val_reward_functions is not None:
            val_reward_funcs = [f.strip() for f in config.val_reward_functions.split(",")]
        
        if val_reward_funcs and len(val_reward_funcs) != len(val_files_list):
            raise ValueError(f"Number of validation datasets ({len(val_files_list)}) must match "
                           f"number of reward functions ({len(val_reward_funcs)})")
        
        val_dataloaders = {}
        for idx, val_file in enumerate(val_files_list):
            import os
            dataset_name = os.path.splitext(os.path.basename(val_file.split("@")[0]))[0]
            
            val_dataset = RLHFDataset(
                data_path=val_file,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
            )
            
            if config.val_batch_size == -1:
                val_batch_size = len(val_dataset)
            else:
                val_batch_size = config.val_batch_size
            
            val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=False,
            )
            
            val_info = {"dataloader": val_dataloader}
            if val_reward_funcs:
                val_info["reward_function"] = val_reward_funcs[idx]
            
            val_dataloaders[dataset_name] = val_info
            print(f"Size of val dataloader '{dataset_name}': {len(val_dataloader)}")
        
        assert len(train_dataloader) >= 1
        for name, info in val_dataloaders.items():
            assert len(info["dataloader"]) >= 1
        print(f"Size of train dataloader: {len(train_dataloader)}")
        print(f"Created {len(val_dataloaders)} validation dataloaders")
        
        return train_dataloader, val_dataloaders
    
    else:
        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

        if config.val_batch_size == -1:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = config.val_batch_size

        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(train_dataloader) >= 1
        assert len(val_dataloader) >= 1
        print(f"Size of train dataloader: {len(train_dataloader)}")
        print(f"Size of val dataloader: {len(val_dataloader)}")
        return train_dataloader, val_dataloader
